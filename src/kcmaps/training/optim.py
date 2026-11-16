"""Training state, optimizer, learning-rate schedule and the compiled data-parallel train step."""

from __future__ import annotations

import copy
import functools
from typing import Any, Dict

import jax
import jax.numpy as jnp
import optax
from flax import struct
from flax.training import train_state
from jax.sharding import PartitionSpec as P

from ..core.config import ema_key
from .losses import scalar_metric


class EMATrainState(train_state.TrainState):
    """``TrainState`` with ``ema_params``: one parameter copy per EMA decay, keyed by ``ema_key(decay)``."""

    ema_params: Dict[str, Any] = struct.field(default_factory=dict)


def make_schedule(optim_cfg, total_steps: int):
    """Linear warmup over ``round(warmup_fraction * total_steps)`` steps to ``lr``, cosine decay to ``end_lr`` at
    ``total_steps``."""
    return optax.warmup_cosine_decay_schedule(
        init_value=0.0, peak_value=float(optim_cfg.lr),
        warmup_steps=round(optim_cfg.warmup_fraction * total_steps), decay_steps=int(total_steps),
        end_value=float(optim_cfg.end_lr))


def make_optimizer(optim_cfg, total_steps: int):
    """``(tx, schedule)``: global-norm clipping, then AdamW."""
    schedule = make_schedule(optim_cfg, total_steps)
    tx = optax.chain(
        optax.clip_by_global_norm(float(optim_cfg.clip_norm)),
        optax.adamw(learning_rate=schedule, b1=float(optim_cfg.b1), b2=float(optim_cfg.b2),
                    eps=float(optim_cfg.eps), weight_decay=float(optim_cfg.weight_decay)),
    )
    return tx, schedule


def create_state(apply_fn, params, tx, ema_decays) -> EMATrainState:
    """Fresh state at step 0 (int32, as the train step stores it) with every EMA copy equal to ``params``."""
    ema = {ema_key(decay): copy.deepcopy(params) for decay in ema_decays}
    state = EMATrainState.create(apply_fn=apply_fn, params=params, ema_params=ema, tx=tx)
    return state.replace(step=jnp.zeros((), jnp.int32))


def make_train_step(mesh, schedule, loss_fn, *, target_decay: float):
    """A jitted, data-parallel ``train_step(state, inputs) -> (state, metrics)`` of ``loss_fn``.

    The state is replicated and donated; every ``inputs`` leaf is sharded on its leading batch axis over the mesh axis
    ``'data'``. The EMA copy ``target_decay`` holds the loss's target parameters. A step whose gradient norm is not
    finite is reverted (parameters and optimizer state); EMAs then advance toward the kept parameters.

    ``metrics`` holds the ``loss_fn`` metrics summed over devices plus ``grad_norm`` (pre-clip), ``lr`` and
    ``skipped_steps`` (1 when the update was reverted), all as float32 ``(sum, count)`` pairs.
    """
    target_key = ema_key(target_decay)
    n_shards = int(mesh.shape["data"])

    def updated_ema(state):
        return {key: jax.tree_util.tree_map(lambda p, ep, f=float(key): f * ep + (1.0 - f) * p, state.params, tree)
                for key, tree in state.ema_params.items()}

    @functools.partial(jax.jit, donate_argnums=(0,))
    def train_step(state, inputs):
        state_specs = jax.tree_util.tree_map(lambda _: P(), state)
        input_specs = jax.tree_util.tree_map(lambda _: P("data"), inputs)

        @functools.partial(jax.shard_map, mesh=mesh, in_specs=(state_specs, input_specs),
                           out_specs=(state_specs, P()), check_vma=True)
        def step(state, inputs):
            target_params = state.ema_params[target_key]

            # Autodiff sums the gradients of replicated parameters over devices, so the loss is divided first.
            def scaled(params, inputs):
                value, metrics = loss_fn(params, target_params, state.step, inputs)
                return value / n_shards, metrics

            (_, metrics), grads = jax.value_and_grad(scaled, has_aux=True)(state.params, inputs)
            metrics = jax.lax.psum(metrics, axis_name="data")
            norm = optax.tree_utils.tree_norm(grads)
            lr = schedule(state.step)
            previous_params, previous_opt_state = state.params, state.opt_state
            state = state.apply_gradients(grads=grads)
            keep = jnp.isfinite(norm)
            select = functools.partial(jnp.where, keep)
            # Reverting the optimizer state also reverts the schedule count, so a skipped step uses no schedule tick.
            state = state.replace(params=jax.tree_util.tree_map(select, state.params, previous_params),
                                  opt_state=jax.tree_util.tree_map(select, state.opt_state, previous_opt_state))
            metrics.update(grad_norm=scalar_metric(norm), lr=scalar_metric(lr), skipped_steps=scalar_metric(~keep))
            state = state.replace(ema_params=updated_ema(state))
            return state, metrics

        return step(state, inputs)

    return train_step


def make_eval_step(mesh, loss_fn, *, target_decay: float):
    """A jitted, data-parallel ``eval_step(state, inputs) -> metrics`` of ``loss_fn`` without gradients or state
    update.

    The current parameters are the model parameters and the EMA copy ``target_decay`` the target parameters, as in
    ``make_train_step``; ``metrics`` are the ``loss_fn`` metrics summed over devices.
    """
    target_key = ema_key(target_decay)

    @jax.jit
    def eval_step(state, inputs):
        state_specs = jax.tree_util.tree_map(lambda _: P(), state)
        input_specs = jax.tree_util.tree_map(lambda _: P("data"), inputs)

        @functools.partial(jax.shard_map, mesh=mesh, in_specs=(state_specs, input_specs), out_specs=P(),
                           check_vma=True)
        def evaluate(state, inputs):
            _, metrics = loss_fn(state.params, state.ema_params[target_key], state.step, inputs)
            return jax.lax.psum(metrics, axis_name="data")

        return evaluate(state, inputs)

    return eval_step
