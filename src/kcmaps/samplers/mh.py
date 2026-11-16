"""Metropolis-adjusted chains: one KC-map proposal of fixed lag as a generalized-MH proposal.

A transition proposes x' = x + displacement(x, eps) and accepts with probability
min(1, exp(-dU - dK) q(R x' -> R x) / q(x -> x')), R reversing momenta; a rejection keeps x with reversed momenta.
The random key of a transition is a function of the seed, the stream (initial momentum, proposal noise, acceptance),
the global chain index and the absolute step only, so batch and scan sizes never change a chain.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from ..core import constants as C
from ..core.utils import directory_lock
from ..models.checkpoint import LoadedModel
from ..models.geometry import flip_momentum
from . import chunks

DIAGNOSTICS = ("accepted", "log_uniform", "potential_delta", "kinetic_delta", "log_proposal_ratio",
               "log_acceptance_ratio")


def _chain_keys(seed, chain_indices, stream):
    stream_key = jax.random.fold_in(jax.random.PRNGKey(seed), stream)
    return jax.vmap(lambda index: jax.random.fold_in(stream_key, index))(jnp.asarray(chain_indices, jnp.uint32))


def _step_keys(seed, chain_indices, stream, step):
    step = jnp.asarray(step, jnp.uint32)
    return jax.vmap(lambda key: jax.random.fold_in(key, step))(_chain_keys(seed, chain_indices, stream))


def _kinetic_energy(kcm, state):
    masses = jnp.asarray(kcm.masses, state.dtype)
    kbt = jnp.asarray(kcm.kbt_kj_per_mol, state.dtype)
    return 0.5 * jnp.sum(state[..., 3:]**2 / (masses * kbt), axis=(-2, -1))


def _initialize(kcm, sources, seed, chain_indices):
    """Centered sources (B, N, 3) -> (states with Maxwell momenta, U/kBT)."""
    keys = _chain_keys(seed, chain_indices, C.MH_INITIAL_MOMENTUM_STREAM)
    momentum = jax.vmap(lambda key: kcm.maxwell_momenta_phys(key, sources.shape[1:]))(keys)
    state = jnp.concatenate([sources, momentum.astype(sources.dtype)], axis=-1)
    return state, kcm.system.energy(state[..., :3])


def _run(kcm, params, state, potential, tau_ps, seed, step_start, chain_indices, length, *, record):
    """``length`` MH transitions of a batch; returns (state, potential, states or None, *DIAGNOSTICS)."""
    def forward(source, eps):
        # The Gaussian normalizer and the fixed-lag base log-det cancel against the reverse direction.
        displacement, logdet = kcm.propose(params, source, tau_ps, eps)
        return source + displacement, -0.5 * jnp.sum(eps**2) - logdet

    def reverse(source, destination):
        eps, logdet = kcm.proposal_inverse(params, source, tau_ps, destination - source)
        return -0.5 * jnp.sum(eps**2) + logdet

    def step(carry, offset):
        state, current = carry
        absolute = step_start + jnp.asarray(offset, jnp.uint32)
        proposal_keys = _step_keys(seed, chain_indices, C.MH_PROPOSAL_NOISE_STREAM, absolute)
        acceptance_keys = _step_keys(seed, chain_indices, C.MH_ACCEPTANCE_STREAM, absolute)
        eps = jax.vmap(lambda key: jax.random.normal(key, state.shape[1:], state.dtype))(proposal_keys)
        candidate, forward_weight = jax.vmap(forward)(state, eps)
        reversed_state = flip_momentum(state)
        reverse_weight = jax.vmap(reverse)(flip_momentum(candidate), reversed_state)
        candidate_potential = kcm.system.energy(candidate[..., :3])
        potential_delta = candidate_potential - current
        kinetic_delta = _kinetic_energy(kcm, candidate) - _kinetic_energy(kcm, state)
        log_proposal_ratio = reverse_weight - forward_weight
        log_ratio = -potential_delta - kinetic_delta + log_proposal_ratio
        log_uniform = -jax.vmap(lambda key: jax.random.exponential(key, dtype=state.dtype))(acceptance_keys)
        finite = (jnp.all(jnp.isfinite(candidate), axis=(-2, -1)) & jnp.isfinite(current)
                  & jnp.isfinite(candidate_potential) & jnp.isfinite(forward_weight) & jnp.isfinite(reverse_weight)
                  & jnp.isfinite(potential_delta) & jnp.isfinite(kinetic_delta) & jnp.isfinite(log_proposal_ratio)
                  & jnp.isfinite(log_ratio) & jnp.isfinite(log_uniform))
        # A non-finite transition is rejected and logged with a NaN ratio.
        log_ratio = jnp.where(finite, log_ratio, jnp.asarray(jnp.nan, state.dtype))
        # log_uniform is the log of a (0, 1] variate, so <= also accepts every ratio >= 1.
        accepted = finite & (log_uniform <= jnp.minimum(jnp.zeros_like(log_ratio), log_ratio))
        new_state = jnp.where(accepted[:, None, None], kcm.recenter_state(candidate), reversed_state)
        new_potential = jnp.where(accepted, candidate_potential, current)
        return (new_state, new_potential), (new_state if record else None, accepted, log_uniform, potential_delta,
                                            kinetic_delta, log_proposal_ratio, log_ratio)

    (state, potential), outputs = jax.lax.scan(step, (state, potential), jnp.arange(length, dtype=jnp.uint32))
    return state, potential, *outputs


def run_mh(model: LoadedModel, output, *, sources=None, tau_fs: float, walkers_per_source: int,
           burn_in_steps: int, steps: int, batch_size: int, step_chunk: int, seed: int) -> Path:
    """Run Metropolis-adjusted chains and write ``mh.npz``, ``run_spec.json`` and ``summary.json`` to ``output``.

    Chain c = s * walkers_per_source + w starts from source s (centered in float32) with Maxwell momenta. Runs with
    JAX x64 off. Settings are range-checked by ``config.validate(cfg, "simulate")``; this function checks only what
    needs the model or the sources.

    Args:
        model: a ``LoadedModel``; ``model.kcm`` is used as given, with its force cap (``load_checkpoint`` gives
            none), which ``run_spec.json`` records.
        output: output directory (created; must be empty).
        sources: (S, N, 3) nm in PDB order; None = the start configuration ``model.start_position`` alone.
        tau_fs: proposal lag (fs), inside the training band.
        walkers_per_source: chains per source.
        burn_in_steps: transitions before the retained ones (diagnostics kept, states not).
        steps: retained transitions.
        batch_size: chains per device batch (the last batch is padded).
        step_chunk: transitions per compiled scan.
        seed: base seed of the three streams.

    ``mh.npz`` holds ``sources`` (S, N, 3), ``initial_phase`` (C, N, 6), ``phase`` (C, steps, N, 6) and the
    diagnostics (C, burn_in_steps + steps): accepted, log_uniform, potential_delta, kinetic_delta (kBT),
    log_proposal_ratio, log_acceptance_ratio.

    Returns:
        The output directory.

    Raises:
        ValueError: bad sources, a lag outside the training band, or a non-finite initial state.
    """
    kcm = model.kcm
    sources = np.asarray(model.start_position[None] if sources is None else sources)
    n_atoms = kcm.masses.shape[0]
    if sources.ndim != 3 or sources.shape[0] == 0 or sources.shape[1:] != (n_atoms, 3):
        raise ValueError(f"sources must have shape (S, {n_atoms}, 3); got {sources.shape}")
    if not np.issubdtype(sources.dtype, np.floating) or not np.isfinite(sources).all():
        raise ValueError("sources must be finite floating point")
    tau_ps = tau_fs / 1000
    if not kcm.train_tau_min_ps <= tau_ps <= kcm.train_tau_max_ps:
        raise ValueError(f"tau_fs {tau_fs} is outside the training band "
                         f"[{kcm.train_tau_min_ps}, {kcm.train_tau_max_ps}] ps")
    total_chains = sources.shape[0] * walkers_per_source
    if total_chains > 2**32:
        raise ValueError("chain indices must fit uint32")
    if str(jax.config.jax_default_prng_impl) != "threefry2x32":
        raise ValueError("MH chains need the Threefry PRNG")
    total_steps = burn_in_steps + steps
    regions = [(start, min(step_chunk, burn_in_steps - start), False) for start in range(0, burn_in_steps, step_chunk)]
    regions += [(burn_in_steps + start, min(step_chunk, steps - start), True) for start in range(0, steps, step_chunk)]
    output = Path(output).expanduser().resolve()
    with directory_lock(output):
        chunks.check_output(output, resume=False)
        started = time.perf_counter()
        with jax.enable_x64(False):
            params = model.params
            tau = jnp.asarray(tau_ps, jnp.float32)
            initialize = jax.jit(lambda batch, indices: _initialize(kcm, batch, seed, indices))
            runners = {}
            for _, length, record in regions:
                if (length, record) not in runners:
                    runners[length, record] = jax.jit(
                        lambda p, s, u, start, indices, length=length, record=record:
                        _run(kcm, p, s, u, tau, seed, start, indices, length, record=record))
            initial = np.empty((total_chains, n_atoms, 6), np.float32)
            phase = np.empty((total_chains, steps, n_atoms, 6), np.float32)
            diagnostics = {name: np.empty((total_chains, total_steps), bool if name == "accepted" else np.float32)
                           for name in DIAGNOSTICS}
            for start in range(0, total_chains, batch_size):
                stop = min(start + batch_size, total_chains)
                count = stop - start
                indices = np.empty(batch_size, np.uint32)
                indices[:count] = np.arange(start, stop, dtype=np.uint32)
                indices[count:] = indices[count - 1]
                batch_sources = sources[indices.astype(np.uint64) // walkers_per_source]
                centered = np.asarray(kcm.center_positions(jnp.asarray(batch_sources, jnp.float32)))
                state, potential = initialize(jax.device_put(centered), jax.device_put(indices))
                initial[start:stop] = np.asarray(state)[:count]
                if not (np.isfinite(initial[start:stop]).all() and np.isfinite(np.asarray(potential)[:count]).all()):
                    raise ValueError(f"non-finite initial state for chains [{start}, {stop})")
                staged_indices = jax.device_put(indices)
                for step_start, length, record in regions:
                    state, potential, states, *values = runners[length, record](
                        params, state, potential, jax.device_put(np.asarray(step_start, np.uint32)), staged_indices)
                    for name, value in zip(DIAGNOSTICS, values):
                        diagnostics[name][start:stop, step_start:step_start + length] = np.asarray(value)[:, :count].T
                    if record:
                        offset = step_start - burn_in_steps
                        phase[start:stop, offset:offset + length] = np.transpose(np.asarray(states)[:, :count],
                                                                                 (1, 0, 2, 3))
        seconds = time.perf_counter() - started

        np.savez(output / "mh.npz", sources=sources, initial_phase=initial, phase=phase, **diagnostics)
        spec = dict(checkpoint=dict(path=str(model.path), step=model.step, ema=model.ema),
                    system=str(model.config.system.name), tau_fs=float(tau_fs),
                    walkers_per_source=walkers_per_source, sources=int(sources.shape[0]), burn_in_steps=burn_in_steps,
                    steps=steps, batch_size=batch_size, step_chunk=step_chunk, seed=seed,
                    streams=dict(initial_momentum=C.MH_INITIAL_MOMENTUM_STREAM,
                                 proposal_noise=C.MH_PROPOSAL_NOISE_STREAM, acceptance=C.MH_ACCEPTANCE_STREAM))
        (output / "run_spec.json").write_text(json.dumps(spec, indent=2) + "\n")
        accepted = diagnostics["accepted"]
        summary = dict(chains=total_chains, acceptance=float(accepted.mean()),
                       burn_in_acceptance=float(accepted[:, :burn_in_steps].mean()) if burn_in_steps else None,
                       retained_acceptance=float(accepted[:, burn_in_steps:].mean()),
                       per_chain_acceptance=accepted[:, burn_in_steps:].mean(axis=1).tolist(), wall_seconds=seconds)
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps({k: summary[k] for k in ("chains", "acceptance", "retained_acceptance", "wall_seconds")}),
              flush=True)
    return output


__all__ = ["run_mh"]
