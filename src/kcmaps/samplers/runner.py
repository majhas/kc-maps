"""``run_chains``: run adaptive chains chunk by chunk into a restartable simulation directory (layout in ``chunks``)."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf

from ..core import constants as C
from ..core.config import AdaptiveClock, adaptive_clock
from ..core.utils import directory_lock
from ..models.checkpoint import LoadedModel
from . import chunks
from .adaptive import WIDE_FIELDS, Carry, initial_carry, make_attempt, make_chunk


def _host(carry: Carry) -> dict:
    return {key: np.asarray(value) for key, value in jax.device_get(carry)._asdict().items()}


def _device(arrays: dict) -> Carry:
    return Carry(*(jnp.asarray(arrays[key], dtype=jnp.uint64 if key in WIDE_FIELDS else None)
                   for key in Carry._fields))


def _running(carry: dict, clock: AdaptiveClock) -> np.ndarray:
    return carry["alive"] & (carry["elapsed_ticks"] < clock.target_ticks) & (carry["proposals"] < clock.nfe_cap)


def _run_spec(model: LoadedModel, kcm, clock: AdaptiveClock, *, chains, seed, force_cap,
              redraw_attempts, target_ns, nfe_cap_factor, frame_ps, save_every_ps, unroll) -> dict:
    spec = {
        "checkpoint": {"path": str(model.path), "step": model.step, "ema": model.ema},
        "system": OmegaConf.to_container(model.config.system),
        "sampler": {
            "chains": int(chains),
            "seed": int(seed),
            "tau_max_fs": clock.tau_max_ticks / C.TICKS_PER_FS,
            "tau_min_fs": clock.tau_min_ticks / C.TICKS_PER_FS,
            "redraw_attempts": int(redraw_attempts),
            "force_cap": None if force_cap is None else float(force_cap),
            "attention": kcm.net.attention_backend,
            "energy_filter": {"kinetic_floor_kbt": C.FILTER_KINETIC_FLOOR_KBT},
            "angular_momentum_map_min_det_ratio": C.ANGULAR_MOMENTUM_MAP_MIN_DET_RATIO,
            "target_ns": float(target_ns),
            "nfe_cap_factor": float(nfe_cap_factor),
            "frame_ps": float(frame_ps),
            "save_every_ps": float(save_every_ps),
            "unroll": int(unroll),
        },
        "runtime": {"jax": jax.__version__, "device": jax.devices()[0].device_kind,
                    "xla_flags": os.environ.get("XLA_FLAGS", "")},
    }
    return json.loads(json.dumps(spec))


def _summary(output: Path, state: dict, carry: dict, clock: AdaptiveClock, stop_reason: str,
             compile_seconds: float, wall_cumulative_seconds: float) -> dict:
    totals = chunks.counter_totals(output, carry["alive"].shape[0])
    tick_ns = C.TICK_PS / 1000
    alive, elapsed, proposals = carry["alive"], carry["elapsed_ticks"], carry["proposals"]
    reached = alive & (elapsed >= clock.target_ticks)
    capped = alive & ~reached & (proposals >= clock.nfe_cap)
    redraws = int(totals["redraws"].sum())
    if _running(carry, clock).any():
        status = stop_reason
    else:
        status = "COMPLETE" if reached.all() else "INCOMPLETE"
    return dict(
        status=status,
        chunks=state["chunks"],
        chains=dict(reached_target=int(reached.sum()), hit_nfe_cap=int(capped.sum()),
                    failed_nonfinite=int((~alive).sum()), running=int(_running(carry, clock).sum())),
        simulated_ns=float(elapsed.sum() * tick_ns),
        simulated_ns_per_chain=(elapsed * tick_ns).tolist(),
        # Every proposal is a committed step, a redraw of a committed step, or an attempt of the step in progress.
        committed_steps=int(proposals.sum()) - redraws - int(carry["redraw_n"].sum()),
        redraws=redraws,
        below_tau_max=int(totals["below_tau_max"].sum()),
        wall_cumulative_seconds=wall_cumulative_seconds,
        compile_seconds=compile_seconds,
    )


def run_chains(model: LoadedModel, output, *, chains: int, tau_max_fs: float, tau_min_fs: float,
               force_cap: float | None, redraw_attempts: int, target_ns: float, frame_ps: float, seed: int,
               save_every_ps: float, nfe_cap_factor: float, wall_seconds: float | None,
               max_chunks: int | None, unroll: int, resume: bool = False) -> Path:
    """Run adaptive chains from the start configuration and write the restartable simulation directory (``chunks``
    layout).

    Chain i starts from ``model.start_position`` (centered in float32) with key
    ``fold_in(PRNGKey(seed), i)``. Runs with JAX x64 enabled (uint64 clock); the physics runs in float32. Settings are
    range-checked by ``config.validate(cfg, "simulate")``; this function checks only what needs the model.

    Args:
        model: a ``LoadedModel``; its ``kcm`` is used with ``force_cap``.
        output: output directory (created; must be empty unless ``resume``).
        chains: number of vmapped chains.
        tau_max_fs: lag of every step's first attempt, on the 1/16 fs clock.
        tau_min_fs: floor of the halving fallback grid.
        force_cap: per-atom cap on the dimensionless mean impulse; None = uncapped.
        redraw_attempts: attempts per lag before falling back to the next shorter lag.
        target_ns: simulated time per chain (multiple of 1.5625 fs).
        frame_ps: frame spacing; the stride is max(1, round(frame_ps / tau_max)) iterations.
        seed: run seed; chain i's key is fold_in(PRNGKey(seed), i).
        save_every_ps: simulated time per chunk file, a whole multiple of ``frame_ps``.
        nfe_cap_factor: a chain also stops at ceil(factor * target / tau_max) proposal attempts.
        wall_seconds: execution budget of this invocation (compilation excluded); None = unlimited.
        max_chunks: chunks committed by this invocation; None = unlimited.
        unroll: scan unrolling of the device loop.
        resume: continue a stopped run in ``output`` (the run spec must match).

    Returns:
        The output directory.

    Raises:
        ValueError: lags outside the training band, non-float32 parameters, a non-finite initial
            state, or a resume that does not match the committed files.
    """
    with jax.enable_x64(True):
        kcm = model.kcm.with_force_cap(force_cap)
        params = model.params
        clock = adaptive_clock(tau_max_fs=tau_max_fs, tau_min_fs=tau_min_fs, target_ns=target_ns,
                               nfe_cap_factor=nfe_cap_factor, frame_ps=frame_ps, save_every_ps=save_every_ps)
        for ticks in (clock.tau_max_ticks, clock.tau_min_ticks):
            if not kcm.train_tau_min_ps <= ticks * C.TICK_PS <= kcm.train_tau_max_ps:
                raise ValueError(f"lag {ticks / C.TICKS_PER_FS} fs is outside the training band "
                                 f"[{kcm.train_tau_min_ps}, {kcm.train_tau_max_ps}] ps")
        if str(jax.config.jax_default_prng_impl) != "threefry2x32" or jax.process_count() != 1:
            raise ValueError("adaptive chains need one JAX process and the Threefry PRNG")
        if any(leaf.dtype != jnp.float32 for leaf in jax.tree.leaves(params)):
            raise ValueError("adaptive chains need float32 parameters")
        source = np.asarray(model.start_position)
        seed_key = jax.random.PRNGKey(seed)
        initial = jax.tree.map(lambda *x: jnp.stack(x),
                               *(initial_carry(kcm, source, chain_key=jax.random.fold_in(seed_key, i))
                                 for i in range(chains)))
        host = _host(initial)
        if not all(np.isfinite(value).all() for value in host.values()):
            raise ValueError("non-finite initial chain state")
        spec = _run_spec(model, kcm, clock, chains=chains, seed=seed, force_cap=force_cap,
                         redraw_attempts=redraw_attempts, target_ns=target_ns, nfe_cap_factor=nfe_cap_factor,
                         frame_ps=frame_ps, save_every_ps=save_every_ps, unroll=unroll)

        output = Path(output).expanduser().resolve()
        with directory_lock(output):
            chunks.check_output(output, resume)
            if resume:
                host, state, wall_cumulative = chunks.load(output, spec, host)
                carry = _device(host)
            else:
                state, wall_cumulative = chunks.initialize(output, spec, host), 0.0
                carry = initial
            active = make_attempt(kcm, params, tau_max_ticks=clock.tau_max_ticks, tau_min_ticks=clock.tau_min_ticks,
                                  redraw_attempts=redraw_attempts)
            compiled, compile_seconds, committed = None, 0.0, 0
            started = time.monotonic()
            stop_reason = "COMPLETE"
            while _running(host, clock).any():
                if max_chunks is not None and committed >= max_chunks:
                    stop_reason = "CHUNK_BUDGET"
                    break
                if wall_seconds is not None and time.monotonic() - started >= wall_seconds:
                    stop_reason = "WALL_BUDGET"
                    break
                if compiled is None:
                    compile_started = time.perf_counter()
                    compiled = make_chunk(active, clock.chunk_blocks, stride=clock.stride,
                                          target_ticks=clock.target_ticks, nfe_cap=clock.nfe_cap,
                                          tau_max_ticks=clock.tau_max_ticks, unroll=unroll).lower(carry).compile()
                    compile_seconds = time.perf_counter() - compile_started
                    started += compile_seconds
                kernel_started = time.perf_counter()
                carry, frames, counters = compiled(carry)
                jax.block_until_ready((carry, frames, counters))
                wall = time.perf_counter() - kernel_started
                host = _host(carry)
                frames, counters = jax.device_get((frames, counters))
                index = state["chunks"]
                wall_cumulative += wall
                chunks.commit(output, state, frames._asdict(), counters._asdict(), wall_cumulative, host)
                committed += 1
                simulated_ns = host["elapsed_ticks"] * (C.TICK_PS / 1000)
                print(f"chunk {index}: running {int(_running(host, clock).sum())}/{chains}, simulated ns "
                      f"{simulated_ns.min():.4f}-{simulated_ns.max():.4f}, redraws {int(counters.redraws.sum())}, "
                      f"wall {wall:.1f} s", flush=True)

            summary = _summary(output, state, host, clock, stop_reason, compile_seconds, wall_cumulative)
            chunks.write_json(output / "summary.json", summary)
        print(json.dumps({k: summary[k] for k in ("status", "chunks", "simulated_ns", "committed_steps", "redraws",
                                                  "below_tau_max")}), flush=True)
        return output


__all__ = ["run_chains"]
