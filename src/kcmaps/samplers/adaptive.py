"""Adaptive chains: one KC-map proposal attempt per chain and device iteration, on an exact 1/16 fs clock.

Each step makes one KC-map proposal at ``tau_max``. The proposal's internal momenta are rescaled by the energy
filter so that U + K_int follows the Langevin energy law. A proposal that is non-finite, leaves the energy guard or
makes the angular-momentum map T nearly singular is redrawn with fresh noise, ``redraw_attempts`` times per lag,
then at each shorter lag of a halving grid down to ``tau_min``; the last floor attempt is committed as is. The next
step starts at ``tau_max`` again.

A chain runs while it is alive (no committed non-finite state), its clock is below the target and its proposal count
is below the NFE cap; lags are never clipped, so a chain ends at most one ``tau_max`` past the target.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from ..core import constants as C
from .energy_filter import apply_filter, energy_terms, max_atom_kinetic, potential_energy, within_guard

WIDE_FIELDS = ("elapsed_ticks", "proposals")


class Carry(NamedTuple):
    """State of one chain (leading chain axis when batched).

    ``elapsed_ticks`` is on the 1/16 fs clock; ``proposals`` counts every attempt including redraws (the NFE).
    ``redraw_n``, ``redraw_level`` and ``redraw_tries`` locate the next attempt within the current step (all zero
    between steps).
    """

    phase: jax.Array
    key: jax.Array
    previous_u: jax.Array
    elapsed_ticks: jax.Array
    proposals: jax.Array
    redraw_n: jax.Array
    redraw_level: jax.Array
    redraw_tries: jax.Array
    alive: jax.Array


class Record(NamedTuple):
    """One proposal attempt: ``committed`` ends its step, ``valid`` is a committed finite attempt, ``redraws`` its
    index within the step."""

    lag_ticks: jax.Array
    committed: jax.Array
    valid: jax.Array
    redraws: jax.Array


class Frames(NamedTuple):
    """Every chain after each block; leading axes (block, chain)."""

    positions: jax.Array
    end_ticks: jax.Array
    alive: jax.Array


class Counters(NamedTuple):
    """Per-chain sums over one chunk's committed steps: redraw attempts, and steps committed below ``tau_max``."""

    redraws: jax.Array
    below_tau_max: jax.Array


def lag_grid(tau_max_ticks: int, tau_min_ticks: int) -> tuple[int, ...]:
    """Lags (ticks) from ``tau_max_ticks`` down to ``tau_min_ticks``, each the previous one halved (rounded down)
    and the last clamped to ``tau_min_ticks``."""
    grid = [tau_max_ticks]
    while grid[-1] > tau_min_ticks:
        grid.append(max(tau_min_ticks, grid[-1] // 2))
    return tuple(grid)


def initial_carry(kcm, positions, *, chain_key) -> Carry:
    """Time-zero carry of one chain: centered ``positions`` (N, 3) nm, Maxwell momenta and chain key from
    ``chain_key`` (the runner's fold_in(PRNGKey(seed), chain index)).

    Needs JAX x64 (uint64 clock); physics runs in float32.
    """
    with jax.enable_x64(False):
        momentum_key = jax.random.fold_in(chain_key, C.CHAIN_MOMENTUM_STREAM)
        phase = kcm.initial_state(positions, momentum_key)
        potential = potential_energy(kcm, phase)
    return Carry(
        phase=phase,
        key=jax.random.fold_in(chain_key, C.CHAIN_STREAM),
        previous_u=potential,
        elapsed_ticks=jnp.uint64(0),
        proposals=jnp.uint64(0),
        redraw_n=jnp.int32(0),
        redraw_level=jnp.int32(0),
        redraw_tries=jnp.int32(0),
        alive=jnp.bool_(True),
    )


def make_attempt(kcm, params, *, tau_max_ticks: int, tau_min_ticks: int, redraw_attempts: int):
    """One-chain transition ``carry -> (carry, Record)`` making one proposal attempt.

    A failing attempt with attempts left is held: phase, clock, key and energy are unchanged, only the redraw
    counters and ``proposals`` advance. Attempt n >= 1 of a step draws its noise from
    fold_in(fold_in(epsilon_key, REDRAW_STREAM), n); attempt 0 uses epsilon_key itself.
    """
    grid = lag_grid(tau_max_ticks, tau_min_ticks)
    max_level = len(grid) - 1
    n_atoms = kcm.masses.shape[0]

    def attempt(c, ticks, epsilon_key):
        tau_ps = ticks.astype(jnp.float32) * C.TICK_PS
        source = c.phase
        epsilon = jax.random.normal(epsilon_key, source.shape, jnp.float32)
        displacement, _ = kcm.propose(params, source, tau_ps, epsilon)
        map_valid = kcm.angular_momentum_map_valid(kcm.transform.to_dimensionless(source),
                                                   kcm.transform.to_dimensionless(displacement))
        proposal = kcm.recenter_state(source + displacement)
        potential = potential_energy(kcm, proposal)
        terms = energy_terms(kcm, source, c.previous_u, proposal, potential, tau_ps,
                             jax.random.fold_in(epsilon_key, C.ENERGY_STREAM))
        passes = within_guard(terms, n_atoms)
        proposal = apply_filter(terms, proposal)
        # Finite momenta can still overflow the kinetic energy; dU overflows only for |U| near the float32 limit.
        finite = (jnp.all(jnp.isfinite(source)) & jnp.all(jnp.isfinite(proposal)) & jnp.isfinite(potential)
                  & jnp.isfinite(max_atom_kinetic(kcm, proposal)) & jnp.isfinite(potential - c.previous_u))
        return proposal, potential, finite, finite & passes & map_valid

    def active(c):
        with jax.enable_x64(False):
            next_key, epsilon_key = jax.random.split(c.key)
            first = c.redraw_n == 0
            level = c.redraw_level
            tries = jnp.where(first, jnp.int32(1), c.redraw_tries)
            ticks = jnp.asarray(grid, jnp.uint32)[level]
            key = jnp.where(first, epsilon_key,
                            jax.random.fold_in(jax.random.fold_in(epsilon_key, C.REDRAW_STREAM), c.redraw_n))
            proposal, potential, finite, ok = attempt(c, ticks, key)
            more = ~ok & ((level + 1 <= max_level) | (tries < redraw_attempts))
            down = tries >= redraw_attempts
            held_level = jnp.where(down, level + 1, level)
            held_tries = jnp.where(down, 1, tries + 1)
        committed = ~more
        result = Carry(
            phase=proposal,
            key=next_key,
            previous_u=potential,
            elapsed_ticks=jnp.where(finite, c.elapsed_ticks + ticks, c.elapsed_ticks),
            proposals=c.proposals + 1,
            redraw_n=jnp.int32(0),
            redraw_level=jnp.int32(0),
            redraw_tries=jnp.int32(0),
            alive=finite,
        )
        held = c._replace(proposals=c.proposals + 1, redraw_n=c.redraw_n + 1, redraw_level=held_level,
                          redraw_tries=held_tries)
        record = Record(lag_ticks=ticks, committed=committed, valid=committed & finite, redraws=c.redraw_n)
        return jax.tree.map(lambda new, old: jnp.where(committed, new, old), result, held), record

    return active


def make_chunk(active, n_blocks: int, *, stride: int, target_ticks: int, nfe_cap: int, tau_max_ticks: int,
               unroll: int = 1):
    """Jitted ``carry -> (carry, Frames, Counters)`` over ``n_blocks`` blocks of ``stride`` iterations; each
    iteration applies ``active`` to every running chain. Blocks in which no chain runs are skipped."""
    batched = jax.vmap(active)
    target, cap = jnp.uint64(target_ticks), jnp.uint64(nfe_cap)

    def is_running(c):
        return c.alive & (c.elapsed_ticks < target) & (c.proposals < cap)

    def chunk(carry):
        def iteration(state, _):
            c, counters = state
            run = is_running(c)
            new, record = batched(c)
            c = jax.tree.map(lambda a, b: jnp.where(run.reshape(run.shape + (1,) * (a.ndim - 1)), a, b), new, c)
            done = run & record.committed
            counters = Counters(
                redraws=counters.redraws + jnp.where(done, record.redraws, 0),
                below_tau_max=counters.below_tau_max + (done & record.valid & (record.lag_ticks < tau_max_ticks)),
            )
            return (c, counters), None

        def block(state, _):
            state = jax.lax.cond(
                jnp.any(is_running(state[0])),
                lambda s: jax.lax.scan(iteration, s, None, length=stride, unroll=unroll)[0],
                lambda s: s,
                state,
            )
            c = state[0]
            return state, Frames(c.phase[..., :3], c.elapsed_ticks, c.alive)

        n = carry.alive.shape[0]
        counters = Counters(jnp.zeros(n, jnp.int64), jnp.zeros(n, jnp.int64))
        (carry, counters), frames = jax.lax.scan(block, (carry, counters), None, length=n_blocks)
        return carry, frames, counters

    return jax.jit(chunk)


__all__ = ["Carry", "Counters", "Frames", "Record", "initial_carry", "lag_grid", "make_attempt", "make_chunk"]
