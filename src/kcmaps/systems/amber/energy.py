"""Total Amber energy (kBT units) and score of one configuration, and the jitted ``AmberBackend``. The force field
comes from the system config (AL2/AL4: amber99sbildn + OBC2; AL6: amber14 + OBC1).
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp

from .bonded import angle_energy, bond_energy, torsion_energy
from .gbsa_obc import gbsa_obc_energy
from .nonbonded import nonbonded_energy
from .topology import AmberParams


def total_energy(x: jnp.ndarray, params: AmberParams) -> jnp.ndarray:
    """Sum of the bonded, nonbonded and GB terms of one configuration ``x: (N, 3)``, in kBT units."""
    e = bond_energy(x, params.bond_indices, params.bond_k, params.bond_r0)
    e = e + angle_energy(x, params.angle_indices, params.angle_k,
                         params.angle_theta0)
    e = e + torsion_energy(x, params.torsion_indices, params.torsion_k,
                           params.torsion_n, params.torsion_phase)
    e = e + nonbonded_energy(
        x, params.nb_pair_mask, params.nb_sigma_ij, params.nb_eps_ij,
        params.nb_qq_ij, params.exc_indices, params.exc_chargeprod,
        params.exc_sigma, params.exc_epsilon,
        params.nb_cutoff, params.nb_rf_dielectric)
    e = e + gbsa_obc_energy(
        x, params.gb_offset_radius, params.gb_scaled_radius, params.gb_charge,
        params.gb_qq, params.gb_prefactor, params.gb_sa_factor,
        params.gb_probe_radius, params.gb_dielectric_offset,
        params.gb_alpha, params.gb_beta, params.gb_gamma,
        params.nb_cutoff)
    return e / params.kBT


def score_single(x: jnp.ndarray, params: AmberParams) -> jnp.ndarray:
    """``∇ log π(x) = -∇(U/kBT)``, single configuration ``(N, 3)`` → ``(N, 3)``."""
    return -jax.grad(total_energy)(x, params)


def _to_jax_params(params: AmberParams, float_dtype=None) -> AmberParams:
    """Cast every numpy array field of ``params`` to ``jnp`` so it lives
    on-device after the first JIT trace. Scalar floats / ints stay as
    Python values (treated as static by JAX).

    ``float_dtype`` overrides the dtype of the float arrays; None follows the
    global x64 flag. Integer indices keep their dtype.
    """
    if float_dtype is None:
        float_dtype = jnp.zeros((), dtype=float).dtype  # honours jax_enable_x64
    repl = {}
    for f in dataclasses.fields(params):
        v = getattr(params, f.name)
        if hasattr(v, "dtype") and hasattr(v, "shape"):
            if v.dtype.kind == "f":
                repl[f.name] = jnp.asarray(v, dtype=float_dtype)
            else:
                repl[f.name] = jnp.asarray(v)
    return dataclasses.replace(params, **repl)


class AmberBackend:
    """JIT-compiled score / energy on a fixed :class:`AmberParams`.

    Held by :class:`PeptideSystem` as a non-pytree field — the wrapped
    JAX functions close over the params so the AmberBackend itself never
    needs to participate in pytree traversal.
    """

    def __init__(self, params: AmberParams, float_dtype=None) -> None:
        self.params = params
        self._dtype = float_dtype
        self._jparams = _to_jax_params(params, float_dtype)
        # vmap → JIT: caller passes (B, N, 3); we return (B, N, 3).
        self._score_fn = jax.jit(
            jax.vmap(lambda xi: score_single(xi, self._jparams)))
        # Compiled on first use (DB loss, energy filter, MH, evaluation).
        self._energy_fn = None

    def _narrow(self, x):
        """Cast the input to the backend's dtype, so the energy is evaluated at the precision of the parameters
        (float64 positions would otherwise promote a float32 backend to float64). Returns the cast input and the
        dtype to restore on the way out.
        """
        if self._dtype is None or x.dtype == self._dtype:
            return x, None
        return x.astype(self._dtype), x.dtype

    def score(self, x_batch: jnp.ndarray) -> jnp.ndarray:
        """``∇ log π(x)`` in units ``1/nm``. ``x_batch: (B, N, 3) → (B, N, 3)``."""
        x_batch, back = self._narrow(x_batch)
        out = self._score_fn(x_batch)
        return out if back is None else out.astype(back)

    def energy(self, x_batch: jnp.ndarray) -> jnp.ndarray:
        """``U(x)/kBT`` in kT units. ``x_batch: (B, N, 3) → (B,)``."""
        if self._energy_fn is None:
            self._energy_fn = jax.jit(
                jax.vmap(lambda xi: total_energy(xi, self._jparams)))
        x_batch, back = self._narrow(x_batch)
        out = self._energy_fn(x_batch)
        return out if back is None else out.astype(back)


def build_amber_backend(pdb_path: str, temperature_K: float, forcefield: tuple,
                        cutoff_nm: float | None = None,
                        float_dtype=None) -> AmberBackend:
    """One-line constructor: PDB → AmberParams → :class:`AmberBackend`."""
    from .topology import build_amber_params
    return AmberBackend(
        build_amber_params(pdb_path, temperature_K, forcefield, cutoff_nm),
        float_dtype)
