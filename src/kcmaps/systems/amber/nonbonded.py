"""LJ-12-6 + Coulomb with scaled 1-4 exceptions in JAX (kJ/mol, one configuration), without cutoff or with OpenMM's
CutoffNonPeriodic reaction field. Below short-range floors both terms are softened so the score stays finite.
"""

from __future__ import annotations

import jax.numpy as jnp

from .geom import pairwise, safe_norm
from .topology import ONE_4PI_EPS0

# LJ floor, σ-relative (the LJ singularity scales with σ). 0.5 puts the splice
# at >=375 kT for every Amber pair (zero Boltzmann weight), far outside
# thermally accessible distances.
SOFTCORE_LAMBDA = 0.5

# Coulomb floor, absolute (nm): 1/r has no length scale, and OpenMM writes
# σ = 1.0 nm as a placeholder for ε = 0 atoms, so a σ-keyed floor would land
# inside the sampled region. 0.01 nm is far below physical non-bonded distances.
COULOMB_R_MIN = 0.01


def reaction_field_constants(cutoff: float, dielectric: float) -> tuple[float, float]:
    """OpenMM's reaction-field coefficients ``(krf, crf)`` for a cutoff.

    ``E_coul(r) = k·q_i q_j · (1/r + krf·r² − crf)`` inside ``rc``. Under implicit
    solvent OpenMM sets ``dielectric = 1``, which zeroes ``krf`` and leaves
    ``crf = 1/rc``: a pure constant shift with no force. That degeneracy is why
    the cutoff is invisible in the dynamics of a small peptide.
    """
    krf = (1.0 / cutoff ** 3) * ((dielectric - 1.0) / (2.0 * dielectric + 1.0))
    crf = (1.0 / cutoff) * (3.0 * dielectric / (2.0 * dielectric + 1.0))
    return krf, crf


def _lj_coulomb_softcore(r: jnp.ndarray,
                         sigma_ij: jnp.ndarray,
                         eps_ij: jnp.ndarray,
                         qq_ij: jnp.ndarray,
                         krf: float = 0.0,
                         crf: float = 0.0) -> jnp.ndarray:
    """LJ-12-6 + Coulomb, each bounded below its own floor.

    LJ below ``r_lj = λ·σ_ij``, Coulomb below the absolute ``COULOMB_R_MIN``
    (see the two constants). Above its floor each term is
    the exact potential. Below, they are bounded DIFFERENTLY on purpose: LJ is
    continued linearly (constant outward force, which is what a repulsive wall
    should do), Coulomb is held constant (an attractive pair continued linearly
    would deepen without limit and pull inward). Both keep the forward value and
    the backward gradient finite. Element-wise on any shape (N×N pair grid or the
    1-D exception list).
    """
    r_lj = SOFTCORE_LAMBDA * sigma_ij
    below_lj = r < r_lj
    r_eff = jnp.maximum(r, r_lj)               # ≥ r_lj ⇒ no fp32 overflow

    inv = sigma_ij / r_eff
    s6 = inv ** 6
    s12 = s6 * s6
    e_lj_eff = 4.0 * eps_ij * (s12 - s6)
    # dE_lj/dr at r_eff (= r_lj on the `below` branch — gives the splice slope)
    dlj_dr = 4.0 * eps_ij * (-12.0 * s12 + 6.0 * s6) / r_eff
    e_lj = jnp.where(below_lj, e_lj_eff + dlj_dr * (r - r_lj), e_lj_eff)

    # Clamped flat, not continued linearly (see the docstring). ONLY the
    # singular 1/r is clamped: krf*r^2 and crf are regular at r=0.
    # krf/crf terms are added only when nonzero, so the no-cutoff graph carries no extra ops.
    e_c = ONE_4PI_EPS0 * qq_ij / jnp.maximum(r, COULOMB_R_MIN)
    if krf:
        e_c = e_c + ONE_4PI_EPS0 * qq_ij * krf * r * r
    if crf:
        e_c = e_c - ONE_4PI_EPS0 * qq_ij * crf
    return e_lj + e_c


def nonbonded_pair_energy(x: jnp.ndarray,
                          pair_mask: jnp.ndarray,
                          sigma_ij: jnp.ndarray,
                          eps_ij: jnp.ndarray,
                          qq_ij: jnp.ndarray,
                          cutoff: float | None = None,
                          rf_dielectric: float = 1.0) -> jnp.ndarray:
    """Σ_{i<j, unmasked} [softcore LJ-12-6 + Coulomb].

    ``pair_mask`` is an upper-triangular boolean (diagonal and 1-2/1-3/1-4
    entries are False); ``sigma_ij``, ``eps_ij``, ``qq_ij`` are the full
    (N, N) precombined matrices.

    ``cutoff`` is a static Python scalar, so ``None`` is resolved at trace time
    and emits exactly the vacuum graph. When set, pairs at ``r >= cutoff`` drop
    out and the in-range Coulomb picks up the reaction field.
    """
    # Safe pairwise distances (diagonal ~1.0, off-diagonal floored at
    # ~1e-6 nm) so the masked backward pass cannot hit ``0 · inf = NaN``
    # on a coincident excluded pair. See :mod:`geom`.
    _, _, dist = pairwise(x)                           # (N, N)
    if cutoff is None:
        krf = crf = 0.0
        mask = pair_mask
    else:
        krf, crf = reaction_field_constants(cutoff, rf_dielectric)
        mask = pair_mask & (dist < cutoff)
    safe_dist = jnp.where(mask, dist, 1.0)
    e_pair = _lj_coulomb_softcore(safe_dist, sigma_ij, eps_ij, qq_ij, krf, crf)
    return jnp.sum(jnp.where(mask, e_pair, 0.0))


def nonbonded_exception_energy(x: jnp.ndarray,
                               exc_indices: jnp.ndarray,
                               exc_chargeprod: jnp.ndarray,
                               exc_sigma: jnp.ndarray,
                               exc_epsilon: jnp.ndarray) -> jnp.ndarray:
    """Σ_{1-4 exc} [softcore LJ-12-6 + Coulomb] on the pre-scaled 1-4
    exception pairs (SCEE / SCNB already applied by OpenMM in
    ``getExceptionParameters``).

    Returns ``0.0`` cleanly when ``exc_indices`` is empty (no scaled
    1-4 pairs in the system).
    """
    p0 = x[exc_indices[:, 0]]
    p1 = x[exc_indices[:, 1]]
    r = safe_norm(p1 - p0, axis=-1)
    return jnp.sum(_lj_coulomb_softcore(r, exc_sigma, exc_epsilon, exc_chargeprod))


def nonbonded_energy(x: jnp.ndarray,
                     pair_mask: jnp.ndarray,
                     sigma_ij: jnp.ndarray,
                     eps_ij: jnp.ndarray,
                     qq_ij: jnp.ndarray,
                     exc_indices: jnp.ndarray,
                     exc_chargeprod: jnp.ndarray,
                     exc_sigma: jnp.ndarray,
                     exc_epsilon: jnp.ndarray,
                     cutoff: float | None = None,
                     rf_dielectric: float = 1.0) -> jnp.ndarray:
    """Convenience: pair sum + scaled-exception sum.

    The cutoff reaches the pair sum only: OpenMM exempts exceptions from both
    the reaction field and the cutoff.
    """
    return (nonbonded_pair_energy(x, pair_mask, sigma_ij, eps_ij, qq_ij,
                                  cutoff, rf_dielectric)
            + nonbonded_exception_energy(x, exc_indices, exc_chargeprod,
                                         exc_sigma, exc_epsilon))
