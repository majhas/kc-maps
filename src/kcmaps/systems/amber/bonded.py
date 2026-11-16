"""Bonded Amber terms in JAX: harmonic bonds and angles and periodic torsions, in kJ/mol for one configuration."""

from __future__ import annotations

import jax.numpy as jnp

from .geom import safe_norm


def bond_energy(x: jnp.ndarray, indices: jnp.ndarray,
                k: jnp.ndarray, r0: jnp.ndarray) -> jnp.ndarray:
    """Σ 0.5·k·(|p_a - p_b| - r0)²  over harmonic bonds.

    Args:
        x:        (N, 3) atom positions, nm.
        indices:  (n_bonds, 2) int — bond endpoint atom indices.
        k:        (n_bonds,) force constants, kJ/(mol·nm²).
        r0:       (n_bonds,) equilibrium lengths, nm.
    """
    p0 = x[indices[:, 0]]
    p1 = x[indices[:, 1]]
    delta = p1 - p0
    r = safe_norm(delta, axis=-1)
    return jnp.sum(0.5 * k * (r - r0) ** 2)


def angle_energy(x: jnp.ndarray, indices: jnp.ndarray,
                 k: jnp.ndarray, theta0: jnp.ndarray) -> jnp.ndarray:
    """Σ 0.5·k·(θ - θ0)²  over harmonic angles centered at ``indices[:, 1]``.

    The cosine is clamped to ``[-1+1e-7, 1-1e-7]`` so ``acos`` stays
    differentiable at degenerate (linear) angles. Physical angles never
    hit the boundary; the clamp only matters for numerical safety on
    random or out-of-distribution inputs such as model proposals.
    """
    p0 = x[indices[:, 0]]
    p1 = x[indices[:, 1]]
    p2 = x[indices[:, 2]]
    r_ab = p0 - p1
    r_cb = p2 - p1
    n_ab = safe_norm(r_ab, axis=-1)
    n_cb = safe_norm(r_cb, axis=-1)
    cos_t = jnp.sum(r_ab * r_cb, axis=-1) / jnp.maximum(n_ab * n_cb, 1e-15)
    cos_t = jnp.clip(cos_t, -1.0 + 1e-7, 1.0 - 1e-7)
    theta = jnp.arccos(cos_t)
    return jnp.sum(0.5 * k * (theta - theta0) ** 2)


def torsion_energy(x: jnp.ndarray, indices: jnp.ndarray, k: jnp.ndarray,
                   n: jnp.ndarray, phase: jnp.ndarray) -> jnp.ndarray:
    """Σ k·(1 + cos(n·φ - phase))  over periodic torsions, ``atan2`` form.

    The dihedral is computed as ``φ = atan2(dot(m1, n2), dot(n1, n2))``
    where ``n1 = b1×b2``, ``n2 = b2×b3``, ``m1 = (b2/|b2|) × n1``. The
    ``(b2/|b2|) × n1`` orientation (not ``n1 × b2/|b2|``) is what makes φ
    match OpenMM's signed dihedral: the two differ by φ → −φ, which is
    invisible whenever ``phase ∈ {0, π}`` (all Amber phases) but flips
    energy and forces for any other phase. Collapsed bonds and collinear
    triplets remain singular: norm floors do not define the derivative of
    ``atan2(0, 0)`` when either plane normal vanishes.
    """
    p0 = x[indices[:, 0]]
    p1 = x[indices[:, 1]]
    p2 = x[indices[:, 2]]
    p3 = x[indices[:, 3]]
    b1 = p1 - p0
    b2 = p2 - p1
    b3 = p3 - p2

    n1 = jnp.cross(b1, b2)
    n2 = jnp.cross(b2, b3)
    b2_norm = safe_norm(b2, axis=-1, keepdims=True)
    m1 = jnp.cross(b2 / b2_norm, n1)

    x_ = jnp.sum(n1 * n2, axis=-1)
    y_ = jnp.sum(m1 * n2, axis=-1)
    phi = jnp.arctan2(y_, x_)

    return jnp.sum(k * (1.0 + jnp.cos(n * phi - phase)))
