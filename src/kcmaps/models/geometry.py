"""Phase-space geometry: units, center-of-mass constraints, the N-CA-C source frame and the angular-momentum map."""

from __future__ import annotations

import dataclasses
from typing import NamedTuple

import jax
import jax.numpy as jnp
from flax import struct


class LinearConstraint(NamedTuple):
    """Functional/direction pair for ``P(v)=v-b*(a.v)``, with ``a.b=1``."""

    functional: jax.Array
    direction: jax.Array

    def normalized_functional(self) -> jax.Array:
        """Return ``a/(a.T b)`` so every consumer uses the same projector."""
        normalizer = jnp.sum(self.functional * self.direction, axis=-2, keepdims=True)
        return self.functional / normalizer

    def offset(self, value: jax.Array) -> jax.Array:
        return -self.direction * jnp.sum(self.normalized_functional() * value, axis=-2, keepdims=True)

    def project(self, value: jax.Array) -> jax.Array:
        return value + self.offset(value)


class PhaseConstraints(NamedTuple):
    """Center-of-mass constraints of the dimensionless position, momentum and noise."""

    position: LinearConstraint
    momentum: LinearConstraint
    noise: LinearConstraint


@dataclasses.dataclass(frozen=True)
class Scales:
    """Fixed units of dimensionless coordinates (``checkpoint.make_scales``; only x_std_nm is stored).

    Attributes:
        T_std_ps: time unit (ps).
        x_std_nm: position unit (nm), the RMS of the centered training positions.
        p_std: per-atom momentum units sqrt(m_i kBT) (amu nm/ps).
    """

    T_std_ps: float
    x_std_nm: float
    p_std: tuple[float, ...]


@struct.dataclass
class CoordinateTransform:
    """Fixed diagonal map between physical and dimensionless states, times and forces."""

    x_std_nm: jax.Array
    p_std: jax.Array
    T_std_ps: jax.Array

    def to_dimensionless(self, state: jax.Array) -> jax.Array:
        """(..., 6) physical state or displacement to dimensionless."""
        state = jnp.asarray(state)
        return jnp.concatenate([state[..., :3] / self.x_std_nm, state[..., 3:] / self.p_std], axis=-1)

    def to_phys(self, state: jax.Array) -> jax.Array:
        """(..., 6) dimensionless state or displacement to physical."""
        state = jnp.asarray(state)
        return jnp.concatenate([self.x_std_nm * state[..., :3], self.p_std * state[..., 3:]], axis=-1)

    def to_dimensionless_time(self, time: jax.Array) -> jax.Array:
        return jnp.asarray(time) / self.T_std_ps

    def to_dimensionless_force(self, force: jax.Array) -> jax.Array:
        return self.T_std_ps * jnp.asarray(force) / self.p_std

    def to_phys_force(self, force: jax.Array) -> jax.Array:
        return self.p_std * jnp.asarray(force) / self.T_std_ps

    def restricted_logdet_to_phys(self) -> jax.Array:
        """Log-volume of the map to physical units on the center-of-mass-free subspaces: for diagonal D and normal n,
        prod(D) |D^-1 n| / |n|, with dimensionless normals m x_std and p_std."""
        p_std = jnp.asarray(self.p_std).reshape(-1)
        n_atoms = p_std.shape[0]
        position = (n_atoms - 1) * jnp.log(self.x_std_nm)
        momentum = (jnp.sum(jnp.log(p_std)) + 0.5*jnp.log(jnp.asarray(n_atoms, p_std.dtype))
                    - 0.5*jnp.log(jnp.sum(p_std*p_std)))
        return 3.0 * (position + momentum)


def make_constraints(masses, x_std_nm, p_std) -> PhaseConstraints:
    """Mass-weighted center of positions, total momentum, and noise conditioned orthogonally on the normal p_std."""
    mass = jnp.asarray(masses, dtype=p_std.dtype).reshape(-1, 1)
    one = jnp.ones_like(mass)
    return PhaseConstraints(position=LinearConstraint(mass*x_std_nm, one/x_std_nm),
                            momentum=LinearConstraint(p_std, mass/p_std), noise=LinearConstraint(p_std, p_std))


def flip_momentum(state):
    """(..., 6) state with its momenta reversed."""
    return jnp.concatenate([state[..., :3], -state[..., 3:]], axis=-1)


# -- N-CA-C source frame --------------------------------------------------------

def source_frame_valid(positions, anchor_indices, min_bond_length_nm, min_sin_angle):
    """Dataset-preflight check of physical (optionally batched) positions: whether their N-CA-C anchors span a
    frame."""
    anchors = jnp.take(positions, jnp.asarray(anchor_indices), axis=-2)
    finite = jnp.all(jnp.isfinite(anchors), axis=(-2, -1))
    anchors = jnp.where(jnp.isfinite(anchors), anchors, 0)
    u = anchors[..., 0, :] - anchors[..., 1, :]
    v = anchors[..., 2, :] - anchors[..., 1, :]
    u2, v2 = jnp.sum(u*u, -1), jnp.sum(v*v, -1)
    un = jnp.sqrt(jnp.where(u2 > 0, u2, 1))
    vn = jnp.sqrt(jnp.where(v2 > 0, v2, 1))
    e1 = u / un[..., None]
    w = v - e1*jnp.sum(e1*v, -1, keepdims=True)
    w2 = jnp.sum(w*w, -1)
    wn = jnp.sqrt(jnp.where(w2 > 0, w2, 1))
    return (finite & jnp.isfinite(u2) & jnp.isfinite(v2) & jnp.isfinite(w2) & (u2 > 0) & (v2 > 0)
            & (un >= min_bond_length_nm) & (vn >= min_bond_length_nm) & (w2 > 0)
            & (wn/vn >= min_sin_angle))


def frame_from_source(source, x_std_nm, anchor_indices):
    """Frame axes of a dimensionless source, without validation (a degenerate source gives NaNs)."""
    indices = jnp.asarray(anchor_indices)
    scale = jnp.broadcast_to(jnp.asarray(x_std_nm), source.shape[-2:-1] + (1,))
    anchors = jnp.take(source[..., :3], indices, axis=-2)*jnp.take(scale, indices, axis=-2)
    u = anchors[..., 0, :] - anchors[..., 1, :]
    v = anchors[..., 2, :] - anchors[..., 1, :]
    e1 = u / jnp.sqrt(jnp.sum(u*u, axis=-1, keepdims=True))
    w = v - e1*jnp.sum(e1*v, axis=-1, keepdims=True)
    e2 = w / jnp.sqrt(jnp.sum(w*w, axis=-1, keepdims=True))
    return jnp.stack((e1, e2, jnp.cross(e1, e2)), axis=-1)


def vector_to_body(value, axes):
    if value.shape[-1] != 3:
        raise ValueError("vector rotation requires (..., atoms, 3)")
    return jnp.matmul(value, axes, precision=jax.lax.Precision.HIGHEST)


def phase_to_body(value, axes):
    if value.shape[-1] != 6:
        raise ValueError("phase rotation requires (..., atoms, 6)")
    return jnp.concatenate((vector_to_body(value[..., :3], axes),
                            vector_to_body(value[..., 3:], axes)), axis=-1)


def phase_to_lab(value, axes):
    return phase_to_body(value, jnp.swapaxes(axes, -1, -2))


# -- angular-momentum map ----------------------------------------------------------

# Only elementwise arithmetic and fixed-axis sums below (closed-form 3x3 algebra, no dot_general or LAPACK), so
# the result does not depend on the batch shape (sharding invariance) and no dot escapes the HIGHEST policy.

def _com_free(x, m):
    return x - jnp.sum(m[:, None] * x, axis=-2, keepdims=True) / jnp.sum(m)


def _angular(r, p):
    return jnp.sum(jnp.cross(r, p), axis=-2)


def _mixed_inertia(a, b, m):
    """I_mix(a, b) = sum m (a.b 1 - b a^T); I_r = I_mix(r, r)."""
    dot = jnp.sum(m[:, None] * a * b, axis=(-2, -1))
    outer = jnp.sum(m[:, None, None] * b[..., :, :, None] * a[..., :, None, :], axis=-3)
    return dot[..., None, None] * jnp.eye(3, dtype=a.dtype) - outer


def _rotation_field(w, r, m):
    return m[:, None] * jnp.cross(w[..., None, :], r)


def _det(a):
    return (a[..., 0, 0] * (a[..., 1, 1] * a[..., 2, 2] - a[..., 1, 2] * a[..., 2, 1])
            - a[..., 0, 1] * (a[..., 1, 0] * a[..., 2, 2] - a[..., 1, 2] * a[..., 2, 0])
            + a[..., 0, 2] * (a[..., 1, 0] * a[..., 2, 1] - a[..., 1, 1] * a[..., 2, 0]))


def _inv(a):
    c = lambda i, j: (a[..., (i + 1) % 3, (j + 1) % 3] * a[..., (i + 2) % 3, (j + 2) % 3]
                      - a[..., (i + 1) % 3, (j + 2) % 3] * a[..., (i + 2) % 3, (j + 1) % 3])
    adjugate = jnp.stack([jnp.stack([c(j, i) for j in range(3)], -1) for i in range(3)], -2)
    return adjugate / _det(a)[..., None, None]


def _solve(matrix, vector):
    return jnp.sum(_inv(matrix) * vector[..., None, :], axis=-1)


def _geometry(x_s, dx, m):
    r_s, dr = _com_free(x_s, m), _com_free(dx, m)
    return r_s, dr, r_s + dr


def _map_det(geometry, m):
    """det T = det(1 - I_out^-1 I_mix(dr, r_out)) = det I_mix(r_s, r_out) / det I_out."""
    r_s, dr, r_out = geometry
    i_out_inv = _inv(_mixed_inertia(r_out, r_out, m))
    product = jnp.sum(i_out_inv[..., :, :, None] * _mixed_inertia(dr, r_out, m)[..., None, :, :], axis=-2)
    return _det(jnp.eye(3, dtype=r_s.dtype) - product)


def angular_momentum_map(x_s, dx, p, m, inverse=False):
    """(T p - p, log|det T|) for the endpoint momentum p of a displacement dx from x_s; (T^-1 p - p, log|det T|)
    with ``inverse``. Physical (..., N, 3) arrays, masses (N,).

    The flow pins Lambda = sum r_s x p' (about the source lever arm); T adds the rigid rotation about the output that
    restores the Langevin law of L about the current positions:
    T p' = p' - U_out I_out^-1 sum dr x p', dr = r_out - r_s, with U_r w = m (w x r), I_r = I_mix(r, r).
    T keeps the total momentum; T^-1 p'' = p'' + U_out I_mix(r_s, r_out)^-1 sum dr x p''. The log-det depends on
    positions only.
    """
    geometry = _geometry(x_s, dx, m)
    r_s, dr, r_out = geometry
    gap = _angular(dr, p)
    if inverse:
        w = _solve(_mixed_inertia(r_s, r_out, m), gap)
    else:
        w = -_solve(_mixed_inertia(r_out, r_out, m), gap)
    return _rotation_field(w, r_out, m), jnp.log(jnp.abs(_map_det(geometry, m)))


def without_rotation(vectors, r, m):
    """Momentum-like per-atom vectors less their rigid rotation about the lever arm r: v - U_r I_r^-1 sum r x v.

    The result carries no angular momentum about r and the same total as v (U_r w sums to zero for CoM-free r).
    """
    r = _com_free(r, m)
    return vectors - _rotation_field(_solve(_mixed_inertia(r, r, m), _angular(r, vectors)), r, m)


def angular_momentum_map_det(x_s, dx, m):
    """Signed det T: ~1 for thermal displacements, cos t (1 + cos t) / 2 for a pure turn t of an isotropic rotor,
    approaching 0 near a quarter turn."""
    return _map_det(_geometry(x_s, dx, m), m)
