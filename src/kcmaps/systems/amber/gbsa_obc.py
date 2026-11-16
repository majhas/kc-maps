"""GBSA-OBC implicit-solvent energy in JAX (kJ/mol, one configuration): Born radii, self-energy, ACE surface term and
GB pair sum, with OBC1 or OBC2 constants.
"""

from __future__ import annotations

import jax.numpy as jnp

from .geom import pairwise


def _born_radii(dist: jnp.ndarray,
                offset_radius: jnp.ndarray,
                scaled_radius: jnp.ndarray,
                dielectric_offset: float,
                alpha: float, beta: float, gamma: float,
                cutoff: float | None = None
                ) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute the OBC effective Born radii ``B`` and ``radius = or + offset``
    from the pairwise distance matrix.

    Returns ``(B, radius)`` each of shape ``(N,)``.
    """
    n = dist.shape[0]
    or_col = offset_radius[:, None]                    # (N, 1)  — i
    sr_row = scaled_radius[None, :]                    # (1, N)  — j

    diag = ~jnp.eye(n, dtype=jnp.bool_)
    # Atom j contributes to atom i's Born integral only if it does not
    # fully engulf atom i (i.e. the overlap region is non-empty).
    include = diag & (or_col < (dist + sr_row))
    if cutoff is not None:
        include = include & (dist < cutoff)

    safe_r = jnp.where(include, dist, 1.0)             # avoid 1/0 elsewhere
    L = jnp.maximum(jnp.broadcast_to(or_col, dist.shape),
                    jnp.abs(safe_r - sr_row))
    U = safe_r + sr_row
    L_inv = 1.0 / jnp.maximum(L, 1e-15)
    U_inv = 1.0 / jnp.maximum(U, 1e-15)
    r_inv = 1.0 / jnp.maximum(safe_r, 1e-15)

    term = 0.5 * (
        L_inv - U_inv
        + 0.25 * (safe_r - sr_row ** 2 * r_inv) * (U_inv ** 2 - L_inv ** 2)
        + 0.5 * jnp.log(jnp.maximum(L / U, 1e-15)) * r_inv
    )
    term = jnp.where(include, term, 0.0)
    I = jnp.sum(term, axis=-1)                         # (N,)

    or_ = offset_radius
    psi = I * or_
    psi2 = psi * psi
    psi3 = psi2 * psi
    radius = or_ + dielectric_offset
    tanh_arg = alpha * psi - beta * psi2 + gamma * psi3
    inv_B = 1.0 / or_ - jnp.tanh(tanh_arg) / radius
    # Clamp from below so inv_B doesn't underflow (would give absurd B).
    B = 1.0 / jnp.maximum(inv_B, 1e-8)
    return B, radius


def gbsa_obc_energy(x: jnp.ndarray,
                    offset_radius: jnp.ndarray,   # (N,)
                    scaled_radius: jnp.ndarray,   # (N,)
                    charge: jnp.ndarray,          # (N,)
                    qq: jnp.ndarray,              # (N, N) = q_i q_j
                    prefactor: float,
                    sa_factor: float,
                    probe_radius: float,
                    dielectric_offset: float,
                    alpha: float, beta: float, gamma: float,
                    cutoff: float | None = None
                    ) -> jnp.ndarray:
    """GBSA-OBC energy (kJ/mol) for a single configuration ``x: (N, 3)``.

    ``cutoff`` is a static Python scalar; ``None`` emits the vacuum graph.
    """
    n = x.shape[0]
    # Safe pairwise geometry: dist has a ~1.0 diagonal and an ~1e-6 nm
    # off-diagonal floor so the masked backward pass stays finite; dist2
    # is reused below instead of recomputing dist**2. See :mod:`geom`.
    diff, dist2, dist = pairwise(x)                    # (N,N,3), (N,N), (N,N)
    eye = jnp.eye(n, dtype=x.dtype)

    B, radius = _born_radii(dist, offset_radius, scaled_radius,
                            dielectric_offset, alpha, beta, gamma, cutoff)

    # Self energy.
    e_self = jnp.sum(0.5 * prefactor * (charge ** 2) / B)

    # ACE surface area (Schaefer-Karplus form).
    r_ace = radius + probe_radius
    e_sa = jnp.sum(sa_factor * r_ace ** 2 * (radius / B) ** 6)

    # Pair GB sum (upper triangular; masked and -1/rc shifted under a cutoff).
    bi = B[:, None]
    bj = B[None, :]
    alpha2 = bi * bj
    # r2 == dist2 + eye (the safe-distance identity term); reuse dist2
    # directly instead of squaring dist. The diagonal extra (eye) is
    # masked out by `upper` below, so it never enters e_pair.
    r2 = dist2 + eye
    D = r2 / (4.0 * jnp.maximum(alpha2, 1e-15))
    exp_term = jnp.exp(-D)
    denom = jnp.sqrt(jnp.maximum(r2 + alpha2 * exp_term, 1e-30))
    # OpenMM shifts the in-range GB pair term by -1/rc, mirroring the reaction
    # field's constant. Unlike Coulomb this does NOT vanish at r = rc (f > rc
    # there), so the GB pair term carries a small jump; that is OpenMM's own
    # behavior, reproduced rather than smoothed.
    inv_rc = 0.0 if cutoff is None else 1.0 / cutoff
    Gpol = prefactor * qq * (1.0 / jnp.maximum(denom, 1e-15) - inv_rc)
    upper = jnp.triu(jnp.ones((n, n), dtype=jnp.bool_), k=1)
    if cutoff is not None:
        upper = upper & (dist < cutoff)
    e_pair = jnp.sum(jnp.where(upper, Gpol, 0.0))

    return e_self + e_sa + e_pair
