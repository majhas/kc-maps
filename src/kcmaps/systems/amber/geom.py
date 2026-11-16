"""Distance primitives with a finite gradient at zero separation (an eps floor inside the sqrt)."""

from __future__ import annotations

import jax.numpy as jnp

# Under-sqrt floor → distance floor ~1e-6 nm. Negligible vs bond lengths
# (~0.1 nm); keeps d(sqrt)/d(.) finite at zero displacement.
_DIST_EPS2 = 1e-12


def safe_norm(v: jnp.ndarray, axis: int = -1, keepdims: bool = False) -> jnp.ndarray:
    """``sqrt(Σ v² + eps)`` — Euclidean norm with a finite gradient at v=0."""
    return jnp.sqrt(jnp.sum(v * v, axis=axis, keepdims=keepdims) + _DIST_EPS2)


def pairwise(x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return ``(diff, dist2, dist)`` for an ``(N, 3)`` configuration.

    ``diff[i,j] = x[i] - x[j]``; ``dist2 = Σ diff²``; ``dist = sqrt(dist2 +
    I + eps)``. The identity term makes the diagonal exactly ~1.0; the
    ``eps`` floor additionally guards coincident *off-diagonal* pairs
    (e.g. two atoms collapsed in an out-of-distribution proposal) so the masked
    backward pass cannot produce ``0 · inf = NaN``.
    """
    n = x.shape[0]
    diff = x[:, None, :] - x[None, :, :]
    dist2 = jnp.sum(diff * diff, axis=-1)
    eye = jnp.eye(n, dtype=x.dtype)
    dist = jnp.sqrt(dist2 + eye + _DIST_EPS2)
    return diff, dist2, dist
