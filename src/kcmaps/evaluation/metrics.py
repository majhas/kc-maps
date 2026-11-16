"""Exact optimal-transport Wasserstein-2 distances between generated and reference features (POT).

Energy W2 is the exact 1-D distance on U/kBT; Ramachandran W2 uses the squared torus distance summed over
every phi and psi column; TICA W2 uses squared Euclidean distance on TIC0, TIC1. Both point clouds carry
uniform weights.
"""

from __future__ import annotations

from typing import Callable, Mapping, Sequence

import numpy as np

EMD_MAX_ITER = 1_000_000_000
FLOOR_SEED = 900  # floor draw d uses default_rng(FLOOR_SEED + d)


def torus_cost(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """``(n, m)`` squared torus distance: sum over columns of the squared wrapped angle difference."""
    c = np.zeros((len(x), len(y)))
    for j in range(x.shape[1]):
        # one column at a time keeps the temporary at (n, m)
        d = np.abs(x[:, j, None] - y[None, :, j])
        c += np.minimum(d, 2 * np.pi - d) ** 2
    return c


def sq_euclidean_cost(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """``(n, m)`` squared Euclidean distance."""
    import ot

    return ot.dist(x, y, metric="sqeuclidean")


def ot_w2(x: np.ndarray, y: np.ndarray, cost: Callable[[np.ndarray, np.ndarray], np.ndarray]) -> float:
    """Exact W2 between uniform point clouds ``x`` and ``y`` under the squared ground cost ``cost``."""
    import ot

    m = cost(np.asarray(x).astype(np.float64), np.asarray(y).astype(np.float64))
    return float(np.sqrt(ot.emd2(ot.unif(len(x)), ot.unif(len(y)), m, numItermax=EMD_MAX_ITER)))


def energy_w2(ref: np.ndarray, gen: np.ndarray) -> float:
    """Exact 1-D W2 between reduced energies."""
    import ot

    return float(np.sqrt(ot.emd2_1d(np.asarray(ref).astype(np.float64), np.asarray(gen).astype(np.float64))))


def check_features(ref: Mapping[str, np.ndarray], gen: Mapping[str, np.ndarray]) -> None:
    """Require finite, non-empty ``energy`` (n,), ``rama`` (n, k >= 1) and ``tica`` (n, >= 2) in both sets, with
    the same ``rama`` width.

    Raises:
        ValueError: on the first violation.
    """
    for label, features in (("reference", ref), ("generated", gen)):
        energy, rama, tica = (np.asarray(features[k]) for k in ("energy", "rama", "tica"))
        n = len(energy)
        if energy.ndim != 1 or n == 0:
            raise ValueError(f"{label} energy must be a non-empty (n,) array, got {energy.shape}")
        if rama.ndim != 2 or rama.shape[0] != n or rama.shape[1] == 0:
            raise ValueError(f"{label} rama must have shape ({n}, k >= 1), got {rama.shape}")
        if tica.ndim != 2 or tica.shape[0] != n or tica.shape[1] < 2:
            raise ValueError(f"{label} tica must have shape ({n}, >= 2), got {tica.shape}")
        for name, array in (("energy", energy), ("rama", rama), ("tica", tica[:, :2])):
            if not np.isfinite(array).all():
                raise ValueError(f"{label} {name} has non-finite values")
    if np.shape(ref["rama"])[1] != np.shape(gen["rama"])[1]:
        raise ValueError(f"rama widths differ: reference {np.shape(ref['rama'])[1]}, generated "
                         f"{np.shape(gen['rama'])[1]}")


def w2_metrics(ref: Mapping[str, np.ndarray], gen: Mapping[str, np.ndarray]) -> dict[str, float]:
    """``energy_w2``, ``rama_w2`` and ``tica_w2`` between two feature sets (keys energy, rama, tica).

    Raises:
        ValueError: the feature sets fail ``check_features``.
    """
    check_features(ref, gen)
    return dict(energy_w2=energy_w2(ref["energy"], gen["energy"]),
                rama_w2=ot_w2(ref["rama"], gen["rama"], torus_cost),
                tica_w2=ot_w2(ref["tica"][:, :2], gen["tica"][:, :2], sq_euclidean_cost))


def summarize_draws(draws: Sequence[Mapping[str, float]]) -> dict:
    """Mean, sample sd (ddof=1) and the per-draw values of each metric."""
    out = {}
    for k in draws[0]:
        v = np.array([d[k] for d in draws], np.float64)
        out[k] = dict(mean=float(v.mean()), sd=float(v.std(ddof=1)), values=v.tolist())
    return out


def reference_floor(reference: Mapping[str, np.ndarray], *, n_frames: int, n_draws: int) -> dict:
    """W2 between two disjoint uniform draws of ``n_frames`` reference frames.

    Draw d uses ``numpy.random.default_rng(FLOOR_SEED + d)``: set A uniformly from all frames, then set B
    uniformly from the frames not in A.
    """
    n = len(reference["energy"])
    if 2 * n_frames > n:
        raise ValueError(f"reference has {n} frames; the floor needs {2 * n_frames}")
    draws = []
    for d in range(n_draws):
        rng = np.random.default_rng(FLOOR_SEED + d)
        a = rng.choice(n, n_frames, replace=False)
        b = rng.choice(np.setdiff1d(np.arange(n), a), n_frames, replace=False)
        draws.append(w2_metrics({k: np.asarray(reference[k][a]) for k in ("energy", "rama", "tica")},
                                {k: np.asarray(reference[k][b]) for k in ("energy", "rama", "tica")}))
    return summarize_draws(draws)
