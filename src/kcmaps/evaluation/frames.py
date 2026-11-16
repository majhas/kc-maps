"""Frame selection from an adaptive-chains simulation directory (format in ``kcmaps.samplers.chunks``).

Every frame is indexed by (chunk, block, column) with its chain, its simulated time, the budget clock at which it was
committed and whether it is usable (its chain alive).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from kcmaps.core.constants import TICK_PS
from kcmaps.samplers.chunks import committed_chunks


@dataclass(frozen=True)
class FrameIndex:
    """Per-frame index of a simulation directory, rows in (chunk, block, column) order.

    Attributes:
        chunk_paths: committed chunk files in commit order.
        chunk, block, column: int64 arrays locating each frame.
        chain: int64 chain id of the frame (the chunk's ``chain_ids`` of its column; default the column).
        time_ps: float64 simulated time of the frame.
        budget_s: float64 budget clock (seconds) at which the frame was committed.
        usable: bool, chain alive at the frame.
    """

    chunk_paths: tuple[Path, ...]
    chunk: np.ndarray
    block: np.ndarray
    column: np.ndarray
    chain: np.ndarray
    time_ps: np.ndarray
    budget_s: np.ndarray
    usable: np.ndarray

    def __len__(self) -> int:
        return int(self.chunk.shape[0])


def chunk_clock(run_dir: str | Path) -> np.ndarray:
    """Budget clock at the end of each committed chunk: its ``wall_cumulative_seconds`` (compilation excluded).

    Returns:
        float64 array of shape ``(n_chunks,)``, non-decreasing.
    """
    ends = []
    for path in committed_chunks(run_dir):
        with np.load(path) as z:
            ends.append(float(z["wall_cumulative_seconds"]))
    ends = np.asarray(ends, np.float64)
    if np.any(np.diff(ends) < 0):
        raise ValueError(f"wall_cumulative_seconds of {run_dir} decreases between chunks")
    return ends


def index_frames(run_dir: str | Path) -> FrameIndex:
    """Index every frame of a simulation directory.

    The budget clock runs linearly over the blocks of a chunk, from the previous chunk's end to this
    chunk's end, so block k of n (1-based) is committed at ``prev + k / n * (end - prev)``.

    Args:
        run_dir: simulation directory; the chunk ends come from :func:`chunk_clock`.

    Returns:
        A :class:`FrameIndex`.
    """
    run_dir = Path(run_dir)
    paths = committed_chunks(run_dir)
    ends = chunk_clock(run_dir)
    cols = {k: [] for k in ("chunk", "block", "column", "chain", "time_ps", "budget_s", "usable")}
    prev = 0.0
    for c, path in enumerate(paths):
        with np.load(path) as z:
            ticks, alive = z["end_ticks"].astype(np.float64), z["alive"]
            ids = z["chain_ids"].astype(np.int64) if "chain_ids" in z.files else np.arange(ticks.shape[1])
        end = float(ends[c])
        nb, nc = ticks.shape
        if ids.shape != (nc,):
            raise ValueError(f"{path}: chain_ids must have one entry per column")
        k = np.arange(1, nb + 1)[:, None]
        budget = np.broadcast_to(prev + k / nb * (end - prev), (nb, nc))
        b, col = np.meshgrid(np.arange(nb), np.arange(nc), indexing="ij")
        cols["chunk"].append(np.full(b.size, c, np.int64))
        cols["block"].append(b.ravel().astype(np.int64))
        cols["column"].append(col.ravel().astype(np.int64))
        cols["chain"].append(ids[col].ravel())
        cols["time_ps"].append((ticks * TICK_PS).ravel())
        cols["budget_s"].append(budget.ravel())
        cols["usable"].append(np.asarray(alive, bool).ravel())
        prev = end
    return FrameIndex(chunk_paths=paths, **{k: np.concatenate(v) for k, v in cols.items()})


def frame_intervals(index: FrameIndex) -> np.ndarray:
    """Simulated time (ps) each frame closes: the time since the previous frame of its chain.

    A chain's first frame gets the median interval of that chain (1.0 for a single-frame chain). Frames
    that repeat the previous one (a redrawn block) get 0.
    """
    dt = np.empty(len(index))
    for ch in np.unique(index.chain):
        m = np.flatnonzero(index.chain == ch)
        d = np.diff(index.time_ps[m])
        if not (d >= -1e-9).all():
            raise ValueError(f"simulated time of chain {ch} decreases")
        dt[m] = np.r_[np.median(d) if len(d) else 1.0, np.clip(d, 0, None)]
    return dt


def select_frames(index: FrameIndex, *, budget_s: float, burn_in_fraction: float) -> np.ndarray:
    """Rows usable within the budget after the per-chain burn-in.

    A row is kept when it was committed within ``budget_s``, its chain is alive, and its time exceeds
    ``burn_in_fraction`` times the largest time its chain reached within the budget.

    Returns:
        Sorted int64 row indices into ``index``.
    """
    within = index.budget_s <= budget_s
    keep = np.zeros(len(index), bool)
    for ch in np.unique(index.chain[within]):
        m = within & (index.chain == ch)
        keep |= m & index.usable & (index.time_ps > burn_in_fraction * index.time_ps[m].max())
    return np.flatnonzero(keep)


def pooled_simulated_ns(index: FrameIndex, *, budget_s: float) -> float:
    """Sum over chains of the largest simulated time reached within the budget, in ns."""
    within = index.budget_s <= budget_s
    if not within.any():
        return 0.0
    return float(sum(index.time_ps[within & (index.chain == ch)].max() for ch in np.unique(index.chain[within])) / 1000)


def draw_rows(
    pool: np.ndarray,
    weights: np.ndarray,
    *,
    n_frames: int,
    n_reference: int,
    n_draws: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generated and reference rows of each draw.

    Draw d uses ``numpy.random.default_rng(seed + d)``: first ``min(n_frames, len(pool))`` generated rows
    without replacement with probability proportional to ``weights``, then ``n_frames`` reference rows
    uniformly without replacement from the same generator.

    Args:
        pool: candidate generated rows (zero-weight rows removed by the caller).
        weights: positive weight of each pool row.
        n_frames: frames per side.
        n_reference: number of reference frames.
        n_draws: number of draws.
        seed: seed of draw 0.

    Returns:
        ``[(sorted generated rows, reference rows)]`` per draw.
    """
    pool = np.asarray(pool)
    weights = np.asarray(weights, np.float64)
    if pool.shape != weights.shape or not (weights > 0).all():
        raise ValueError("pool and weights must have the same shape and positive weights")
    if n_reference < n_frames:
        raise ValueError(f"reference has {n_reference} frames; {n_frames} are needed")
    n_gen = min(n_frames, len(pool))
    out = []
    for d in range(n_draws):
        rng = np.random.default_rng(seed + d)
        gen = np.sort(rng.choice(pool, n_gen, replace=False, p=weights / weights.sum()))
        out.append((gen, rng.choice(n_reference, n_frames, replace=False)))
    return out


def load_positions(index: FrameIndex, rows: np.ndarray) -> np.ndarray:
    """Positions (nm, float32) of sorted unique ``rows``, shape ``(len(rows), atoms, 3)``."""
    rows = np.asarray(rows)
    if np.any(np.diff(rows) <= 0):
        raise ValueError("rows must be sorted and unique")
    xyz = None
    for c in np.unique(index.chunk[rows]):
        sel = rows[index.chunk[rows] == c]
        with np.load(index.chunk_paths[int(c)]) as z:
            a = z["positions"]
        if xyz is None:
            xyz = np.empty((len(rows),) + a.shape[2:], np.float32)
        xyz[np.searchsorted(rows, sel)] = a[index.block[sel], index.column[sel]]
    if xyz is None:
        raise ValueError("no rows to load")
    if not np.isfinite(xyz).all():
        raise ValueError("selected positions contain non-finite values")
    return xyz


def chain_rows(index: FrameIndex, *, budget_s: float, burn_in_fraction: float) -> list[np.ndarray]:
    """Per chain, its usable rows within the budget in time order, without frames whose time equals the previous
    frame's (no simulated-time progress, e.g. a stopped chain), after dropping the first
    ``floor(burn_in_fraction * n)`` of its ``n`` such rows."""
    within = (index.budget_s <= budget_s) & index.usable
    out = []
    for ch in np.unique(index.chain):
        rows = np.flatnonzero(index.chain == ch)
        progress = np.r_[True, np.diff(index.time_ps[rows]) > 0]
        m = rows[progress & within[rows]]
        out.append(m[int(np.floor(burn_in_fraction * len(m))):])
    return out
