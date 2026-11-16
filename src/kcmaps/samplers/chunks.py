"""On-disk format of an adaptive-chains simulation directory, written by the runner and read by evaluation.

    run_spec.json           checkpoint, system section, sampler settings, seed and runtime (must match on resume)
    restart.npz             the carry after the last committed chunk, one array per carry field
    state.json              ``chunks``, the number of committed chunks; written last
    chunks/chunk_NNNNN.npz  per block (block, chain): positions (nm), end_ticks (1/16 fs), alive; per chain: redraws,
                            below_tau_max; scalar wall_cumulative_seconds (sampler execution time at the chunk's end,
                            compilation excluded); optional chain_ids (column,) int64 naming the chain of each
                            column (default the column index; the sampler writes every chain in every chunk)
    summary.json            counters totalled over all committed chunks; status and compile_seconds of the last
                            invocation

Every file is written to a temporary name and renamed into place.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

COUNTER_FIELDS = ("redraws", "below_tau_max")
FINISHED = ("COMPLETE", "INCOMPLETE")


def chunk_path(run_dir: Path, index: int) -> Path:
    """Path of chunk ``index``."""
    return Path(run_dir) / "chunks" / f"chunk_{index:05d}.npz"


def write_json(path: Path, value) -> None:
    """Atomically write ``value`` as JSON."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _savez(path: Path, **arrays) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def read_npz(path: Path) -> dict:
    """All arrays of an ``.npz`` file."""
    with np.load(path, allow_pickle=False) as stored:
        return {name: stored[name] for name in stored.files}


def check_output(output: Path, resume: bool) -> None:
    """Raise ValueError unless ``output`` holds a run to resume (``resume``) or no output yet (hidden entries and
    Hydra's ``*.log`` files do not count)."""
    if resume and not (output / "state.json").exists():
        raise ValueError(f"nothing to resume in {output}")
    if not resume and any(not p.name.startswith(".") and p.suffix != ".log" for p in output.iterdir()):
        raise ValueError(f"refusing to overwrite {output}; use an empty directory, or resume=true to continue a "
                         "stopped adaptive run")


def initialize(output: Path, spec: dict, restart: dict) -> dict:
    """Write the run_spec and restart files of a fresh run; return its state."""
    (output / "chunks").mkdir()
    write_json(output / "run_spec.json", spec)
    _savez(output / "restart.npz", **restart)
    state = {"chunks": 0}
    write_json(output / "state.json", state)
    return state


def load(output: Path, spec: dict, template: dict) -> tuple[dict, dict, float]:
    """Check a stopped run against ``spec``; return (restart arrays, state, the last chunk's
    wall_cumulative_seconds or 0.0).

    ``template`` gives the expected restart keys, dtypes and shapes.

    Raises:
        ValueError: a finished run, a different spec, chunk files other than the committed ones, or a restart that
            does not continue the last chunk.
    """
    summary = output / "summary.json"
    if summary.exists() and json.loads(summary.read_text())["status"] in FINISHED:
        raise ValueError("cannot resume a completed or failed run")
    stored_spec = json.loads((output / "run_spec.json").read_text())
    if stored_spec != spec:
        differing = sorted(k for k in set(spec) | set(stored_spec) if spec.get(k) != stored_spec.get(k))
        raise ValueError(f"resume needs the same checkpoint, sampler settings and runtime; differing: {differing}")
    state = json.loads((output / "state.json").read_text())
    expected = {chunk_path(output, i).name for i in range(state["chunks"])}
    actual = {path.name for path in (output / "chunks").iterdir()}
    if actual != expected:
        raise ValueError(f"chunks on disk differ from the {state['chunks']} committed ones: missing "
                         f"{sorted(expected - actual)}, unexpected {sorted(actual - expected)}")
    restart = read_npz(output / "restart.npz")
    if set(restart) != set(template) or any(restart[k].dtype != template[k].dtype
                                            or restart[k].shape != template[k].shape for k in template):
        raise ValueError("restart.npz does not match the carry layout")
    wall_cumulative_seconds = 0.0
    if state["chunks"]:
        last = read_npz(chunk_path(output, state["chunks"] - 1))
        if not (np.array_equal(last["end_ticks"][-1], restart["elapsed_ticks"])
                and np.array_equal(last["alive"][-1], restart["alive"])
                and np.array_equal(last["positions"][-1], restart["phase"][..., :3], equal_nan=True)):
            raise ValueError("the last chunk and restart.npz disagree")
        wall_cumulative_seconds = float(last["wall_cumulative_seconds"])
    return restart, state, wall_cumulative_seconds


def commit(output: Path, state: dict, frames: dict, counters: dict, wall_cumulative_seconds: float,
           restart: dict) -> None:
    """Write the next chunk and the restart, then ``state.json`` (``state`` is updated in place)."""
    path = chunk_path(output, state["chunks"])
    if path.exists():
        raise ValueError(f"refusing to replace {path}")
    _savez(path, **frames, **counters, wall_cumulative_seconds=np.asarray(wall_cumulative_seconds, np.float64))
    _savez(output / "restart.npz", **restart)
    state["chunks"] += 1
    # Written last: a crash before it leaves an uncommitted chunk file, which resume rejects.
    write_json(output / "state.json", state)


def committed_chunks(run_dir: str | Path) -> tuple[Path, ...]:
    """Paths of the committed chunks of ``run_dir`` in commit order.

    Raises:
        FileNotFoundError: ``state.json`` or a committed chunk is missing.
        ValueError: there are no committed chunks.
    """
    run_dir = Path(run_dir)
    n = json.loads((run_dir / "state.json").read_text())["chunks"]
    if not n:
        raise ValueError(f"{run_dir} has no committed chunks")
    paths = tuple(chunk_path(run_dir, i) for i in range(n))
    missing = [path.name for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"{run_dir} is missing committed chunks {missing}")
    return paths


def counter_totals(run_dir: Path, n_chains: int) -> dict:
    """Per-chain int64 sums of ``COUNTER_FIELDS`` over the committed chunks (zeros when there are none)."""
    totals = {key: np.zeros(n_chains, np.int64) for key in COUNTER_FIELDS}
    if json.loads((Path(run_dir) / "state.json").read_text())["chunks"]:
        for path in committed_chunks(run_dir):
            chunk = read_npz(path)
            for key in COUNTER_FIELDS:
                totals[key] += chunk[key]
    return totals


__all__ = ["COUNTER_FIELDS", "check_output", "chunk_path", "commit", "committed_chunks",
           "counter_totals", "initialize", "load", "read_npz", "write_json"]
