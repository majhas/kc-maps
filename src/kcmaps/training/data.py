"""Training positions: checked load, mass-weighted centering, source-frame validity, uniform batches and the
position scale x_std. ``python -m kcmaps.training.data`` builds train.npz and val.npz from the source datasets."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from ..core import constants as C
from ..models.geometry import source_frame_valid

_CHUNK = 8192
POSITIONS_KEY = "positions"  # array name of the (frames, N, 3) positions in a data file


def center_frames(frames: np.ndarray, masses: np.ndarray) -> None:
    """Subtract each frame's mass-weighted center in place, accumulating in float64."""
    mass64 = np.asarray(masses, np.float64)
    total = float(mass64.sum())
    for start in range(0, frames.shape[0], _CHUNK):
        block = frames[start:start + _CHUNK].astype(np.float64)
        block -= np.einsum("fai,a->fi", block, mass64)[:, None, :] / total
        frames[start:start + _CHUNK] = block


@jax.jit
def _frame_validity(positions, anchors):
    return source_frame_valid(positions, anchors, C.SOURCE_FRAME_MIN_BOND_NM, C.SOURCE_FRAME_MIN_SIN)


def check_source_frames(positions, anchors) -> None:
    """Raise ValueError naming the first frame whose N-CA-C anchors are not a valid frame."""
    anchors = jnp.asarray(anchors)
    for start in range(0, len(positions), _CHUNK):
        valid = np.asarray(_frame_validity(positions[start:start + _CHUNK], anchors))
        if not valid.all():
            raise ValueError(f"invalid N-CA-C source frame in training frame {start + int(np.flatnonzero(~valid)[0])}")


@dataclasses.dataclass(frozen=True, eq=False)
class Dataset:
    """Centered float32 training positions on device.

    Attributes:
        positions: (frames, N, 3) nm, mass-weighted centered.
        n_frames: number of frames.
        start_position: (N, 3) float32 nm, the start configuration of the samplers (not centered).
    """

    positions: jax.Array
    n_frames: int
    start_position: np.ndarray

    def sample_batch(self, key, batch_size: int):
        """(batch_size, N, 3) positions of frames drawn uniformly with replacement."""
        return jnp.take(self.positions, jax.random.randint(key, (int(batch_size),), 0, self.n_frames), axis=0)


def load_dataset(path, system) -> Dataset:
    """Load, verify and center the (frames, N, 3) positions ``POSITIONS_KEY`` of an ``.npz`` file.

    Args:
        path: the ``.npz`` file.
        system: ``PeptideSystem`` (masses and source-frame anchors).

    Raises:
        KeyError: the file has no ``POSITIONS_KEY`` array.
        TypeError: the array is not floating.
        ValueError: not an ``.npz`` file, wrong shape, non-finite values or an invalid source frame.
    """
    path = Path(path)
    if path.suffix != ".npz":
        raise ValueError(f"training data must be an .npz archive, got {path}")
    with np.load(path, allow_pickle=False) as archive:
        if POSITIONS_KEY not in archive.files:
            raise KeyError(f"{POSITIONS_KEY!r} is not in {path}; keys: {sorted(archive.files)}")
        positions = np.array(archive[POSITIONS_KEY])
    masses = np.asarray(system.masses, np.float64)
    if positions.ndim != 3 or positions.shape[1:] != (masses.size, 3) or positions.shape[0] == 0:
        raise ValueError(f"training positions must have shape (frames, {masses.size}, 3), got {positions.shape}")
    if not np.issubdtype(positions.dtype, np.floating):
        raise TypeError(f"training positions must be floating, got {positions.dtype}")
    if not all(np.isfinite(positions[s:s + _CHUNK]).all() for s in range(0, positions.shape[0], _CHUNK)):
        raise ValueError("training positions contain non-finite values")
    start_position = np.array(positions[0], np.float32)
    center_frames(positions, masses)
    positions = positions.astype(np.dtype(jax.dtypes.canonicalize_dtype(positions.dtype)), copy=False)
    device_positions = jnp.asarray(positions)
    check_source_frames(device_positions, system.anchors)
    return Dataset(positions=device_positions, n_frames=int(positions.shape[0]), start_position=start_position)


def fit_x_std(positions) -> tuple[float, dict]:
    """Scalar RMS of centered positions about the origin, sqrt(sum x^2 / (frames N 3)), in float64.

    Args:
        positions: (frames, N, 3) centered positions (nm), e.g. ``Dataset.positions``.

    Returns:
        (x_std in nm, record ``{shape}``) as stored in ``kcm.json``.

    Raises:
        ValueError: x_std is not positive.
    """
    array = np.asarray(positions)
    square_sum = np.float64(0.0)
    for start in range(0, array.shape[0], _CHUNK):
        block = array[start:start + _CHUNK].astype(np.float64)
        square_sum += np.sum(block * block, dtype=np.float64)
    value = float(np.sqrt(square_sum / np.float64(math.prod(array.shape))))
    if not value > 0.0:
        raise ValueError("x_std must be positive")
    return value, {"shape": [int(n) for n in array.shape]}


def _check_sha256(path, expected: str) -> None:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    if digest.hexdigest() != expected:
        raise ValueError(f"sha256 of {path} is {digest.hexdigest()}, splits.json records {expected}")


def build_splits(system: str, sources, splits_dir, out_dir) -> dict[str, int]:
    """Write ``out_dir/<system>/{train,val}.npz`` (``positions``, float32 nm) from the source files.

    The pool is the source frames in the order recorded in ``splits_dir/splits.json`` (AL2: ``traj0.h5``; AL4, AL6:
    the train, val and test ``.npy`` files concatenated); train and val are the first two slices of
    ``pool[<system>_split_indices.npy]`` with the recorded counts.

    Returns:
        Frames written per split.
    """
    splits_dir, out_dir = Path(splits_dir), Path(out_dir)
    record = json.loads((splits_dir / "splits.json").read_text())["systems"][system]
    index_file = splits_dir / f"{system}_split_indices.npy"
    _check_sha256(index_file, record["sha256"])
    source = record["source"]
    if "files" in source:
        if len(sources) != len(source["files"]):
            raise ValueError(f"{system} needs the files {source['files']} in this order")
        for path, split in zip(sources, ("train", "val", "test")):
            _check_sha256(path, source["raw_sha256"][split])
        pool = np.concatenate([np.load(path, allow_pickle=False).astype(np.float32) for path in sources])
    else:
        import h5py

        if len(sources) != 1:
            raise ValueError(f"{system} needs the one file {source['file']}")
        _check_sha256(sources[0], source["raw_sha256"])
        with h5py.File(sources[0], "r") as handle:
            pool = np.asarray(handle[source["key"]], np.float32)
    index = np.load(index_file, allow_pickle=False)
    if len(index) != len(pool):
        raise ValueError(f"{len(pool)} source frames, {len(index)} split indices")
    (out_dir / system).mkdir(parents=True, exist_ok=True)
    start, written = 0, {}
    for split in ("train", "val"):
        stop = start + record["counts"][split]
        np.savez(out_dir / system / f"{split}.npz", **{POSITIONS_KEY: pool[index[start:stop]]})
        written[split] = stop - start
        start = stop
    return written


def main(argv=None) -> None:
    data_dir = os.environ.get("KCMAPS_DATA", str(Path(os.environ.get("KCMAPS_ROOT", ".")) / "data"))
    parser = argparse.ArgumentParser(prog="python -m kcmaps.training.data", description=build_splits.__doc__)
    parser.add_argument("system", choices=["AL2", "AL4", "AL6"])
    parser.add_argument("sources", nargs="+", help="AL2: traj0.h5; AL4, AL6: the train, val and test .npy files")
    parser.add_argument("--splits", default=str(Path(data_dir) / "splits"), help="default: $KCMAPS_DATA/splits")
    parser.add_argument("--out", default=data_dir, help="default: $KCMAPS_DATA")
    args = parser.parse_args(argv)
    for split, n in build_splits(args.system, args.sources, args.splits, args.out).items():
        print(f"wrote {Path(args.out) / args.system / f'{split}.npz'}: {n} frames")


if __name__ == "__main__":
    main()
