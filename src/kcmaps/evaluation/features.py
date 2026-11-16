"""Molecular features of Cartesian frames: energies, phi/psi torsions and TICA coordinates.

Generated frames are featurized here; reference feature files (``load_reference``) must hold the same features
computed with these definitions. The phi/psi order and the TICA feature layout follow transferable-samplers
(MIT; its TICA preprocessing is based on Klein and Noe 2024, CC BY 4.0).
"""

from __future__ import annotations

import json
import struct
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import mdtraj as md
import numpy as np

TICA_ATOM_SELECTION = "symbol == C or symbol == N or symbol == S"
TICA_FEATURE_SPEC = "cns_distances_sincos_phi_psi_omega"
TICA_ASSETS = Path(__file__).resolve().parents[1] / "assets" / "tica"
REFERENCE_KEYS = ("energy", "rama", "tica", "frames_per_trajectory", "frame_ps")


def _trajectory(positions: np.ndarray, topology: md.Topology) -> md.Trajectory:
    xyz = np.asarray(positions)
    if xyz.ndim != 3 or xyz.shape[1:] != (topology.n_atoms, 3):
        raise ValueError(f"positions must have shape (n, {topology.n_atoms}, 3); got {xyz.shape}")
    if not np.issubdtype(xyz.dtype, np.floating):
        xyz = xyz.astype(np.float64)
    if not np.isfinite(xyz).all():
        raise ValueError("positions contain non-finite coordinates")
    xyz = np.ascontiguousarray(xyz)
    if not xyz.flags.writeable:
        xyz = xyz.copy()
    return md.Trajectory(xyz=xyz, topology=topology)


def phi_psi(positions: np.ndarray, topology: md.Topology) -> np.ndarray:
    """Backbone torsions on the torus: every defined phi, then every defined psi (MDTraj order).

    Args:
        positions: ``(n, atoms, 3)`` coordinates in nm.
        topology: MDTraj topology in the atom order of ``positions``.

    Returns:
        ``(n, n_phi + n_psi)`` angles in radians.
    """
    traj = _trajectory(positions, topology)
    _, phi = md.compute_phi(traj)
    _, psi = md.compute_psi(traj)
    torus = np.concatenate((phi, psi), axis=1)
    if torus.shape[1] == 0 or not np.isfinite(torus).all():
        raise ValueError("phi/psi features are empty or non-finite")
    return np.ascontiguousarray(torus)


def tica_features(positions: np.ndarray, topology: md.Topology) -> np.ndarray:
    """TICA input features: upper-triangle pair distances between C/N/S atoms, then sin and cos of the
    phi, psi and omega torsions of the C/N/S sub-trajectory.

    Args:
        positions: ``(n, atoms, 3)`` coordinates in nm.
        topology: MDTraj topology in the atom order of ``positions``.

    Returns:
        ``(n, n_features)`` array; distances in nm.
    """
    traj = _trajectory(positions, topology)
    sel = np.asarray(topology.select(TICA_ATOM_SELECTION), np.int64)
    if sel.size == 0:
        raise ValueError("topology has no C/N/S atoms")
    sub = traj.atom_slice(sel)
    row, col = np.triu_indices(sub.xyz.shape[1], k=1)
    dist = np.linalg.norm(sub.xyz[:, row, :] - sub.xyz[:, col, :], axis=-1)
    _, phi = md.compute_phi(sub)
    _, psi = md.compute_psi(sub)
    _, omega = md.compute_omega(sub)
    dih = np.concatenate((np.sin(phi), np.cos(phi), np.sin(psi), np.cos(psi), np.sin(omega), np.cos(omega)), axis=-1)
    feats = np.concatenate((dist, dih), axis=-1)
    if feats.shape[1] == 0 or not np.isfinite(feats).all():
        raise ValueError("TICA features are empty or non-finite")
    return np.ascontiguousarray(feats)


@dataclass(frozen=True)
class TicaBasis:
    """A frozen linear TICA basis: ``tics = (features - mean) @ projection``, first two columns kept.

    Attributes:
        mean: ``(n_features,)`` float64.
        projection: ``(n_features, n_components)`` float64.
        metadata: the basis ``metadata.json`` content.
    """

    mean: np.ndarray
    projection: np.ndarray
    metadata: dict

    @classmethod
    def load(cls, path: str | Path) -> "TicaBasis":
        """Load a basis directory (``arrays.npz`` + ``metadata.json``).

        Raises:
            ValueError: the arrays are not a float64 mean and projection of matching width, or the feature
                specification differs from :data:`TICA_FEATURE_SPEC`.
        """
        path = Path(path)
        meta = json.loads((path / "metadata.json").read_text())
        with np.load(path / "arrays.npz") as z:
            mean, proj = z["mean"], z["projection"]
        m = meta["metadata"]
        if m["features"]["specification"] != TICA_FEATURE_SPEC or m["coordinate_units"] != "nm":
            raise ValueError(f"{path} is not a {TICA_FEATURE_SPEC} basis in nm")
        if mean.dtype != np.float64 or proj.dtype != np.float64 or mean.ndim != 1 or proj.ndim != 2 \
                or proj.shape[0] != mean.shape[0] or proj.shape[1] < 2:
            raise ValueError(f"{path} has inconsistent arrays")
        return cls(mean=mean, projection=proj, metadata=meta)

    @classmethod
    def packaged(cls, system: str) -> "TicaBasis":
        """The basis shipped with the package for ``system`` (AL2, AL4 or AL6)."""
        path = TICA_ASSETS / system
        if not path.is_dir():
            raise ValueError(f"no packaged TICA basis for {system!r}")
        return cls.load(path)

    def check_topology(self, topology: md.Topology) -> None:
        """Raise ValueError unless ``topology`` has the basis's atom count and atom names and selects the same C/N/S
        atoms."""
        t = self.metadata["metadata"]["topology"]
        sel = np.asarray(topology.select(TICA_ATOM_SELECTION), np.int64).tolist()
        names = [atom.name for atom in topology.atoms]
        if t["n_atoms"] != topology.n_atoms or t["atom_names"] != names or t["selected_atom_indices"] != sel:
            raise ValueError("TICA basis topology does not match the system")

    def transform(self, features: np.ndarray) -> np.ndarray:
        """``(n, n_features)`` features -> ``(n, 2)`` TIC0, TIC1."""
        f = np.asarray(features)
        if f.ndim != 2 or f.shape[1] != self.mean.shape[0]:
            raise ValueError(f"features must have shape (n, {self.mean.shape[0]}); got {f.shape}")
        out = (f - self.mean) @ self.projection
        if not np.isfinite(out).all():
            raise ValueError("TICA projection is non-finite")
        return np.ascontiguousarray(out[:, :2])


def tica_coordinates(positions: np.ndarray, topology: md.Topology, basis: TicaBasis, *, batch: int) -> np.ndarray:
    """TIC0, TIC1 of ``positions`` (nm), featurized and projected ``batch`` frames at a time."""
    out = [basis.transform(tica_features(positions[a:a + batch], topology)) for a in range(0, len(positions), batch)]
    return np.concatenate(out, axis=0)


def energies(positions: np.ndarray, energy_fn: Callable, *, batch: int) -> np.ndarray:
    """Reduced potential energy U/kBT of each frame, evaluated in float64 ``batch`` frames at a time.

    Args:
        positions: ``(n, atoms, 3)`` coordinates in nm.
        energy_fn: batched ``(b, atoms, 3) -> (b,)`` energy; JAX x64 must be enabled.

    Raises:
        ValueError: x64 is disabled, or the output is not finite float64 of shape ``(b,)``.
    """
    import jax
    import jax.numpy as jnp

    if not jax.config.jax_enable_x64:
        raise ValueError("energies require JAX x64")
    out = []
    for a in range(0, len(positions), batch):
        x = positions[a:a + batch]
        u = np.asarray(energy_fn(jnp.asarray(x, dtype=jnp.float64)))
        if u.shape != (len(x),) or u.dtype != np.float64 or not np.isfinite(u).all():
            raise ValueError(f"energy_fn returned {u.dtype} {u.shape}; expected finite float64 ({len(x)},)")
        out.append(u)
    return np.concatenate(out)


def featurize(
    positions: np.ndarray,
    *,
    topology: md.Topology,
    energy_fn: Callable,
    basis: TicaBasis,
    energy_batch: int,
    tica_batch: int,
) -> dict[str, np.ndarray]:
    """Equilibrium features of ``positions`` (nm): ``energy`` (U/kBT), ``rama`` (phi/psi torus, rad) and
    ``tica`` (TIC0, TIC1)."""
    return dict(energy=energies(positions, energy_fn, batch=energy_batch), rama=phi_psi(positions, topology),
                tica=tica_coordinates(positions, topology, basis, batch=tica_batch))


def _npz_member(path: Path, key: str) -> np.ndarray:
    """Read-only array ``key`` of an npz, memory-mapped when the member is stored uncompressed."""
    with zipfile.ZipFile(path) as zf:
        info = zf.getinfo(f"{key}.npy")
    if info.compress_type != zipfile.ZIP_STORED:
        with np.load(path) as z:
            return z[key]
    with open(path, "rb") as f:
        f.seek(info.header_offset)
        head = f.read(30)
        name_len, extra_len = struct.unpack("<HH", head[26:30])
        f.seek(info.header_offset + 30 + name_len + extra_len)
        version = np.lib.format.read_magic(f)
        reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
        shape, fortran, dtype = reader(f)
        offset = f.tell()
    return np.memmap(path, dtype=dtype, mode="r", offset=offset, shape=shape, order="F" if fortran else "C")


def load_reference(path: str | Path) -> dict[str, np.ndarray]:
    """Reference feature file: ``energy`` (n,) U/kBT, ``rama`` (n, torsions) rad, ``tica`` (n, >= 2), frames sorted by
    trajectory, ``frames_per_trajectory`` (trajectories,) int64 summing to n, and the scalar frame interval
    ``frame_ps``. Arrays are memory-mapped when stored uncompressed.

    Raises:
        ValueError: a key is missing, the file is empty, the arrays have different lengths, ``rama``/``tica`` lack
            columns, or the trajectory lengths or frame interval are invalid.
    """
    path = Path(path)
    with zipfile.ZipFile(path) as zf:
        names = {n[:-4] for n in zf.namelist() if n.endswith(".npy")}
    missing = set(REFERENCE_KEYS) - names
    if missing:
        raise ValueError(f"{path} lacks {sorted(missing)}")
    ref = {k: _npz_member(path, k) for k in REFERENCE_KEYS}
    n = len(ref["energy"])
    if n == 0 or any(len(ref[k]) != n for k in ("rama", "tica")) or ref["energy"].ndim != 1 or ref["rama"].ndim != 2 \
            or ref["rama"].shape[1] == 0 or ref["tica"].ndim != 2 or ref["tica"].shape[1] < 2:
        raise ValueError(f"{path} arrays are inconsistent")
    counts, frame_ps = np.asarray(ref["frames_per_trajectory"]), np.asarray(ref["frame_ps"])
    if counts.ndim != 1 or counts.dtype != np.int64 or not (counts > 0).all() or counts.sum() != n:
        raise ValueError(f"{path} frames_per_trajectory must be positive int64 summing to {n}")
    if frame_ps.shape != () or not (np.isfinite(frame_ps) and frame_ps > 0):
        raise ValueError(f"{path} frame_ps must be a positive scalar")
    ref["frames_per_trajectory"], ref["frame_ps"] = counts, frame_ps
    return ref
