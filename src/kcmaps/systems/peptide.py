"""Peptide physics: packaged topology, masses, thermodynamics, Amber energy and score, structural encodings.

Units: positions nm, masses amu, kBT kJ/mol, friction 1/ps, time ps.
"""

from __future__ import annotations

import dataclasses
import io
import math
from pathlib import Path

import numpy as np

from .amber.topology import BOLTZMANN_CONSTANT_KJ_PER_MOL_K

PDB_DIR = Path(__file__).resolve().parents[1] / "assets" / "pdb"

_ELEMENT_MASSES = {"H": 1.008, "C": 12.011, "N": 14.007, "O": 15.999, "S": 32.06, "P": 30.974}

# Embedding ids start at 1 (0 is padding); they index trained embedding tables, so the maps are append-only.
ATOM_TYPES = {
    "C": 1, "CA": 2, "CB": 3, "CD": 4, "CD1": 5, "CD2": 6, "CE": 7, "CE1": 8, "CE2": 9, "CE3": 10, "CG": 11,
    "CG1": 12, "CG2": 13, "CH2": 14, "CZ": 15, "CZ2": 16, "CZ3": 17, "H": 18, "HA": 19, "HB": 20, "HD": 21,
    "HD1": 22, "HD2": 23, "HE": 24, "HE1": 25, "HE2": 26, "HE3": 27, "HG": 28, "HG1": 29, "HG2": 30, "HH": 31,
    "HH1": 32, "HH2": 33, "HZ": 34, "HZ2": 35, "HZ3": 36, "N": 37, "ND1": 38, "ND2": 39, "NE": 40, "NE1": 41,
    "NE2": 42, "NH1": 43, "NH2": 44, "NZ": 45, "O": 46, "OD": 47, "OE": 48, "OG": 49, "OG1": 50, "OH": 51,
    "OXT": 52, "SD": 53, "SG": 54, "CH3": 55,
}
RESIDUE_TYPES = {
    "ALA": 1, "ARG": 2, "ASN": 3, "ASP": 4, "CYS": 5, "GLN": 6, "GLU": 7, "GLY": 8, "HIS": 9, "ILE": 10,
    "LEU": 11, "LYS": 12, "MET": 13, "PHE": 14, "PRO": 15, "SER": 16, "THR": 17, "TRP": 18, "TYR": 19, "VAL": 20,
    "ACE": 21, "NME": 22,
}
# Embedding table sizes (parameter shapes), with headroom over the vocabularies.
NUM_ATOM_TYPES = 64
NUM_RESIDUE_TYPES = 32
NUM_ATOM_SUFFIXES = 4  # 0: none; 1/2/3: the digit removed from the atom name


def packaged_pdb(name: str) -> Path:
    """Return the path of a packaged PDB.

    Raises:
        FileNotFoundError: no packaged PDB of that name.
    """
    path = PDB_DIR / name
    if Path(name).name != name or not path.is_file():
        raise FileNotFoundError(f"no packaged PDB {name!r} in {PDB_DIR}")
    return path


def _atom_element(atom_name: str) -> str:
    for character in atom_name.strip():
        if character.isalpha():
            return character.upper()
    raise ValueError(f"could not parse element from atom name {atom_name!r}")


def _canonical_atom_name(atom_name: str, residue_name: str) -> str:
    # NME's methyl carbon is named C in the PDB; it is chemically ACE's CH3, not a carbonyl.
    if residue_name == "NME" and atom_name == "C":
        return "CH3"
    name = atom_name
    if name and name[0] == "H" and name[-1] in ("1", "2", "3"):
        if not (residue_name in ("HIS", "HIE", "PHE", "TRP", "TYR") and name[:2] in ("HE", "HD", "HZ", "HH")):
            name = name[:-1]
    if name[:2] in ("OE", "OD"):
        name = name[:-1]
    return name


@dataclasses.dataclass(frozen=True)
class Encodings:
    """Per-atom structural encodings of a PDB.

    Attributes:
        atom_type: atom-name ids (``ATOM_TYPES``).
        atom_suffix: the digit removed by name canonicalization, 0 if none.
        aa_type: residue-name ids (``RESIDUE_TYPES``).
        aa_pos: 1-based residue index in order of appearance.
        seq_len: number of residues.
    """

    atom_type: tuple[int, ...]
    atom_suffix: tuple[int, ...]
    aa_type: tuple[int, ...]
    aa_pos: tuple[int, ...]
    seq_len: int


def read_pdb(pdb_path) -> tuple[tuple[str, ...], np.ndarray, Encodings]:
    """Atom names, masses (amu, float64) and structural encodings of the ``ATOM`` records of a PDB, in file order,
    parsed from the strict PDB columns."""
    names, masses, atom_type, atom_suffix, aa_type, aa_pos = [], [], [], [], [], []
    residue_index, last = 0, None
    with open(pdb_path) as handle:
        for line in handle:
            if not line.startswith("ATOM"):
                continue
            atom_name, residue_name, residue_seq = line[12:16].strip(), line[17:20].strip(), line[22:26].strip()
            element = _atom_element(line[12:16])
            if element not in _ELEMENT_MASSES:
                raise ValueError(f"unknown element {element!r} in {pdb_path}")
            if residue_seq != last:
                residue_index, last = residue_index + 1, residue_seq
            name = _canonical_atom_name(atom_name, residue_name)
            if name not in ATOM_TYPES or residue_name not in RESIDUE_TYPES:
                raise ValueError(f"unsupported atom {atom_name!r} in residue {residue_name!r}")
            names.append(atom_name)
            masses.append(_ELEMENT_MASSES[element])
            atom_type.append(ATOM_TYPES[name])
            atom_suffix.append(int(atom_name[-1]) if atom_name[-1:] in ("1", "2", "3") and name == atom_name[:-1]
                               else 0)
            aa_type.append(RESIDUE_TYPES[residue_name])
            aa_pos.append(residue_index)
    if not names:
        raise ValueError(f"no ATOM records in {pdb_path}")
    encodings = Encodings(tuple(atom_type), tuple(atom_suffix), tuple(aa_type), tuple(aa_pos), residue_index)
    return tuple(names), np.asarray(masses, dtype=np.float64), encodings


def source_frame_anchors(pdb_path) -> tuple[int, int, int]:
    """Atom indices of N, CA and C of the first residue that has all three, in PDB order.

    Raises:
        ValueError: alternate locations, HETATM records, an atom order that disagrees with OpenMM's topology, or no
            such residue.
    """
    from openmm.app import PDBFile

    text = Path(pdb_path).read_bytes().decode()
    records = [line for line in text.splitlines() if line.startswith(("ATOM  ", "HETATM"))]
    if any(line.startswith("HETATM") or line[16:17].strip() for line in records):
        raise ValueError("the source frame requires ATOM records without alternate locations")
    pdb = PDBFile(io.StringIO(text))
    atoms = list(pdb.topology.atoms())
    if len(atoms) != len(records) or any(
            (a.id, a.residue.name, a.residue.id, a.residue.chain.id.strip(), a.element.symbol)
            != (r[6:11].strip(), r[17:20].strip(), r[22:26].strip(), r[21:22].strip(),
                r[76:78].strip() or _atom_element(r[12:16]))
            for a, r in zip(atoms, records)):
        raise ValueError("PDB atom order disagrees with the OpenMM topology")
    for residue in pdb.topology.residues():
        names = [atom.name for atom in residue.atoms()]
        if {"N", "CA", "C"}.issubset(names):
            selected = []
            for name in ("N", "CA", "C"):
                matches = [a.index for a in residue.atoms() if a.name == name]
                if len(matches) != 1 or records[matches[0]][12:16].strip() != name:
                    raise ValueError(f"the source-frame residue needs exactly one {name}")
                selected.append(matches[0])
            return tuple(selected)
    raise ValueError("no residue contains N, CA and C")


@dataclasses.dataclass(frozen=True, eq=False)
class PeptideSystem:
    """Physical peptide: masses, thermodynamics, Amber/OBC energy and score.

    Attributes:
        pdb_path: packaged PDB path.
        n_atoms: number of atoms.
        atom_names: PDB atom names in file order.
        masses: (N,) float64 amu.
        temperature_K: bath temperature.
        kbt_kj_per_mol: kBT in kJ/mol.
        gamma_ps_inv: friction, 1/ps.
        encodings: structural encodings.
        anchors: N, CA, C atom indices of the source frame.
    """

    pdb_path: Path
    n_atoms: int
    atom_names: tuple[str, ...]
    masses: np.ndarray
    temperature_K: float
    kbt_kj_per_mol: float
    gamma_ps_inv: float
    encodings: Encodings
    anchors: tuple[int, int, int]
    _backend: object = dataclasses.field(repr=False)

    def score(self, x):
        """grad log pi(x) in 1/nm for positions (B, N, 3) nm."""
        return self._backend.score(x)

    def energy(self, x):
        """U(x)/kBT for positions (B, N, 3) nm; returns (B,) in the backend's float dtype."""
        return self._backend.energy(x)

    def topology(self):
        """The mdtraj topology of the packaged PDB."""
        import mdtraj

        return mdtraj.load(str(self.pdb_path)).topology


def load_system(system_cfg, *, float_dtype=None) -> PeptideSystem:
    """Build the peptide from a ``system`` config section.

    Args:
        system_cfg: mapping with ``pdb``, ``temperature_K``, ``gamma_ps_inv``, ``forcefield`` and ``cutoff_nm``.
        float_dtype: dtype of the Amber parameters (default: JAX's current default float). Pass ``jnp.float64``
            with x64 enabled for float64 energies.

    Raises:
        FileNotFoundError: no packaged PDB of that name.
        ValueError: non-positive temperature or friction, or an Amber topology that disagrees with the PDB.
    """
    from .amber import build_amber_backend

    path = packaged_pdb(str(system_cfg["pdb"]))
    temperature = float(system_cfg["temperature_K"])
    gamma = float(system_cfg["gamma_ps_inv"])
    if not (math.isfinite(temperature) and temperature > 0 and math.isfinite(gamma) and gamma > 0):
        raise ValueError("temperature_K and gamma_ps_inv must be finite and positive")
    cutoff = system_cfg["cutoff_nm"]
    cutoff = None if cutoff is None else float(cutoff)
    forcefield = tuple(str(name) for name in system_cfg["forcefield"])
    atom_names, masses, encodings = read_pdb(path)
    backend = build_amber_backend(str(path), temperature_K=temperature, forcefield=forcefield, cutoff_nm=cutoff,
                                  float_dtype=float_dtype)
    if backend.params.n_atoms != masses.size:
        raise ValueError("Amber topology and PDB disagree on the atom count")
    return PeptideSystem(
        pdb_path=path, n_atoms=int(masses.size), atom_names=atom_names, masses=masses, temperature_K=temperature,
        kbt_kj_per_mol=BOLTZMANN_CONSTANT_KJ_PER_MOL_K * temperature, gamma_ps_inv=gamma, encodings=encodings,
        anchors=source_frame_anchors(path), _backend=backend)
