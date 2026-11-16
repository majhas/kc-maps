"""Amber parameters of a PDB under OpenMM force-field XMLs, extracted once into numpy buffers (``AmberParams``)."""

from __future__ import annotations

import dataclasses
import math
import re
from typing import Any

import numpy as np

# Coulomb constant for the explicit nonbonded sum (kJ·nm/(mol·e²)). OpenMM
# uses 138.935485 (8th-digit difference, well below the float32 budget).
ONE_4PI_EPS0 = 138.935456

# Boltzmann constant in OpenMM native units: kJ/(mol·K).
BOLTZMANN_CONSTANT_KJ_PER_MOL_K = 8.314462618e-3


# ------------------------------------------------------------------ #
# AmberParams dataclass
# ------------------------------------------------------------------ #


@dataclasses.dataclass(frozen=True, eq=False)
class AmberParams:
    """Immutable container of every numerical parameter the JAX energy
    needs. All arrays are numpy (host side); cast to ``jnp`` only inside
    the energy module so this object is safe to construct outside JIT.

    ``eq=False`` keeps the default identity-based ``__eq__``/``__hash__``:
    the auto-generated value-equality would call ``bool()`` on numpy
    array fields (ambiguous-truth-value error) and ``hash()`` on
    unhashable arrays, so any comparison or dict/set use would crash.
    """

    n_atoms: int
    kBT: float                       # kJ/mol at the target temperature

    # --- bonded ---
    bond_indices: np.ndarray         # (n_bonds, 2) int32
    bond_k: np.ndarray               # (n_bonds,)  float32  kJ/(mol·nm²)
    bond_r0: np.ndarray              # (n_bonds,)  float32  nm

    angle_indices: np.ndarray        # (n_angles, 3) int32
    angle_k: np.ndarray              # (n_angles,)  float32  kJ/(mol·rad²)
    angle_theta0: np.ndarray         # (n_angles,)  float32  rad

    torsion_indices: np.ndarray      # (n_tors, 4) int32
    torsion_k: np.ndarray            # (n_tors,)  float32  kJ/mol
    torsion_n: np.ndarray            # (n_tors,)  float32  periodicity (int-valued)
    torsion_phase: np.ndarray        # (n_tors,)  float32  rad

    # --- nonbonded (LJ + Coulomb; vacuum, or CutoffNonPeriodic) ---
    nb_pair_mask: np.ndarray         # (N, N) bool — upper-triangle excluding exclusions
    nb_sigma_ij: np.ndarray          # (N, N) float32  Lorentz-Berthelot
    nb_eps_ij: np.ndarray            # (N, N) float32  Lorentz-Berthelot
    nb_qq_ij: np.ndarray             # (N, N) float32  charge_i · charge_j (e²)
    exc_indices: np.ndarray          # (n_exc, 2) int32 — scaled 1-4 pairs (may be len 0)
    exc_chargeprod: np.ndarray       # (n_exc,) float32  e²
    exc_sigma: np.ndarray            # (n_exc,) float32  nm
    exc_epsilon: np.ndarray          # (n_exc,) float32  kJ/mol

    # --- GBSA-OBC (OBC1 or OBC2) ---
    gb_offset_radius: np.ndarray     # (N,) float32  nm  ('or' in the OBC energy string)
    gb_scaled_radius: np.ndarray     # (N,) float32  nm  ('sr' = or * scale_factor)
    gb_charge: np.ndarray            # (N,) float32  e
    gb_qq: np.ndarray                # (N, N) float32  charge_i · charge_j
    gb_prefactor: float              # -138.935485 · (1/εu - 1/εs)
    gb_sa_factor: float              # kJ/(mol·nm²), ACE surface term
    gb_probe_radius: float           # 0.14 nm (water)
    gb_dielectric_offset: float      # 0.009 nm
    gb_alpha: float                  # 0.8 (OBC1) or 1.0 (OBC2)
    gb_beta: float                   # 0.0 (OBC1) or 0.8 (OBC2)
    gb_gamma: float                  # 2.909125 (OBC1) or 4.85 (OBC2)

    # --- nonbonded cutoff (OpenMM CutoffNonPeriodic); None = vacuum ---
    # Static Python scalars: `_to_jax_params` only converts array fields, so
    # these stay trace-time constants and `None` compiles the vacuum graph.
    nb_cutoff: float | None = None   # nm; None => full O(N^2) vacuum sum
    nb_rf_dielectric: float = 1.0    # reaction-field solvent dielectric


# ------------------------------------------------------------------ #
# Per-force extractors
# ------------------------------------------------------------------ #


def _extract_bonded(force: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from openmm import unit
    n = force.getNumBonds()
    idx = np.zeros((n, 2), dtype=np.int32)
    k = np.zeros(n, dtype=np.float32)
    r0 = np.zeros(n, dtype=np.float32)
    for i in range(n):
        p1, p2, length, force_k = force.getBondParameters(i)
        idx[i] = (p1, p2)
        r0[i] = length.value_in_unit(unit.nanometer)
        k[i] = force_k.value_in_unit(unit.kilojoule_per_mole / unit.nanometer**2)
    return idx, k, r0


def _extract_angles(force: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from openmm import unit
    n = force.getNumAngles()
    idx = np.zeros((n, 3), dtype=np.int32)
    k = np.zeros(n, dtype=np.float32)
    theta0 = np.zeros(n, dtype=np.float32)
    for i in range(n):
        p1, p2, p3, angle, force_k = force.getAngleParameters(i)
        idx[i] = (p1, p2, p3)
        theta0[i] = angle.value_in_unit(unit.radian)
        k[i] = force_k.value_in_unit(unit.kilojoule_per_mole / unit.radian**2)
    return idx, k, theta0


def _extract_torsions(force: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from openmm import unit
    n = force.getNumTorsions()
    idx = np.zeros((n, 4), dtype=np.int32)
    k = np.zeros(n, dtype=np.float32)
    per = np.zeros(n, dtype=np.float32)
    phase = np.zeros(n, dtype=np.float32)
    for i in range(n):
        p1, p2, p3, p4, periodicity, ph, force_k = force.getTorsionParameters(i)
        idx[i] = (p1, p2, p3, p4)
        per[i] = periodicity
        phase[i] = ph.value_in_unit(unit.radian)
        k[i] = force_k.value_in_unit(unit.kilojoule_per_mole)
    return idx, k, per, phase


def _extract_nonbonded(force: Any) -> dict:
    """Extract LJ + Coulomb + 1-4 exception params from an OpenMM
    ``NonbondedForce``.

    Non-periodic only: ``NoCutoff`` (vacuum) or ``CutoffNonPeriodic``. For the
    latter we also read the cutoff and the reaction-field dielectric, which
    together define the in-range Coulomb form the JAX port reproduces.
    """
    import openmm
    from openmm import unit

    method = force.getNonbondedMethod()
    if method not in (openmm.NonbondedForce.NoCutoff,
                      openmm.NonbondedForce.CutoffNonPeriodic):
        raise ValueError(
            f"NonbondedForce method {method} is periodic (CutoffPeriodic / "
            f"Ewald / PME). This backend has no PBC: use NoCutoff or "
            f"CutoffNonPeriodic.")

    cutoff = None
    rf_dielectric = 1.0
    if method == openmm.NonbondedForce.CutoffNonPeriodic:
        cutoff = float(force.getCutoffDistance().value_in_unit(unit.nanometer))
        rf_dielectric = float(force.getReactionFieldDielectric())
        if force.getUseSwitchingFunction():
            raise ValueError(
                "NonbondedForce uses a switching function; the JAX port "
                "implements OpenMM's sharp LJ truncation only.")

    n = force.getNumParticles()
    charge = np.zeros(n, dtype=np.float32)
    sigma = np.zeros(n, dtype=np.float32)
    epsilon = np.zeros(n, dtype=np.float32)
    for i in range(n):
        q, sig, eps = force.getParticleParameters(i)
        charge[i] = q.value_in_unit(unit.elementary_charge)
        sigma[i] = sig.value_in_unit(unit.nanometer)
        epsilon[i] = eps.value_in_unit(unit.kilojoule_per_mole)

    # Exceptions: collect both exclusions (zero chargeprod & epsilon, full
    # 1-2 / 1-3 mask) and scaled 1-4 pairs (non-zero — kept as a sparse
    # add-on list to be evaluated separately from the all-pairs sum).
    n_exc = force.getNumExceptions()
    exclusion_set: set[tuple[int, int]] = set()
    exc_idx_list: list[tuple[int, int]] = []
    exc_qq_list: list[float] = []
    exc_sigma_list: list[float] = []
    exc_eps_list: list[float] = []
    for i in range(n_exc):
        p1, p2, qprod, sig, eps = force.getExceptionParameters(i)
        qprod_v = qprod.value_in_unit(unit.elementary_charge**2)
        sig_v = sig.value_in_unit(unit.nanometer)
        eps_v = eps.value_in_unit(unit.kilojoule_per_mole)
        exclusion_set.add((min(p1, p2), max(p1, p2)))
        if abs(qprod_v) > 0.0 or abs(eps_v) > 0.0:
            exc_idx_list.append((p1, p2))
            exc_qq_list.append(qprod_v)
            exc_sigma_list.append(sig_v)
            exc_eps_list.append(eps_v)

    # Upper-triangle pair mask, excluding every (1-2, 1-3, 1-4) exclusion.
    # 1-4 pairs are still excluded from the all-pairs sum because the
    # scaled exception term replaces them, *not* adds to them — OpenMM's
    # semantics for NonbondedForce exceptions are "if listed, only this
    # term applies for that pair".
    excluded = np.zeros((n, n), dtype=bool)
    if exclusion_set:
        ex = np.asarray(sorted(exclusion_set))          # (n_excl, 2), i<j
        excluded[ex[:, 0], ex[:, 1]] = True
    pair_mask = np.triu(np.ones((n, n), dtype=bool), k=1) & ~excluded

    # Lorentz-Berthelot combination, precomputed (N, N).
    sigma_ij = 0.5 * (sigma[:, None] + sigma[None, :])
    eps_ij = np.sqrt(epsilon[:, None] * epsilon[None, :])
    qq_ij = charge[:, None] * charge[None, :]

    return dict(
        nb_pair_mask=pair_mask,
        nb_sigma_ij=sigma_ij.astype(np.float32),
        nb_eps_ij=eps_ij.astype(np.float32),
        nb_qq_ij=qq_ij.astype(np.float32),
        exc_indices=np.asarray(exc_idx_list, dtype=np.int32).reshape(-1, 2),
        exc_chargeprod=np.asarray(exc_qq_list, dtype=np.float32),
        exc_sigma=np.asarray(exc_sigma_list, dtype=np.float32),
        exc_epsilon=np.asarray(exc_eps_list, dtype=np.float32),
        nb_cutoff=cutoff,
        nb_rf_dielectric=rf_dielectric,
    )


def _parse_float(expr: str, name: str, default: float | None = None) -> float:
    m = re.search(rf"{name}\s*=\s*([0-9.eE+-]+)", expr)
    if m:
        return float(m.group(1))
    if default is not None:
        return default
    raise ValueError(f"could not parse {name!r} from CustomGB expression: {expr}")


def _extract_custom_gb(force: Any) -> dict:
    """Extract OBC1 params from a ``CustomGBForce`` produced by
    ``implicit/obc1.xml``. Per-particle params are already ``[charge,
    offset_radius (or), scaled_radius (sr)]`` — no further reduction
    needed.
    """
    import openmm
    from openmm import unit as _unit

    gb_method = force.getNonbondedMethod()
    if gb_method not in (openmm.CustomGBForce.NoCutoff,
                         openmm.CustomGBForce.CutoffNonPeriodic):
        raise ValueError(
            f"CustomGBForce method {gb_method} is periodic. This backend has "
            f"no PBC: use NoCutoff or CutoffNonPeriodic.")
    gb_cutoff = (float(force.getCutoffDistance().value_in_unit(_unit.nanometer))
                 if gb_method == openmm.CustomGBForce.CutoffNonPeriodic else None)

    n = force.getNumParticles()
    names = [force.getPerParticleParameterName(i)
             for i in range(force.getNumPerParticleParameters())]
    name_to_idx = {n: i for i, n in enumerate(names)}
    if not {"charge", "or", "sr"}.issubset(name_to_idx):
        raise ValueError(
            f"CustomGB per-particle params {names} missing one of "
            "{'charge', 'or', 'sr'} — not an OBC1 force?")

    charge = np.zeros(n, dtype=np.float32)
    offset_radius = np.zeros(n, dtype=np.float32)
    scaled_radius = np.zeros(n, dtype=np.float32)
    for i in range(n):
        p = force.getParticleParameters(i)
        charge[i] = float(p[name_to_idx["charge"]])
        offset_radius[i] = float(p[name_to_idx["or"]])
        scaled_radius[i] = float(p[name_to_idx["sr"]])

    # Electric constant and dielectrics are parsed from the expression.
    self_expr = force.getEnergyTermParameters(0)[0]   # E_self
    sa_expr = force.getEnergyTermParameters(1)[0]     # E_SA
    born_expr = force.getComputedValueParameters(1)[1]  # B = 1/(1/or - tanh(...)/radius)

    # E_self looks like "-0.5*138.935485*(1/soluteDielectric - 1/solventDielectric)*charge^2/B; ..."
    m = re.search(r"-?0\.5\*([0-9.]+)", self_expr)
    if m is None:
        raise ValueError(f"could not parse electric constant from: {self_expr}")
    electric_constant = float(m.group(1))
    solvent_d = _parse_float(self_expr, "solventDielectric")
    solute_d = _parse_float(self_expr, "soluteDielectric")
    offset = _parse_float(self_expr, "offset")
    # The prefactor closed form assumes Debye-Hückel screening κ=0 (the
    # self-energy is `(1/εu − 1/εs)`; for κ>0 it becomes `(1/εu −
    # exp(−κB)/εs)`, which this parser does NOT model). Fail loud rather
    # than silently dropping the screening term.
    kappa = _parse_float(self_expr, "kappa", default=0.0)
    if abs(kappa) > 0.0:
        raise ValueError(
            f"CustomGB has non-zero kappa={kappa} (Debye screening); the "
            "prefactor closed form only supports kappa=0.")

    prefactor = -electric_constant * (1.0 / solute_d - 1.0 / solvent_d)

    # Born radii formula: B = 1/(1/or - tanh(α·psi − β·psi^2 + γ·psi^3)/radius).
    # β enters with a minus sign in the evaluator (OBC1 β=0, OBC2 β=0.8).
    tanh_inner = re.search(r"tanh\(([^)]+)\)", born_expr)
    if tanh_inner is None:
        raise ValueError(f"could not find tanh(...) in Born expression: {born_expr}")
    inner = tanh_inner.group(1)
    # α: coefficient·psi (OBC1: 0.8) or a bare leading `psi` (OBC2: 1.0).
    alpha_m = re.search(r"(?<![\d.])([0-9.]+)\*psi(?!\^)", inner)
    if alpha_m is not None:
        alpha = float(alpha_m.group(1))
    elif re.search(r"(?<![\w.^])psi(?!\^)", inner) is not None:
        alpha = 1.0
    else:
        raise ValueError(f"could not parse OBC alpha from tanh argument: {inner}")
    # β: absent for OBC1 (no psi^2 term) → 0.0; present for OBC2.
    beta_m = re.search(r"([0-9.]+)\*psi\^2", inner)
    beta = float(beta_m.group(1)) if beta_m else 0.0
    gamma_m = re.search(r"([0-9.]+)\*psi\^3", inner)
    if gamma_m is None:
        raise ValueError(f"could not parse OBC gamma from tanh argument: {inner}")
    gamma = float(gamma_m.group(1))

    # ACE surface-area term: "<sa_factor>*(radius+<probe>)^2*(radius/B)^6"
    sa_m = re.search(r"^([0-9.]+)\*", sa_expr)
    sa_factor = float(sa_m.group(1)) if sa_m else 28.3919551
    probe_m = re.search(r"radius\s*\+\s*([0-9.]+)", sa_expr)
    probe_radius = float(probe_m.group(1)) if probe_m else 0.14

    qq = (charge[:, None] * charge[None, :]).astype(np.float32)

    return dict(
        gb_offset_radius=offset_radius,
        gb_scaled_radius=scaled_radius,
        gb_charge=charge,
        gb_qq=qq,
        gb_prefactor=float(prefactor),
        gb_sa_factor=float(sa_factor),
        gb_probe_radius=float(probe_radius),
        gb_dielectric_offset=float(offset),
        gb_alpha=float(alpha),
        gb_beta=float(beta),
        gb_gamma=float(gamma),
    ), gb_cutoff


# OBC2 constants that OpenMM compiles into GBSAOBCForce instead of exposing.
_OBC2_ALPHA, _OBC2_BETA, _OBC2_GAMMA = 1.0, 0.8, 4.85
_OBC_DIELECTRIC_OFFSET = 0.009   # nm
_OBC_PROBE_RADIUS = 0.14         # nm
# OpenMM stores the ACE prefactor per unit area; the CustomGB expression folds in
# the 4*pi of the sphere. Same physics, different bookkeeping, so convert.
_ACE_SOLID_ANGLE = 4.0 * math.pi


def _extract_gbsa_obc(force: Any) -> tuple:
    """Extract OBC2 params from a ``GBSAOBCForce`` (``amber99_obc.xml``).

    Same ``(params, cutoff)`` as :func:`_extract_custom_gb`, so the JAX GB energy is
    shared. Emits OpenMM's ``or`` (the dielectric-offset-reduced radius) and ``sr``
    (its scaled form); the JAX side reconstructs the bare radius as ``or + offset``.
    The two differ only in how OpenMM hands the parameters over: a CustomGBForce
    carries a parseable energy string, while GBSAOBCForce is compiled C++ whose
    constants must be supplied (see above).
    """
    import openmm
    from openmm import unit as _unit
    gb_method = force.getNonbondedMethod()
    if gb_method not in (openmm.GBSAOBCForce.NoCutoff,
                         openmm.GBSAOBCForce.CutoffNonPeriodic):
        raise ValueError(
            f"GBSAOBCForce method {gb_method} is periodic. This backend has no "
            f"PBC: use NoCutoff or CutoffNonPeriodic.")
    gb_cutoff = (float(force.getCutoffDistance().value_in_unit(_unit.nanometer))
                 if gb_method == openmm.GBSAOBCForce.CutoffNonPeriodic else None)

    n = force.getNumParticles()
    charge = np.zeros(n, dtype=np.float32)
    offset_radius = np.zeros(n, dtype=np.float32)
    scaled_radius = np.zeros(n, dtype=np.float32)
    for i in range(n):
        q, radius, scale = force.getParticleParameters(i)
        charge[i] = q.value_in_unit(openmm.unit.elementary_charge)
        offset_radius[i] = (radius.value_in_unit(openmm.unit.nanometer)
                            - _OBC_DIELECTRIC_OFFSET)
        scaled_radius[i] = float(scale) * offset_radius[i]

    solvent_d = force.getSolventDielectric()
    solute_d = force.getSoluteDielectric()
    prefactor = -ONE_4PI_EPS0 * (1.0 / solute_d - 1.0 / solvent_d)
    sa_factor = force.getSurfaceAreaEnergy().value_in_unit(
        openmm.unit.kilojoule_per_mole / openmm.unit.nanometer ** 2) * _ACE_SOLID_ANGLE

    qq = (charge[:, None] * charge[None, :]).astype(np.float32)

    return dict(
        gb_offset_radius=offset_radius,
        gb_scaled_radius=scaled_radius,
        gb_charge=charge,
        gb_qq=qq,
        gb_prefactor=float(prefactor),
        gb_sa_factor=float(sa_factor),
        gb_probe_radius=float(_OBC_PROBE_RADIUS),
        gb_dielectric_offset=float(_OBC_DIELECTRIC_OFFSET),
        gb_alpha=float(_OBC2_ALPHA),
        gb_beta=float(_OBC2_BETA),
        gb_gamma=float(_OBC2_GAMMA),
    ), gb_cutoff


# ------------------------------------------------------------------ #
# Top-level builder
# ------------------------------------------------------------------ #


def build_amber_params(pdb_path: str, temperature_K: float, forcefield: tuple,
                       cutoff_nm: float | None = None) -> AmberParams:
    """Build the AmberParams for a PDB under an Amber force field + OBC solvent.

    Args:
        pdb_path: absolute path to the PDB file (single chain, no PBC).
        temperature_K: temperature for the kT divisor in :class:`AmberParams`.
        forcefield: OpenMM force-field XMLs, e.g. ``amber14-all.xml`` +
            ``implicit/obc1.xml`` (CustomGBForce, OBC1) or ``amber99sbildn.xml`` +
            ``amber99_obc.xml`` (GBSAOBCForce, OBC2); the GB extractor is chosen
            by the force type OpenMM produces, not by the XML name.
        cutoff_nm: if given, build the system with ``CutoffNonPeriodic`` at this
            cutoff (nm) instead of vacuum ``NoCutoff``. Use it to match a
            reference trajectory generated under a cutoff (AL6 uses
            2.0 nm). ``None`` keeps the exact, full O(N^2) vacuum potential,
            which is the more accurate object and the default here.

    Returns:
        Frozen :class:`AmberParams` ready for the JAX energy module.

    Raises:
        RuntimeError: one of the five forces (HarmonicBond, HarmonicAngle,
            PeriodicTorsion, Nonbonded, CustomGB or GBSAOBC) is missing or
            duplicated, or the System contains any other energy-carrying force.
        ValueError: ``cutoff_nm`` is not positive, or a force uses a periodic method.
    """
    import openmm
    from openmm import unit
    from openmm.app import CutoffNonPeriodic, ForceField, NoCutoff, PDBFile

    if cutoff_nm is not None and not cutoff_nm > 0.0:
        raise ValueError(f"cutoff_nm must be positive or None; got {cutoff_nm}")
    create_kwargs = dict(
        nonbondedMethod=NoCutoff if cutoff_nm is None else CutoffNonPeriodic,
        constraints=None,
        rigidWater=False,
        removeCMMotion=False,  # no integrator here; CMMotionRemover is a runtime concept
    )
    if cutoff_nm is not None:
        create_kwargs["nonbondedCutoff"] = cutoff_nm * unit.nanometer
    system = ForceField(*forcefield).createSystem(PDBFile(pdb_path).topology, **create_kwargs)

    bonded = angle = torsion = nb = gb = None
    seen = set()
    for f in system.getForces():
        if isinstance(f, openmm.CMMotionRemover):  # carries no energy
            continue
        kind = type(f).__name__
        if kind in seen:
            raise RuntimeError(f"System for {pdb_path} has more than one {kind}; the JAX port supports one of each.")
        seen.add(kind)
        if isinstance(f, openmm.HarmonicBondForce):
            bonded = _extract_bonded(f)
        elif isinstance(f, openmm.HarmonicAngleForce):
            angle = _extract_angles(f)
        elif isinstance(f, openmm.PeriodicTorsionForce):
            torsion = _extract_torsions(f)
        elif isinstance(f, openmm.NonbondedForce):
            nb = _extract_nonbonded(f)
        elif isinstance(f, openmm.CustomGBForce):
            gb, gb_cutoff = _extract_custom_gb(f)
        elif isinstance(f, openmm.GBSAOBCForce):
            gb, gb_cutoff = _extract_gbsa_obc(f)
        else:
            raise RuntimeError(
                f"System for {pdb_path} contains {kind}, which the JAX energy does not implement.")
    if "CustomGBForce" in seen and "GBSAOBCForce" in seen:
        raise RuntimeError(f"System for {pdb_path} has both CustomGBForce and GBSAOBCForce.")

    missing = [n for n, v in (("HarmonicBond", bonded), ("HarmonicAngle", angle),
                              ("PeriodicTorsion", torsion), ("Nonbonded", nb),
                              ("GB (CustomGB/OBC1 or GBSAOBC/OBC2)", gb)) if v is None]
    if missing:
        raise RuntimeError(
            f"Force(s) absent from System for {pdb_path}: {missing}. "
            "Did the force field load correctly?")

    # createSystem gives both forces the same cutoff; assert it rather than
    # trusting it, since a mismatch would silently desync the GB pair sum from
    # the Coulomb sum and neither term would look wrong on its own.
    if gb_cutoff != nb["nb_cutoff"]:
        raise RuntimeError(
            f"cutoff mismatch between NonbondedForce ({nb['nb_cutoff']}) and "
            f"CustomGBForce ({gb_cutoff}); the JAX port assumes one cutoff.")

    bond_idx, bond_k, bond_r0 = bonded
    ang_idx, ang_k, ang_theta0 = angle
    tor_idx, tor_k, tor_n, tor_phase = torsion

    return AmberParams(
        n_atoms=system.getNumParticles(),
        kBT=BOLTZMANN_CONSTANT_KJ_PER_MOL_K * float(temperature_K),
        bond_indices=bond_idx,
        bond_k=bond_k,
        bond_r0=bond_r0,
        angle_indices=ang_idx,
        angle_k=ang_k,
        angle_theta0=ang_theta0,
        torsion_indices=tor_idx,
        torsion_k=tor_k,
        torsion_n=tor_n,
        torsion_phase=tor_phase,
        **nb,
        **gb,
    )
