"""Entry point ``python -m kcmaps.evaluate simulation=<simulation dir>``: equilibrium W2 metrics and, optionally,
kinetics of adaptive-chains output against a reference feature file. Evaluation never samples."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import hydra
import numpy as np
from omegaconf import DictConfig

from kcmaps.core.config import ConfigError, validate
from kcmaps.evaluation import features as F
from kcmaps.evaluation import frames as FR
from kcmaps.evaluation import kinetics as K
from kcmaps.evaluation import metrics as M

def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=1) + "\n")
    os.replace(tmp, path)


def system_config(simulation: Path) -> DictConfig:
    """System section of a simulation directory's ``run_spec.json``, strictly validated.

    Raises:
        ValueError: ``run_spec.json`` has no system section (not an adaptive-chains output of this package).
        ConfigError: the system section does not pass validation.
    """
    spec = json.loads((simulation / "run_spec.json").read_text())
    if not isinstance(spec.get("system"), dict):
        raise ValueError(f"{simulation} is not an adaptive-chains simulation directory (no system section in "
                         "run_spec.json)")
    return validate({"system": spec["system"]}, "system").system


def score_equilibrium(
    index: FR.FrameIndex,
    reference: dict[str, np.ndarray],
    *,
    topology,
    energy_fn,
    basis: F.TicaBasis,
    budget_s: float,
    burn_in_fraction: float,
    n_frames: int,
    n_draws: int,
    draw_seed: int,
    min_frames: int,
    energy_batch: int,
    tica_batch: int,
) -> dict:
    """W2 metrics of the frames of ``index`` usable within ``budget_s`` against ``reference``.

    Frames are pooled over chains after the per-chain burn-in and drawn with probability proportional to
    the simulated time each one closes (:func:`kcmaps.evaluation.frames.draw_rows`); each draw scores up to
    ``n_frames`` generated frames against ``n_frames`` uniform reference frames.

    Returns:
        dict with the pool sizes, ``metrics`` (per metric mean, sd, values; None when fewer than
        ``min_frames`` frames are usable) and the mean generated energy.
    """
    pool = FR.select_frames(index, budget_s=budget_s, burn_in_fraction=burn_in_fraction)
    w = FR.frame_intervals(index)[pool]
    pool, w = pool[w > 0], w[w > 0]
    out = dict(budget_s=budget_s, burn_in_fraction=burn_in_fraction, pooled_frames=int(len(pool)),
               pooled_simulated_ns=FR.pooled_simulated_ns(index, budget_s=budget_s),
               n_gen=int(min(n_frames, len(pool))),
               n_reference=int(n_frames), n_draws=n_draws, draw_seed=draw_seed)
    if len(pool) < min_frames:
        return dict(out, metrics=None, note=f"fewer than {min_frames} usable frames")
    draws = FR.draw_rows(pool, w, n_frames=n_frames, n_reference=len(reference["energy"]), n_draws=n_draws,
                         seed=draw_seed)
    urows = np.unique(np.concatenate([g for g, _ in draws]))
    feats = F.featurize(FR.load_positions(index, urows), topology=topology, energy_fn=energy_fn, basis=basis,
                        energy_batch=energy_batch, tica_batch=tica_batch)
    results = []
    for g, r in draws:
        gen = {k: v[np.searchsorted(urows, g)] for k, v in feats.items()}
        ref = {k: np.asarray(reference[k][r]) for k in ("energy", "rama", "tica")}
        results.append(M.w2_metrics(ref, gen))
    return dict(out, metrics=M.summarize_draws(results), mean_energy=float(np.mean(feats["energy"])))


def reference_chains(reference: dict[str, np.ndarray]) -> tuple[list[dict], float]:
    """Per reference trajectory, its phi, psi and time (ps, 0 at its first frame); and the frame interval (ps).

    Raises:
        ValueError: the reference has more than one phi/psi pair, or no trajectory has two frames.
    """
    rama, counts = np.asarray(reference["rama"]), np.asarray(reference["frames_per_trajectory"])
    frame_ps = float(reference["frame_ps"])
    if rama.shape[1] != 2:
        raise ValueError("kinetics needs exactly one (phi, psi) pair")
    if counts.max() < 2:
        raise ValueError("reference trajectories need at least two frames")
    chains = []
    for start, n in zip(np.cumsum(counts) - counts, counts):
        chains.append(dict(phi=rama[start:start + n, 0].astype(np.float64),
                           psi=rama[start:start + n, 1].astype(np.float64), t=np.arange(n) * frame_ps))
    return chains, frame_ps


def _lag_frames(max_lag_ps: float, frame_ps: float, arm: str) -> int:
    lag = int(round(max_lag_ps / frame_ps))
    if lag < 1:
        raise ValueError(f"kinetics.max_lag_ps {max_lag_ps} is below one {arm} frame ({frame_ps} ps)")
    return lag


def _relaxation_ps(record: dict, frame_ps: float) -> dict:
    scale = lambda v: None if v is None else v * frame_ps
    return {state: {k: [scale(x) for x in v] if isinstance(v, list) else scale(v) for k, v in times.items()}
            for state, times in record["relaxation_frames"].items()}


def score_kinetics(index: FR.FrameIndex, reference: dict[str, np.ndarray], *, topology, budget_s: float,
                   burn_in_fraction: float, bootstrap: int, seed: int, max_lag_ps: float) -> tuple[dict, dict]:
    """Kinetics of the reference trajectories and of the generated chains on the reference's cores.

    Generated chains keep their usable frames within ``budget_s`` that advance simulated time, after dropping the
    first ``floor(burn_in_fraction * n)`` frames. One generator (``seed``) draws the bootstrap of the reference,
    then of the generated chains. Each arm has one frame interval (reference: its ``frame_ps``; generated: the
    median interval); ACF lag k is k times that interval, and ``max_lag_ps`` is converted to frames with it. Exit
    rates are per ns of labeled simulated time.

    Returns:
        (JSON record, arrays for ``kinetics_acf.npz``).

    Raises:
        ValueError: no generated chain has two usable frames, or ``max_lag_ps`` is below one frame of an arm.
    """
    ref_chains, ref_frame_ps = reference_chains(reference)
    ref_lag = _lag_frames(max_lag_ps, ref_frame_ps, "reference")
    states = K.define_states(np.concatenate([c["phi"] for c in ref_chains]),
                             np.concatenate([c["psi"] for c in ref_chains]))
    gen_chains = []
    for rows in FR.chain_rows(index, budget_s=budget_s, burn_in_fraction=burn_in_fraction):
        if len(rows) == 0:
            continue
        rama = F.phi_psi(FR.load_positions(index, rows), topology)
        if rama.shape[1] != 2:
            raise ValueError("kinetics needs exactly one (phi, psi) pair")
        gen_chains.append(dict(phi=rama[:, 0].astype(np.float64), psi=rama[:, 1].astype(np.float64),
                               t=index.time_ps[rows]))
    dt = np.concatenate([np.diff(c["t"]) for c in gen_chains]) if gen_chains else np.zeros(0)
    if dt.size == 0:
        raise ValueError("kinetics: no generated chain has two usable frames within the budget after burn-in")
    gen_frame_ps = float(np.median(dt))
    gen_lag = _lag_frames(max_lag_ps, gen_frame_ps, "generated")
    rng = np.random.default_rng(seed)
    ref_rec, ref_acf = K.chain_kinetics(ref_chains, states, rng, n_boot=bootstrap, max_lag=ref_lag)
    gen_rec, gen_acf = K.chain_kinetics(gen_chains, states, rng, n_boot=bootstrap, max_lag=gen_lag)
    ref_rec = dict(ref_rec, frame_ps=ref_frame_ps, max_lag_frames=ref_lag,
                   relaxation_ps=_relaxation_ps(ref_rec, ref_frame_ps))
    gen_rec = dict(gen_rec, frame_ps=gen_frame_ps, max_lag_frames=gen_lag,
                   relaxation_ps=_relaxation_ps(gen_rec, gen_frame_ps))
    record = dict(states=list(states.names), segmentation=states.summary, max_lag_ps=float(max_lag_ps),
                  bootstrap=bootstrap, seed=seed, reference=ref_rec, generated=gen_rec)
    arrays = dict(states=np.array(states.names), reference_lag_ps=np.arange(ref_lag + 1) * ref_frame_ps,
                  generated_lag_ps=np.arange(gen_lag + 1) * gen_frame_ps, reference=ref_acf, generated=gen_acf)
    return record, arrays


def _print_summary(eq: dict, kin: dict | None, out_dir: Path) -> None:
    """Print each equilibrium metric as mean +- sd, then the output directory."""
    if eq["metrics"] is None:
        print(f"{eq['note']}; no equilibrium metrics")
    for name, m in (eq["metrics"] or {}).items():
        print(f"{name}: {m['mean']:.4f} +- {m['sd']:.4f}")
    print(f"results in {out_dir}")
    if kin is not None:
        print(f"kinetics: {out_dir / 'kinetics.json'}")


def evaluate(cfg: DictConfig | dict) -> dict:
    """Score a simulation directory and write ``equilibrium.json`` (and ``kinetics.json`` +
    ``kinetics_acf.npz`` when ``kinetics.enabled``) to ``paths.output_dir``.

    Returns:
        ``{"equilibrium": ..., "kinetics": ... or None}``.

    Raises:
        ConfigError: invalid config, or kinetics enabled for a system with more than one (phi, psi) pair (checked
            before any scoring).
    """
    cfg = validate(cfg, "evaluate")
    import jax
    import jax.numpy as jnp

    from kcmaps.systems import load_system

    with jax.enable_x64(True):
        simulation = Path(cfg.simulation)
        sys_cfg = system_config(simulation)
        name = str(sys_cfg.name)
        system = load_system(sys_cfg, float_dtype=jnp.float64)
        topology = system.topology()
        basis = F.TicaBasis.packaged(name)
        basis.check_topology(topology)
        ref_path = Path(cfg.reference_dir) / f"{name}.npz"
        reference = F.load_reference(ref_path)
        torsions = reference["rama"].shape[1]
        if cfg.kinetics.enabled and torsions != 2:
            raise ConfigError(f"kinetics needs exactly one (phi, psi) pair; the {name} reference has {torsions} "
                              "torsions")
        index = FR.index_frames(simulation)
        budget_s = float(cfg.budget_minutes) * 60.0
        eq = score_equilibrium(index, reference, topology=topology, energy_fn=system.energy, basis=basis,
                               budget_s=budget_s, burn_in_fraction=cfg.burn_in_fraction, n_frames=cfg.n_frames,
                               n_draws=cfg.n_draws, draw_seed=cfg.draw_seed, min_frames=cfg.min_frames,
                               energy_batch=cfg.energy_batch, tica_batch=cfg.tica_batch)
        eq = dict(system=name, simulation=str(simulation), reference_file=str(ref_path),
                  reference_frames=int(len(reference["energy"])), **eq)
        if cfg.floor:
            eq["floor"] = M.reference_floor(reference, n_frames=cfg.n_frames, n_draws=cfg.n_draws)
        out_dir = Path(cfg.paths.output_dir)
        _write_json(out_dir / "equilibrium.json", eq)
        kin = None
        if cfg.kinetics.enabled:
            kin, arrays = score_kinetics(index, reference, topology=topology, budget_s=budget_s,
                                         burn_in_fraction=cfg.burn_in_fraction, bootstrap=cfg.kinetics.bootstrap,
                                         seed=cfg.kinetics.seed, max_lag_ps=cfg.kinetics.max_lag_ps)
            kin = dict(system=name, simulation=str(simulation), reference_file=str(ref_path), **kin)
            _write_json(out_dir / "kinetics.json", kin)
            np.savez(out_dir / "kinetics_acf.npz", **arrays)
        _print_summary(eq, kin, out_dir)
        return dict(equilibrium=eq, kinetics=kin)


@hydra.main(version_base="1.3", config_path="../../configs", config_name="evaluate")
def main(cfg: DictConfig) -> None:
    try:
        evaluate(cfg)
    except ConfigError as error:
        raise SystemExit(f"invalid configuration: {error}") from None


if __name__ == "__main__":
    main()
