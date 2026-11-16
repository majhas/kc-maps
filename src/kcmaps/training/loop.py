"""Training run: setup, fresh start or resume, the step loop, validation, logging and checkpoint cadence."""

from __future__ import annotations

import json
import math
import signal
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from omegaconf import OmegaConf

from ..core import constants as C
from ..core.config import ConfigError, check_resume, load_run_config, run_config, save_run_config
from ..models import Scales, build_kcm, init_params
from ..models import checkpoint as ckpt
from ..systems import load_system
from . import batches, losses, optim
from .data import fit_x_std, load_dataset

# Consecutive log windows with only reverted steps or a non-finite loss before the run aborts.
STALL_LOGS = 10
# Seed of the fixed validation batches (frames, lags and per-row keys), the same in every run.
VAL_SEED = 0
# Metrics logged under diagnostics/ (the rest of a train step's metrics go under train/).
DIAGNOSTICS = ("kc_nll", "kc_frac_capped", "db_acceptance")
# Metrics logged under val/.
VAL_METRICS = ("kc_loss", "kc_loss_short", "kc_loss_mid", "kc_loss_long", "kc_nll", "db_loss", "pf_loss")
# Display order of the logged metrics.
_ORDER = ("loss", "kc_loss", "kc_loss_short", "kc_loss_mid", "kc_loss_long", "db_loss", "db_loss_short", "db_loss_mid",
          "db_loss_long", "pf_loss", "pf_loss_short", "pf_loss_mid", "pf_loss_long", "aux_weight",
          "grad_norm", "lr", "skipped_steps", "kc_nll", "kc_frac_capped", "db_acceptance")


class TrainingStalled(RuntimeError):
    """Every step of ``STALL_LOGS`` consecutive log windows was reverted or had a non-finite loss."""


def configure_jax() -> None:
    """Training numerics: float32, HIGHEST matmul precision and the partitionable Threefry key law."""
    jax.config.update("jax_enable_x64", False)
    jax.config.update("jax_default_matmul_precision", C.MATMUL_PRECISION)
    jax.config.update("jax_default_prng_impl", C.TRAINING_PRNG["implementation"])
    jax.config.update("jax_threefry_partitionable", C.TRAINING_PRNG["threefry_partitionable"])
    jax.config.update("jax_random_seed_offset", C.TRAINING_PRNG["random_seed_offset"])


def make_mesh() -> Mesh:
    """One-axis ``('data',)`` mesh over every device."""
    return Mesh(mesh_utils.create_device_mesh((jax.device_count(),)), ("data",))


def metric_means(metrics) -> dict:
    """Host means ``sum / count`` of ``(sum, count)`` metric pairs; empty lag bands (count 0) are left out."""
    metrics = jax.device_get(metrics)
    return {name: float(pair[0] / pair[1]) for name, pair in metrics.items() if pair[1] > 0}


class Logger:
    """One stdout line per logged step (``train/`` or ``val/`` scalars) and, optionally, every scalar to wandb."""

    def __init__(self, run_cfg, run_dir: Path):
        self.wandb = None
        if run_cfg.trainer.wandb:
            self.wandb = _start_wandb(run_cfg, run_dir)

    def log(self, step: int, scalars: dict) -> None:
        """Log ``{section/name: value}`` at ``step``."""
        for section in ("train", "val"):
            fields = " ".join(f"{key.split('/', 1)[1]}={value:.4g}" for key, value in scalars.items()
                              if key.startswith(f"{section}/"))
            if fields:
                print(f"step {step:>7d} | {section} {fields}", flush=True)
        if self.wandb is not None:
            try:
                self.wandb.log(scalars, step=step)
            except Exception as error:  # noqa: BLE001 -- telemetry never stops training
                print(f"wandb log failed ({error}); continuing without wandb", flush=True)
                self.wandb = None

    def close(self) -> None:
        if self.wandb is not None:
            self.wandb.finish()


def _start_wandb(run_cfg, run_dir: Path):
    try:
        import wandb
        id_path = run_dir / "wandb_id.txt"
        run_id = id_path.read_text().strip() if id_path.is_file() else None
        run = wandb.init(project=run_cfg.trainer.wandb_project, name=run_dir.name, dir=str(run_dir), id=run_id,
                         resume="allow" if run_id else None, config=OmegaConf.to_container(run_cfg))
        if run_id is None:
            id_path.write_text(run.id)
        return run
    except Exception as error:  # noqa: BLE001 -- telemetry never stops training
        print(f"wandb unavailable ({type(error).__name__}: {error}); logging to stdout only", flush=True)
        return None


def _stall_check(stall_logs: int, loss: float, skipped: int, window: int) -> int:
    if skipped == window or not math.isfinite(loss):
        stall_logs += 1
        if stall_logs >= STALL_LOGS:
            raise TrainingStalled(f"no progress in {stall_logs} log windows (loss={loss}, {skipped}/{window} steps "
                                  "reverted); resume from an earlier checkpoint")
        return stall_logs
    return 0


def _train_scalars(means: dict) -> dict:
    return {f"{'diagnostics' if name in DIAGNOSTICS else 'train'}/{name}": means[name]
            for name in _ORDER if name in means}


def _fresh_meta(system, x_std, fit, mesh) -> dict:
    return {
        "atom_names": list(system.atom_names),
        "scales": {"x_std_nm": x_std},
        "x_std_fit": fit,
        "runtime": {"device_count": int(mesh.devices.size)},
    }


def validation_inputs(law, dataset, batch_size: int, n_batches: int) -> list:
    """``n_batches`` fixed input batches of ``batch_size`` rows drawn from ``dataset`` with the key ``VAL_SEED``.

    Lags follow the lag law at the end of the horizon anneal, so every batch covers the full lag band.
    """
    sample = batches.make_input_sampler(law, dataset, batch_size)
    key, batches_ = jax.random.PRNGKey(VAL_SEED), []
    for _ in range(n_batches):
        key, inputs = sample(key, law.anneal_steps)
        batches_.append(inputs)
    return batches_


def _resume_step(run_dir: Path) -> int | None:
    """Latest checkpoint step of a complete run directory, or None when it holds no run files.

    Raises:
        CheckpointError: some but not all of config.yaml, kcm.json, start_position.npy and a checkpoint exist.
    """
    names = (ckpt.CONFIG_NAME, ckpt.META_NAME, ckpt.START_POSITION_NAME)
    present = [name for name in names if (run_dir / name).exists()]
    checkpoints = run_dir / "checkpoints"
    if checkpoints.is_dir() and any(checkpoints.iterdir()):
        present.append("checkpoints/")
    step = ckpt.latest_step(run_dir)
    if not present:
        return None
    if len(present) == len(names) + 1 and step is not None:
        return step
    raise ckpt.CheckpointError(f"{run_dir} holds a partial run ({', '.join(present)}); remove it or train into a "
                               "new directory")


def run(cfg, run_dir) -> Path:
    """Train a validated train config into ``run_dir`` (fresh, or resumed from its latest checkpoint).

    Checkpoints are written every ``trainer.save_every`` steps, at ``objective.db.start_step`` (with
    ``aux_loss=db``) and at ``total_steps``. Every ``trainer.val_every``
    steps the objective is evaluated, without an update, on ``trainer.val_batches`` fixed batches of
    ``data.val_file`` (see ``validation_inputs``). SIGINT/SIGTERM stop the run at the next log step after saving.

    Returns:
        the run directory.

    Raises:
        ConfigError: the resumed run's config or runtime layout differs, or the resumed run is already complete.
        CheckpointError: ``run_dir`` holds a partial run.
        TrainingStalled: see ``STALL_LOGS``.
    """
    configure_jax()
    run_dir = Path(run_dir).resolve()
    run_cfg = run_config(cfg)
    resume_step = _resume_step(run_dir)
    trainer, objective = run_cfg.trainer, run_cfg.objective
    if resume_step is not None and resume_step >= trainer.total_steps:
        raise ConfigError(f"the run is complete: resumed step {resume_step} has reached trainer.total_steps "
                          f"{trainer.total_steps}")
    mesh = make_mesh()
    n_devices = int(mesh.devices.size)
    if trainer.batch_size % (n_devices * C.LANE_BLOCK):
        raise ConfigError(f"trainer.batch_size {trainer.batch_size} must be a multiple of {C.LANE_BLOCK} x "
                          f"{n_devices} devices")
    per_device_batch = trainer.batch_size // n_devices
    n_kc, n_second = batches.lane_counts(per_device_batch)

    system = load_system(run_cfg.system)
    data_dir = Path(cfg.paths.data_dir)
    dataset = load_dataset(data_dir / run_cfg.data.file, system)
    val_dataset = load_dataset(data_dir / run_cfg.data.val_file, system)
    if resume_step is not None:
        check_resume(load_run_config(run_dir / ckpt.CONFIG_NAME), run_cfg)
        meta = json.loads((run_dir / ckpt.META_NAME).read_text())
        if meta["runtime"]["device_count"] != n_devices:
            raise ConfigError(f"resume requires {meta['runtime']['device_count']} devices, got {n_devices}")
        if tuple(meta["x_std_fit"]["shape"]) != tuple(dataset.positions.shape):
            raise ConfigError(f"the training positions have shape {tuple(dataset.positions.shape)}, the run's "
                              f"{tuple(meta['x_std_fit']['shape'])}")
    else:
        x_std, fit = fit_x_std(np.asarray(dataset.positions))
        meta = _fresh_meta(system, x_std, fit, mesh)
    scales: Scales = ckpt.check_meta(meta, run_cfg, system)
    kcm = build_kcm(run_cfg.model, system, scales)

    seed_key = jax.random.PRNGKey(trainer.seed)
    data_key = jax.random.fold_in(seed_key, 0)
    params = init_params(kcm, dataset.positions[0], seed_key)
    tx, schedule = optim.make_optimizer(run_cfg.optim, trainer.total_steps)
    state = optim.create_state(kcm.net.apply, params, tx, trainer.ema_decays)
    if resume_step is not None:
        state, data_key = ckpt.restore_state(run_dir, state, resume_step)
        save_run_config(run_dir / ckpt.CONFIG_NAME, run_cfg)
        print(f"resumed {run_dir} at step {resume_step}", flush=True)
    else:
        ckpt.write_run(run_dir, run_cfg, meta, dataset.start_position)

    db_start = int(objective.db.start_step) if objective.aux_loss == "db" else None
    second = objective.aux_loss
    print(f"{run_cfg.system.name}: {dataset.n_frames} frames, x_std {scales.x_std_nm:.6g} nm; {n_devices} devices x "
          f"{per_device_batch} rows (kc {n_kc if second else per_device_batch}"
          f"{f', {second} {n_second}' if second else ''}"
          f"{f' from step {db_start}' if db_start is not None else ''})", flush=True)

    law = batches.LagLaw.from_config(run_cfg)
    sample_inputs = batches.make_input_sampler(law, dataset, per_device_batch * n_devices)
    target = float(objective.kc.target_ema)
    phases = {}
    for active in (False, True) if db_start is not None else (False,):
        loss_fn = losses.make_loss(kcm, run_cfg, db_active=active)
        phases[active] = (optim.make_train_step(mesh, schedule, loss_fn, target_decay=target),
                          optim.make_eval_step(mesh, loss_fn, target_decay=target))

    def phase(step):
        return phases[db_start is not None and step >= db_start]

    sharding = NamedSharding(mesh, P("data"))
    val_inputs = [jax.device_put(inputs, sharding)
                  for inputs in validation_inputs(law, val_dataset, trainer.batch_size, trainer.val_batches)]

    def validate_at(state, step):
        _, eval_step = phase(step)
        totals = None
        for inputs in val_inputs:
            metrics = eval_step(state, inputs)
            totals = metrics if totals is None else jax.tree.map(jnp.add, totals, metrics)
        means = metric_means(totals)
        return {f"val/{name}": means[name] for name in VAL_METRICS if name in means}

    def is_save_step(step):
        return step % trainer.save_every == 0 or step == db_start

    start_step = int(jax.device_get(state.step))
    stop = {"now": False}
    handlers = {sig: signal.signal(sig, lambda *_: stop.__setitem__("now", True))
                for sig in (signal.SIGINT, signal.SIGTERM)}
    logger = Logger(run_cfg, run_dir)
    try:
        print(f"training steps {start_step} -> {trainer.total_steps}", flush=True)
        stall_logs, state_step, interrupted = 0, start_step, False
        window_start, skipped = start_step, 0.0
        for step in range(start_step, trainer.total_steps):
            data_key, inputs = sample_inputs(data_key, step)
            inputs = jax.device_put(inputs, sharding)
            train_step, _ = phase(step)
            state, metrics = train_step(state, inputs)
            skipped = skipped + metrics.pop("skipped_steps")
            state_step = step + 1
            if state_step % trainer.log_every == 0:
                means = metric_means(metrics)
                means["skipped_steps"] = float(jax.device_get(skipped)[0])
                logger.log(state_step, _train_scalars(means))
                stall_logs = _stall_check(stall_logs, means["loss"], round(means["skipped_steps"]),
                                          state_step - window_start)
                window_start, skipped = state_step, 0.0
                interrupted = stop["now"]
            if state_step % trainer.val_every == 0:
                logger.log(state_step, validate_at(state, state_step))
            if is_save_step(state_step):
                ckpt.save_state(run_dir, state, data_key)
            if interrupted:
                print(f"stopped by signal at step {state_step}", flush=True)
                break
        final_step = state_step if interrupted else trainer.total_steps
        if final_step > start_step and not is_save_step(final_step):
            ckpt.save_state(run_dir, state, data_key)
        print(f"done at step {final_step}: {run_dir}", flush=True)
    finally:
        logger.close()
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
    return run_dir


__all__ = ["DIAGNOSTICS", "Logger", "STALL_LOGS", "TrainingStalled", "VAL_METRICS", "VAL_SEED", "configure_jax",
           "make_mesh", "metric_means", "run", "validation_inputs"]
