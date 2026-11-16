"""Typed config schemas, strict validation and run-config IO.

Every entry point resolves its Hydra config and passes it through ``validate``: unknown keys, missing keys, unresolved
``???`` values and wrong types fail with ``ConfigError``; then the semantic checks of the task run.
"""

from __future__ import annotations

import dataclasses
import math
import typing
from fractions import Fraction
from pathlib import Path
from typing import Any, List, NamedTuple, Optional, Union

from omegaconf import DictConfig, OmegaConf

from . import constants as C

RESUME_MUTABLE = (("trainer", "log_every"), ("trainer", "save_every"),
                  ("trainer", "val_every"), ("trainer", "val_batches"), ("trainer", "wandb"),
                  ("trainer", "wandb_project"), ("data", "val_file"))

if not OmegaConf.has_resolver("mul"):
    OmegaConf.register_new_resolver("mul", lambda a, b: a * b)


class ConfigError(ValueError):
    """An invalid configuration."""


# -- schemas (values come only from the yaml files) ------------------------------------------------


@dataclasses.dataclass
class SystemSchema:
    name: str
    pdb: str
    temperature_K: float
    gamma_ps_inv: float
    forcefield: List[str]
    cutoff_nm: Optional[float]


@dataclasses.dataclass
class DataSchema:
    file: str
    val_file: str


@dataclasses.dataclass
class ModelSchema:
    num_blocks: int
    layers_per_block: int
    source_layers: int
    channels: int
    num_heads: int
    ffn_hidden: int
    conditioner_hidden: int
    time_embedding_max_freq_per_ps: float
    T_std_ps: float
    train_tau_min_ps: float
    train_tau_max_ps: float


@dataclasses.dataclass
class HotSourceSchema:
    fraction: float
    mean_K: float
    std_K: float
    floor_K: float


@dataclasses.dataclass
class KCSchema:
    weight: float
    cap: float
    target_ema: float
    log_uniform_fraction: float
    hot_source: HotSourceSchema


@dataclasses.dataclass
class DBSchema:
    weight: float
    start_step: int
    ramp_steps: int


@dataclasses.dataclass
class PFSchema:
    weight: float
    start_step: int
    ramp_steps: int
    tau_max_ps: float


@dataclasses.dataclass
class ObjectiveSchema:
    kc: KCSchema
    aux_loss: Optional[str]
    db: DBSchema
    pf: PFSchema


AUX_LOSSES = (None, "db", "pf")


@dataclasses.dataclass
class OptimSchema:
    lr: float
    end_lr: float
    warmup_fraction: float
    b1: float
    b2: float
    eps: float
    weight_decay: float
    clip_norm: float


@dataclasses.dataclass
class TrainerSchema:
    seed: int
    total_steps: int
    batch_size: int
    ema_decays: List[float]
    horizon_start_ps: float
    horizon_anneal_fraction: float
    log_every: int
    val_every: int
    val_batches: int
    save_every: int
    wandb: bool
    wandb_project: str


@dataclasses.dataclass
class PathsSchema:
    root_dir: str
    data_dir: str
    log_dir: str
    output_dir: str


@dataclasses.dataclass
class RunSchema:
    system: SystemSchema
    data: DataSchema
    model: ModelSchema
    objective: ObjectiveSchema
    optim: OptimSchema
    trainer: TrainerSchema


RUN_SECTIONS = tuple(field.name for field in dataclasses.fields(RunSchema))


@dataclasses.dataclass
class TrainSchema(RunSchema):
    task_name: str
    run_name: str
    paths: PathsSchema


@dataclasses.dataclass
class ScalesSchema:
    """The one fitted unit; T_std_ps comes from the model section and p_std = sqrt(m kBT) from the system."""
    x_std_nm: float


@dataclasses.dataclass
class InferenceSchema:
    model: ModelSchema
    system: SystemSchema
    scales: ScalesSchema


@dataclasses.dataclass
class SystemSectionSchema:
    system: SystemSchema


@dataclasses.dataclass
class AdaptiveSchema:
    name: str
    ema: Optional[float]
    attention: str
    chains: int
    seed: int
    tau_max_fs: float
    tau_min_fs: float
    force_cap: Optional[float]
    redraw_attempts: int
    target_ns: float
    nfe_cap_factor: float
    frame_ps: float
    save_every_ps: float
    max_chunks: Optional[int]
    wall_seconds: Optional[float]
    unroll: int


@dataclasses.dataclass
class MHSchema:
    name: str
    ema: Optional[float]
    attention: str
    sources: Optional[str]
    tau_fs: float
    walkers_per_source: int
    burn_in_steps: int
    steps: int
    batch_size: int
    step_chunk: int
    seed: int


SAMPLERS = {"adaptive": AdaptiveSchema, "mh": MHSchema}


@dataclasses.dataclass
class SimulateSchema:
    task_name: str
    checkpoint: str
    step: Optional[int]
    resume: bool
    run_name: str
    sampler: Union[AdaptiveSchema, MHSchema]
    paths: PathsSchema


@dataclasses.dataclass
class KineticsSchema:
    enabled: bool
    bootstrap: int
    seed: int
    max_lag_ps: float


@dataclasses.dataclass
class EvaluateSchema:
    task_name: str
    simulation: str
    reference_dir: str
    budget_minutes: float
    burn_in_fraction: float
    n_frames: int
    n_draws: int
    draw_seed: int
    min_frames: int
    floor: bool
    energy_batch: int
    tica_batch: int
    kinetics: KineticsSchema
    run_name: str
    paths: PathsSchema


# -- strict structural check ----------------------------------------------------------------------


def _check(value: Any, hint: Any, path: str) -> Any:
    """Return ``value`` as plain Python data typed by ``hint``; raise ConfigError on any mismatch."""
    if isinstance(value, str) and value == "???":
        raise ConfigError(f"{path}: missing value")
    origin = typing.get_origin(hint)
    if origin is Union:
        args = typing.get_args(hint)
        if value is None:
            if type(None) in args:
                return None
            raise ConfigError(f"{path}: must not be null")
        options = [option for option in args if option is not type(None)]
        if len(options) > 1:  # the sampler union, selected by its name
            schema = SAMPLERS.get(value.get("name")) if isinstance(value, dict) else None
            if schema not in options:
                raise ConfigError(f"{path}: cannot select a schema for {value!r}")
            return _check(value, schema, path)
        return _check(value, options[0], path)
    if dataclasses.is_dataclass(hint):
        if not isinstance(value, dict):
            raise ConfigError(f"{path}: expected a mapping, got {type(value).__name__}")
        hints = typing.get_type_hints(hint)
        missing, unknown = set(hints) - set(value), set(value) - set(hints)
        if missing or unknown:
            raise ConfigError(f"{path}: missing keys {sorted(missing)}, unknown keys {sorted(unknown)}")
        return {key: _check(value[key], hints[key], f"{path}.{key}" if path else key) for key in hints}
    if origin in (list, List):
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"{path}: expected a list, got {type(value).__name__}")
        (item,) = typing.get_args(hint)
        return [_check(entry, item, f"{path}[{i}]") for i, entry in enumerate(value)]
    if hint is bool:
        if type(value) is not bool:
            raise ConfigError(f"{path}: expected bool, got {value!r}")
        return value
    if hint is int:
        if type(value) is not int:
            raise ConfigError(f"{path}: expected int, got {value!r}")
        return value
    if hint is float:
        if type(value) is int:
            return float(value)
        if type(value) is not float:
            raise ConfigError(f"{path}: expected float, got {value!r}")
        return value
    if hint is str:
        if not isinstance(value, str):
            raise ConfigError(f"{path}: expected str, got {value!r}")
        return value
    raise ConfigError(f"{path}: unsupported schema type {hint!r}")


def _conform(config, schema) -> DictConfig:
    """Check a mapping (DictConfig or dict, resolved here) against a schema dataclass and return it as a read-only
    DictConfig in struct mode.

    Raises:
        ConfigError: unknown or missing keys, ``???`` values, wrong types or unresolvable interpolations.
    """
    try:
        container = {key: OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value
                     for key, value in config.items() if key != "hydra"}
    except Exception as error:
        raise ConfigError(f"cannot resolve config: {error}") from error
    plain = _check(container, schema, "")
    cfg = OmegaConf.create(plain)
    OmegaConf.set_struct(cfg, True)
    OmegaConf.set_readonly(cfg, True)
    return cfg


# -- semantic checks ------------------------------------------------------------------------------

def ema_key(decay: float) -> str:
    """Checkpoint key of an EMA decay, ``f"{decay:.9f}"``."""
    return f"{float(decay):.9f}"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigError(message)


def _finite_positive(value: float) -> bool:
    return math.isfinite(value) and value > 0


def _reaches_weight(start_step: int, ramp_steps: int, total_steps: int) -> bool:
    # Updates run at steps 0..total_steps-1; a ramp is still zero at start_step itself.
    return start_step + (1 if ramp_steps > 0 else 0) < total_steps


def _validate_system(cfg) -> None:
    system = cfg.system
    _require(len(system.forcefield) > 0, "system.forcefield must not be empty")
    _require(system.cutoff_nm is None or _finite_positive(system.cutoff_nm), "system.cutoff_nm must be null or > 0")
    _require(_finite_positive(system.temperature_K) and _finite_positive(system.gamma_ps_inv),
             "system.temperature_K and system.gamma_ps_inv must be positive")


def _validate_model(cfg) -> None:
    _validate_system(cfg)
    model = cfg.model
    _require(model.num_blocks > 0 and model.num_blocks % 2 == 0, "model.num_blocks must be even and positive")
    _require(min(model.layers_per_block, model.source_layers, model.channels, model.num_heads, model.ffn_hidden,
                 model.conditioner_hidden) > 0, "model widths and depths must be positive")
    _require(model.channels % model.num_heads == 0, "model.channels must be divisible by model.num_heads")
    _require(_finite_positive(model.time_embedding_max_freq_per_ps) and _finite_positive(model.T_std_ps),
             "model.time_embedding_max_freq_per_ps and model.T_std_ps must be positive")
    _require(C.CONTEXT_LOG_WIDTH_FLOOR_PS <= model.train_tau_min_ps
             and 2 * model.train_tau_min_ps < model.train_tau_max_ps,
             "the lag band needs 1e-6 ps <= model.train_tau_min_ps and 2 train_tau_min_ps < train_tau_max_ps")


def _validate_run(cfg) -> None:
    _validate_model(cfg)
    model, objective, optim, trainer = cfg.model, cfg.objective, cfg.optim, cfg.trainer
    _require(2 * model.train_tau_min_ps <= trainer.horizon_start_ps < model.train_tau_max_ps,
             "trainer.horizon_start_ps must lie in [2 model.train_tau_min_ps, model.train_tau_max_ps)")
    _require(objective.aux_loss in AUX_LOSSES, "objective.aux_loss must be db, pf or null")
    kc = objective.kc
    _require(_finite_positive(kc.weight) and math.isfinite(kc.cap) and kc.cap >= 0,
             "objective.kc.weight must be positive and objective.kc.cap nonnegative")
    _require(0 < kc.log_uniform_fraction < 1, "objective.kc.log_uniform_fraction must lie in (0, 1)")
    hot = kc.hot_source
    _require(0 <= hot.fraction <= 1 and _finite_positive(hot.mean_K) and math.isfinite(hot.std_K) and hot.std_K >= 0
             and _finite_positive(hot.floor_K), "objective.kc.hot_source values out of range")
    if objective.aux_loss == "db":
        db = objective.db
        _require(_finite_positive(db.weight), "objective.db.weight must be positive")
        _require(1 <= db.start_step < trainer.total_steps, "objective.db.start_step must lie in [1, total_steps)")
        _require(db.ramp_steps >= 0, "objective.db.ramp_steps must be nonnegative")
        _require(_reaches_weight(db.start_step, db.ramp_steps, trainer.total_steps),
                 "objective.db must reach a positive weight before trainer.total_steps")
    if objective.aux_loss == "pf":
        pf = objective.pf
        _require(_finite_positive(pf.weight), "objective.pf.weight must be positive")
        _require(pf.start_step >= 0 and pf.ramp_steps >= 0, "objective.pf steps must be nonnegative")
        _require(_reaches_weight(pf.start_step, pf.ramp_steps, trainer.total_steps),
                 "objective.pf must reach a positive weight before trainer.total_steps")
        _require(model.train_tau_min_ps < pf.tau_max_ps <= model.train_tau_max_ps,
                 "objective.pf.tau_max_ps must lie in (model.train_tau_min_ps, model.train_tau_max_ps]")
    _require(_finite_positive(optim.lr) and math.isfinite(optim.end_lr) and optim.end_lr >= 0,
             "optim.lr must be positive and optim.end_lr nonnegative")
    _require(0 <= optim.warmup_fraction < 1, "optim.warmup_fraction must lie in [0, 1)")
    _require(0 <= optim.b1 < 1 and 0 <= optim.b2 < 1, "optim.b1 and optim.b2 must lie in [0, 1)")
    _require(_finite_positive(optim.eps) and _finite_positive(optim.clip_norm)
             and math.isfinite(optim.weight_decay) and optim.weight_decay >= 0, "optimizer constants out of range")
    _require(trainer.total_steps >= 1, "trainer.total_steps must be positive")
    _require(round(optim.warmup_fraction * trainer.total_steps) < trainer.total_steps,
             "the warmup must end before trainer.total_steps")
    _require(trainer.batch_size > 0 and trainer.batch_size % C.LANE_BLOCK == 0,
             "trainer.batch_size must be a positive multiple of 4")
    _require(0 <= trainer.horizon_anneal_fraction <= 1, "trainer.horizon_anneal_fraction must lie in [0, 1]")
    _require(min(trainer.log_every, trainer.val_every, trainer.val_batches, trainer.save_every) >= 1,
             "trainer.log_every, val_every, val_batches and save_every must be >= 1")
    decays = list(trainer.ema_decays)
    _require(len(decays) > 0 and all(0 < d < 1 for d in decays), "trainer.ema_decays must lie in (0, 1)")
    keys = [ema_key(d) for d in decays]
    _require(len(set(keys)) == len(keys), "trainer.ema_decays must have distinct 9-decimal keys")
    _require(all(float(k) == float(d) for k, d in zip(keys, decays)),
             "trainer.ema_decays must round-trip through their 9-decimal keys")
    _require(kc.target_ema in decays, "objective.kc.target_ema must be one of trainer.ema_decays")


_INT32_MAX, _UINT32_MAX, _UINT64_MAX = 2**31 - 1, 2**32 - 1, 2**64 - 1


class AdaptiveClock(NamedTuple):
    """Adaptive-sampler settings on the 1/16 fs tick clock."""

    tau_max_ticks: int
    tau_min_ticks: int
    target_ticks: int
    nfe_cap: int
    stride: int
    chunk_blocks: int


def _on_clock(value_fs: float, name: str) -> int:
    ticks = round(value_fs * C.TICKS_PER_FS)
    _require(math.isfinite(value_fs) and math.isclose(value_fs * C.TICKS_PER_FS, ticks, rel_tol=0, abs_tol=1e-9),
             f"{name} must lie on the 1/16 fs clock")
    return ticks


def adaptive_clock(*, tau_max_fs: float, tau_min_fs: float, target_ns: float, nfe_cap_factor: float,
                   frame_ps: float, save_every_ps: float) -> AdaptiveClock:
    """Tick-clock settings of the adaptive sampler: lags, target, NFE cap ceil(nfe_cap_factor * target / tau_max),
    frame stride max(1, round(frame_ps / tau_max)) iterations and save_every_ps / frame_ps frames per chunk.

    Raises:
        ConfigError: a value off the clock or out of range.
    """
    tau_max_ticks = _on_clock(tau_max_fs, "sampler.tau_max_fs")
    tau_min_ticks = _on_clock(tau_min_fs, "sampler.tau_min_fs")
    _require(C.LAG_MIN_TICKS <= tau_min_ticks <= tau_max_ticks <= C.LAG_MAX_TICKS,
             "sampler lags need 1.5625 fs <= tau_min_fs <= tau_max_fs <= 100 fs")
    _require(_finite_positive(target_ns), "sampler.target_ns must be positive")
    requested = target_ns * 1000 / C.TICK_PS
    target_ticks = round(requested)
    _require(math.isclose(requested, target_ticks, rel_tol=0, abs_tol=1e-5) and target_ticks % C.LAG_MIN_TICKS == 0
             and target_ticks + tau_max_ticks <= _UINT64_MAX,
             "sampler.target_ns must be a multiple of 1.5625 fs that fits the uint64 clock")
    _require(math.isfinite(nfe_cap_factor) and nfe_cap_factor >= 1, "sampler.nfe_cap_factor must be >= 1")
    _require(_finite_positive(frame_ps) and _finite_positive(save_every_ps),
             "sampler.frame_ps and sampler.save_every_ps must be positive")
    frame_ticks = _on_clock(frame_ps * 1000, "sampler.frame_ps")
    save_ticks = _on_clock(save_every_ps * 1000, "sampler.save_every_ps")
    _require(save_ticks % frame_ticks == 0 and 1 <= save_ticks // frame_ticks <= _INT32_MAX,
             "sampler.save_every_ps must be a whole multiple of sampler.frame_ps (at most int32 frames per chunk)")
    nfe_cap = math.ceil(Fraction(repr(float(nfe_cap_factor))) * target_ticks / tau_max_ticks)
    stride = max(1, round(frame_ps * C.TICKS_PER_PS / tau_max_ticks))
    _require(nfe_cap <= _INT32_MAX, f"the NFE cap ceil(nfe_cap_factor * target / tau_max) = {nfe_cap} exceeds int32 "
             f"at tau_max_fs {tau_max_fs}; lower sampler.target_ns")
    _require(stride <= _INT32_MAX, "the frame stride round(frame_ps / tau_max) must fit int32")
    return AdaptiveClock(tau_max_ticks, tau_min_ticks, target_ticks, nfe_cap, stride, save_ticks // frame_ticks)


def _validate_simulate(cfg) -> None:
    sampler = cfg.sampler
    _require(sampler.attention in ("native", "bf16_cudnn"), "sampler.attention must be native or bf16_cudnn")
    _require(sampler.ema is None or 0 < sampler.ema < 1, "sampler.ema must be null or in (0, 1)")
    if sampler.name == "adaptive":
        _require(1 <= sampler.chains <= _INT32_MAX, "sampler.chains must be an int32 >= 1")
        _require(0 <= sampler.seed <= _UINT32_MAX, "sampler.seed must fit uint32")
        _require(sampler.force_cap is None or _finite_positive(sampler.force_cap), "sampler.force_cap must be > 0")
        _require(1 <= sampler.redraw_attempts <= 64, "sampler.redraw_attempts must lie in [1, 64]")
        _require(1 <= sampler.unroll <= _INT32_MAX
                 and (sampler.max_chunks is None or 1 <= sampler.max_chunks <= _INT32_MAX),
                 "sampler.unroll and max_chunks (or null) must be int32 >= 1")
        _require(sampler.wall_seconds is None or _finite_positive(sampler.wall_seconds),
                 "sampler.wall_seconds must be null or > 0")
        adaptive_clock(tau_max_fs=sampler.tau_max_fs, tau_min_fs=sampler.tau_min_fs, target_ns=sampler.target_ns,
                       nfe_cap_factor=sampler.nfe_cap_factor, frame_ps=sampler.frame_ps,
                       save_every_ps=sampler.save_every_ps)
    else:
        _require(not cfg.resume, "resume applies to the adaptive sampler only")
        _require(_finite_positive(sampler.tau_fs), "sampler.tau_fs must be positive")
        _require(min(sampler.walkers_per_source, sampler.steps, sampler.batch_size, sampler.step_chunk) >= 1
                 and sampler.burn_in_steps >= 0, "MH sampler counts out of range")
        _require(0 <= sampler.seed <= _UINT32_MAX, "sampler.seed must fit uint32")
        _require(sampler.burn_in_steps + sampler.steps <= 2**32, "MH steps must fit uint32")
    _require(cfg.step is None or cfg.step >= 0, "step must be null or nonnegative")


def _validate_evaluate(cfg) -> None:
    _require(cfg.n_frames >= 1 and cfg.n_draws >= 2 and 1 <= cfg.min_frames <= cfg.n_frames,
             "evaluation needs n_frames >= 1, n_draws >= 2 and min_frames in [1, n_frames]")
    _require(0 <= cfg.burn_in_fraction < 1, "burn_in_fraction must lie in [0, 1)")
    _require(_finite_positive(cfg.budget_minutes), "budget_minutes must be positive")
    _require(cfg.draw_seed >= 0 and cfg.energy_batch > 0 and cfg.tica_batch > 0,
             "draw_seed must be >= 0 and the batch sizes > 0")
    kinetics = cfg.kinetics
    _require(kinetics.bootstrap > 0 and kinetics.seed >= 0 and _finite_positive(kinetics.max_lag_ps),
             "kinetics.bootstrap, seed and max_lag_ps out of range")


TASKS = {"train": (TrainSchema, _validate_run), "run": (RunSchema, _validate_run),
         "inference": (InferenceSchema, _validate_model), "system": (SystemSectionSchema, _validate_system),
         "simulate": (SimulateSchema, _validate_simulate), "evaluate": (EvaluateSchema, _validate_evaluate)}


def validate(cfg, task: str) -> DictConfig:
    """Strictly validate a config for ``task`` in {'train', 'simulate', 'evaluate', 'run', 'inference', 'system'}
    ('run' = the six run sections of a run directory, 'inference' = the model, system and scales of an inference
    checkpoint, 'system' = the system section of a simulation's ``run_spec.json``).

    Returns:
        A resolved, read-only DictConfig without the ``hydra`` node.

    Raises:
        ConfigError: on any structural or semantic error.
    """
    if task not in TASKS:
        raise ConfigError(f"unknown task {task!r}")
    schema, check = TASKS[task]
    merged = _conform(cfg, schema)
    check(merged)
    return merged


def run_config(cfg) -> DictConfig:
    """The six run sections of a validated train config."""
    return validate({name: OmegaConf.to_container(cfg[name], resolve=True) for name in RUN_SECTIONS}, "run")


def save_run_config(path, run_cfg) -> None:
    """Write a validated run config (``run_config``) to ``path`` (yaml)."""
    Path(path).write_text(OmegaConf.to_yaml(run_cfg))


def load_run_config(path) -> DictConfig:
    """Read and validate a run directory's ``config.yaml``."""
    return validate(OmegaConf.to_container(OmegaConf.load(path), resolve=False), "run")


def check_resume(saved, new) -> None:
    """Raise ConfigError unless two validated run configs differ only in ``RESUME_MUTABLE`` fields."""
    left = OmegaConf.to_container(saved)
    right = OmegaConf.to_container(new)
    for section, key in RESUME_MUTABLE:
        left[section].pop(key)
        right[section].pop(key)
    if left != right:
        differing = sorted(f"{s}.{k}" for s in left for k in set(left[s]) | set(right[s])
                           if left[s].get(k) != right[s].get(k))
        raise ConfigError(f"resume requires an identical run config; differing: {differing}")
