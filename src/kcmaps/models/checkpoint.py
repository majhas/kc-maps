"""Checkpoints: training run directories and inference checkpoints, written and loaded strictly.

A training run directory (recognized by its ``kcm.json``)::

    config.yaml                         the six run sections (``core.config.RUN_SECTIONS``), resolved
    kcm.json                            atom names, scales {x_std_nm}, x_std fit {shape}, runtime {device_count}
    start_position.npy                  (N, 3) float32 nm: the start configuration of the samplers
    checkpoints/step_NNNNNN.msgpack     full state {step, params, opt_state, ema_params}
    checkpoints/step_NNNNNN.rng.npy     uint32[2] data key

An inference checkpoint (``export``) holds exactly three files::

    config.yaml                         sections model, system and scales {x_std_nm}
    params.msgpack                      one float32 parameter tree (the chosen EMA copy)
    start_position.npy                  (N, 3) float32 nm: the start configuration of the samplers
"""

from __future__ import annotations

import dataclasses
import io
import json
import math
import os
import re
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax.serialization import from_bytes, msgpack_restore, msgpack_serialize, to_bytes, to_state_dict
from omegaconf import OmegaConf

from ..core import constants as C
from ..core.config import ema_key, load_run_config, save_run_config, validate
from ..systems import load_system
from . import KCM, Scales, build_kcm, params_template

CONFIG_NAME = "config.yaml"
META_NAME = "kcm.json"
START_POSITION_NAME = "start_position.npy"
PARAMS_NAME = "params.msgpack"
INFERENCE_FILES = sorted((CONFIG_NAME, PARAMS_NAME, START_POSITION_NAME))
_STEP_FILE = re.compile(r"step_(\d+)\.msgpack")

META_KEYS = {"atom_names", "scales", "x_std_fit", "runtime"}
_NESTED_KEYS = {
    "scales": {"x_std_nm"},
    "x_std_fit": {"shape"},
    "runtime": {"device_count"},
}


class CheckpointError(ValueError):
    """A run directory, inference checkpoint or checkpoint file that fails strict validation."""


@dataclasses.dataclass(frozen=True)
class LoadedModel:
    """A model restored from a training run directory or an inference checkpoint.

    Attributes:
        kcm: the ``KCM`` (no force cap; use ``kcm.with_force_cap``).
        params: float32 parameters ``{'params': ...}``, on device.
        start_position: (N, 3) float32 nm, the start configuration of the samplers.
        config: the validated ``config.yaml`` (run sections, or model/system/scales).
        path: the checkpoint directory.
        step: training step of the parameters (None for an inference checkpoint).
        ema: EMA decay of the parameters (None for an inference checkpoint).
    """

    kcm: KCM
    params: Any
    start_position: np.ndarray
    config: Any
    path: Path
    step: int | None
    ema: float | None


def checkpoint_path(run_dir, step: int) -> Path:
    return Path(run_dir) / "checkpoints" / f"step_{int(step):06d}.msgpack"


def rng_path(run_dir, step: int) -> Path:
    return Path(run_dir) / "checkpoints" / f"step_{int(step):06d}.rng.npy"


def latest_step(run_dir) -> int | None:
    """Largest step with a ``checkpoints/step_N.msgpack`` file, or None."""
    directory = Path(run_dir) / "checkpoints"
    steps = [int(m.group(1)) for p in directory.glob("step_*.msgpack") if (m := _STEP_FILE.fullmatch(p.name))]
    return max(steps) if steps else None


def _write_atomic(path: Path, payload: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


# -- strict tree validation --------------------------------------------------------------------------


def _signature(value):
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        return tuple(value.shape), np.dtype(value.dtype)
    array = np.asarray(value)
    return array.shape, array.dtype


def validate_tree(expected, actual, path: str = "state") -> None:
    """Require identical container keys and leaf shapes and dtypes.

    Raises:
        CheckpointError: on the first difference.
    """
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            raise CheckpointError(f"{path}: expected a mapping, found {type(actual).__name__}")
        if set(expected) != set(actual):
            raise CheckpointError(f"{path}: missing {sorted(set(expected) - set(actual))}, "
                                  f"unexpected {sorted(set(actual) - set(expected))}")
        for key in expected:
            validate_tree(expected[key], actual[key], f"{path}.{key}")
        return
    if isinstance(expected, (tuple, list)):
        if not isinstance(actual, (tuple, list)) or len(actual) != len(expected):
            raise CheckpointError(f"{path}: container length differs")
        for index, (wanted, found) in enumerate(zip(expected, actual)):
            validate_tree(wanted, found, f"{path}[{index}]")
        return
    if isinstance(actual, (dict, list, tuple)):
        raise CheckpointError(f"{path}: expected a leaf")
    (shape, dtype), (actual_shape, actual_dtype) = _signature(expected), _signature(actual)
    if (shape, dtype) != (actual_shape, actual_dtype):
        raise CheckpointError(f"{path}: leaf {actual_shape} {actual_dtype} differs from {shape} {dtype}")


def _require_finite(tree, path: str) -> None:
    for keypath, leaf in jax.tree_util.tree_leaves_with_path(tree):
        array = np.asarray(leaf)
        if np.issubdtype(array.dtype, np.floating) and not np.isfinite(array).all():
            raise CheckpointError(f"{path}: non-finite values at {jax.tree_util.keystr(keypath)}")


# -- run directory -----------------------------------------------------------------------------------


def _check_start_position(position, n_atoms=None) -> np.ndarray:
    position = np.asarray(position)
    if position.ndim != 2 or position.shape[1] != 3 or (n_atoms is not None and position.shape[0] != n_atoms) \
            or position.dtype != np.float32 or not np.isfinite(position).all():
        raise CheckpointError(f"the start position must be finite float32 ({n_atoms or 'N'}, 3)")
    return position


def write_run(run_dir, run_cfg, meta: dict, start_position) -> None:
    """Create a run directory with ``config.yaml``, ``kcm.json``, ``start_position.npy`` and ``checkpoints/``.

    Raises:
        CheckpointError: ``start_position`` is not finite float32 (N, 3).
    """
    position = _check_start_position(start_position)
    run_dir = Path(run_dir)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    save_run_config(run_dir / CONFIG_NAME, run_cfg)
    _write_atomic(run_dir / META_NAME, (json.dumps(meta, indent=2, allow_nan=False) + "\n").encode())
    np.save(run_dir / START_POSITION_NAME, position)


def save_state(run_dir, state, rng_key) -> Path:
    """Write ``checkpoints/step_N.rng.npy`` and then ``step_N.msgpack`` (the commit point) for a full state.

    Args:
        state: a flax struct whose state dict is {step, params, opt_state, ema_params}.
        rng_key: uint32[2] data key.
    """
    step = int(jax.device_get(state.step))
    key = _check_rng_key(np.asarray(jax.device_get(rng_key)))
    (Path(run_dir) / "checkpoints").mkdir(parents=True, exist_ok=True)
    buffer = io.BytesIO()
    np.save(buffer, key)
    _write_atomic(rng_path(run_dir, step), buffer.getvalue())
    path = checkpoint_path(run_dir, step)
    _write_atomic(path, to_bytes(jax.device_get(state)))
    return path


def restore_state(run_dir, template, step: int):
    """Restore a full state written by ``save_state``.

    Args:
        template: a state with the expected structure (its values are ignored).
        step: the step to restore.

    Returns:
        (state, rng_key uint32[2]).

    Raises:
        CheckpointError: structure, step or finiteness mismatch.
    """
    raw = _read_state(run_dir, step, to_state_dict(template))
    state = from_bytes(template, msgpack_serialize(raw))
    return state, jnp.asarray(_check_rng_key(np.load(rng_path(run_dir, step))))


def _check_rng_key(key: np.ndarray) -> np.ndarray:
    if key.dtype != np.uint32 or key.shape != (2,):
        raise CheckpointError("the data key must be a raw uint32[2] key")
    return key


def _read_state(run_dir, step: int, expected: dict) -> dict:
    """Read ``checkpoints/step_N.msgpack``; check the full-state keys, the tree of the ``expected`` entries, the step
    and finiteness."""
    raw = msgpack_restore(checkpoint_path(run_dir, step).read_bytes())
    if not isinstance(raw, dict) or set(raw) != {"step", "params", "opt_state", "ema_params"}:
        raise CheckpointError("a checkpoint must hold {step, params, opt_state, ema_params}")
    checked = {name: raw[name] for name in expected}
    validate_tree(expected, checked)
    if int(raw["step"]) != int(step):
        raise CheckpointError(f"file step {step} differs from state step {int(raw['step'])}")
    _require_finite(checked, "checkpoint")
    return raw


def _check_meta_keys(meta) -> None:
    if not isinstance(meta, dict) or set(meta) != META_KEYS:
        raise CheckpointError(f"kcm.json keys must be {sorted(META_KEYS)}")
    for name, keys in _NESTED_KEYS.items():
        if not isinstance(meta[name], dict) or set(meta[name]) != keys:
            raise CheckpointError(f"kcm.json {name} keys must be {sorted(keys)}")


def make_scales(x_std_nm, model_cfg, system, where: str) -> Scales:
    """``Scales`` of a stored position unit: T_std_ps from the model config, p_std = sqrt(m kBT) of the system."""
    if type(x_std_nm) is not float or not math.isfinite(x_std_nm) or x_std_nm <= 0:
        raise CheckpointError(f"{where} x_std_nm must be a positive float")
    p_std = np.sqrt(np.asarray(system.masses, np.float64) * system.kbt_kj_per_mol)
    return Scales(T_std_ps=float(model_cfg.T_std_ps), x_std_nm=x_std_nm, p_std=tuple(float(v) for v in p_std))


def check_meta(meta: dict, run_cfg, system) -> Scales:
    """Validate ``kcm.json`` against the run config and the system; return its ``Scales``.

    Raises:
        CheckpointError: any disagreement.
    """
    _check_meta_keys(meta)
    if meta["atom_names"] != list(system.atom_names):
        raise CheckpointError("kcm.json atom_names disagree with the atoms of the PDB")
    scales = make_scales(meta["scales"]["x_std_nm"], run_cfg.model, system, "kcm.json")
    shape = meta["x_std_fit"]["shape"]
    if not isinstance(shape, list) or len(shape) != 3 or type(shape[0]) is not int or shape[0] < 1 \
            or shape[1:] != [system.n_atoms, 3]:
        raise CheckpointError(f"kcm.json x_std_fit shape must be (frames, {system.n_atoms}, 3)")
    return scales


# -- loading -----------------------------------------------------------------------------------------


def _is_inference(path: Path) -> bool:
    if (path / META_NAME).is_file():
        return False
    if sorted(p.name for p in path.iterdir()) != INFERENCE_FILES:
        raise CheckpointError(f"{path} is neither a training run directory (with {META_NAME}) nor an inference "
                              f"checkpoint (exactly {INFERENCE_FILES})")
    return True


def _read_config(path: Path, inference: bool):
    if inference:
        return validate(OmegaConf.to_container(OmegaConf.load(path / CONFIG_NAME), resolve=False), "inference")
    return load_run_config(path / CONFIG_NAME)


def load_config(path):
    """The validated ``config.yaml`` of a training run directory or an inference checkpoint (both have
    ``system``)."""
    path = Path(path)
    return _read_config(path, _is_inference(path))


def _build(model_cfg, system_cfg, scales_of, attention):
    system = load_system(system_cfg)
    kcm = build_kcm(model_cfg, system, scales_of(system), attention=attention)
    return kcm, system


def load_checkpoint(path, *, step: int | None = None, ema: float | None = None,
                    attention: str = "native") -> LoadedModel:
    """Strictly load a training run directory (one EMA copy of one step) or an inference checkpoint.

    Sets the default matmul precision to HIGHEST. Loading runs with JAX x64 off, and the model expects x64 off (the
    JAX default). Of a full state only ``step``, ``params`` and ``ema_params`` are validated; the optimizer state is
    ignored.

    Args:
        path: training run directory or inference checkpoint.
        step: checkpoint step of a run directory; None = latest.
        ema: EMA decay of a run directory, one of ``trainer.ema_decays``; None = ``objective.kc.target_ema``.
        attention: 'native' or the inference-only 'bf16_cudnn'.

    Raises:
        CheckpointError, ConfigError, ValueError: any disagreement between files, PDB, scales and parameter tree.
    """
    jax.config.update("jax_default_matmul_precision", C.MATMUL_PRECISION)
    with jax.enable_x64(False):
        path = Path(path).resolve()
        inference = _is_inference(path)
        config = _read_config(path, inference)
        if inference:
            if step is not None or ema is not None:
                raise CheckpointError("an inference checkpoint holds one parameter set; step and ema do not apply")
            x_std = config.scales.x_std_nm
            kcm, system = _build(config.model, config.system,
                                 lambda system: make_scales(x_std, config.model, system, CONFIG_NAME), attention)
            raw = msgpack_restore((path / PARAMS_NAME).read_bytes())
            validate_tree(params_template(kcm), raw, "params")
            _require_finite(raw, "params")
            chosen = raw
        else:
            meta = json.loads((path / META_NAME).read_text())
            kcm, system = _build(config.model, config.system, lambda system: check_meta(meta, config, system),
                                 attention)
            step = latest_step(path) if step is None else int(step)
            if step is None:
                raise CheckpointError(f"no checkpoints in {path / 'checkpoints'}")
            template = params_template(kcm)
            expected = {"step": np.int32(0), "params": template,
                        "ema_params": {ema_key(d): template for d in config.trainer.ema_decays}}
            raw = _read_state(path, step, expected)
            ema = float(config.objective.kc.target_ema) if ema is None else float(ema)
            if ema_key(ema) not in raw["ema_params"]:
                raise CheckpointError(f"EMA {ema} is not in the checkpoint; available {sorted(raw['ema_params'])}")
            chosen = raw["ema_params"][ema_key(ema)]
        start_position = _check_start_position(np.load(path / START_POSITION_NAME, allow_pickle=False),
                                               system.n_atoms)
        params = jax.device_put(jax.tree.map(np.asarray, chosen))
        return LoadedModel(kcm=kcm, params=params, start_position=start_position, config=config, path=path,
                           step=step, ema=ema)


def export(run_dir, ema: float | None, output) -> Path:
    """Write the inference checkpoint of one EMA copy of a run directory's latest step to ``output`` (created; must
    not exist).

    Args:
        run_dir: training run directory.
        ema: EMA decay; None = ``objective.kc.target_ema`` of the run config.
        output: the new inference checkpoint directory.
    """
    model = load_checkpoint(run_dir, ema=ema)
    meta = json.loads((Path(run_dir) / META_NAME).read_text())
    config = validate({"model": OmegaConf.to_container(model.config.model),
                       "system": OmegaConf.to_container(model.config.system), "scales": meta["scales"]}, "inference")
    output = Path(output)
    output.mkdir(parents=True)
    (output / CONFIG_NAME).write_text(OmegaConf.to_yaml(config))
    _write_atomic(output / PARAMS_NAME, msgpack_serialize(jax.tree.map(np.asarray, jax.device_get(model.params))))
    np.save(output / START_POSITION_NAME, model.start_position)
    return output


__all__ = ["CheckpointError", "LoadedModel", "check_meta", "checkpoint_path", "export", "latest_step",
           "load_checkpoint", "load_config", "make_scales", "restore_state", "rng_path", "save_state", "validate_tree", "write_run"]
