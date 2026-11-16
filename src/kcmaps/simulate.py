"""Entry point ``python -m kcmaps.simulate checkpoint=<inference checkpoint or run dir> [sampler=mh]``.

The model, system, scales and start configuration come from the checkpoint directory. Output goes to the Hydra run
directory (``paths.output_dir``). Unless ``XLA_FLAGS`` already sets it, ``--xla_gpu_enable_cublaslt=true`` is added to
``XLA_FLAGS`` before JAX starts a backend; the flags in effect are recorded in the adaptive ``run_spec.json``.
"""

from __future__ import annotations

import os

CUBLASLT_FLAG = "--xla_gpu_enable_cublaslt=true"


def _default_xla_flags() -> None:
    flags = os.environ.get("XLA_FLAGS", "")
    if "xla_gpu_enable_cublaslt" not in flags:
        os.environ["XLA_FLAGS"] = f"{flags} {CUBLASLT_FLAG}".strip()


# XLA reads the flags when the first backend starts, which importing jax does not do.
_default_xla_flags()

from pathlib import Path  # noqa: E402

import hydra  # noqa: E402
from omegaconf import DictConfig  # noqa: E402

from kcmaps.models.checkpoint import load_checkpoint  # noqa: E402
from kcmaps.core.config import ConfigError, validate  # noqa: E402

# Sampler fields that select the model or the sources rather than being sampler arguments.
NON_ARGS = ("name", "ema", "attention", "sources")


def simulate(cfg) -> Path:
    """Validate a ``simulate`` config, load the checkpoint and run ``sampler.name`` (adaptive | mh).

    Returns:
        The output directory.

    Raises:
        ConfigError: invalid config. CheckpointError, ValueError: from loading or sampling.
    """
    cfg = validate(cfg, "simulate")
    sampler = cfg.sampler
    model = load_checkpoint(cfg.checkpoint, step=cfg.step, ema=sampler.ema, attention=sampler.attention)
    output = Path(cfg.paths.output_dir)
    args = {key: value for key, value in sampler.items() if key not in NON_ARGS}
    if sampler.name == "adaptive":
        from kcmaps.samplers import run_chains

        return run_chains(model, output, resume=cfg.resume, **args)
    import numpy as np

    from kcmaps.samplers import run_mh

    sources = None if sampler.sources is None else np.load(sampler.sources, allow_pickle=False)
    return run_mh(model, output, sources=sources, **args)


@hydra.main(version_base="1.3", config_path="../../configs", config_name="simulate")
def main(cfg: DictConfig) -> None:
    try:
        simulate(cfg)
    except ConfigError as error:
        raise SystemExit(f"invalid configuration: {error}") from None


if __name__ == "__main__":
    main()
