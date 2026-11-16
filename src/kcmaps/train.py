"""Training entry point: ``python -m kcmaps.train experiment=al2 [overrides]``."""

from __future__ import annotations

import os

# GPU autotuning off fixes the kernel choice; added to XLA_FLAGS (unless it sets the level) before the first JAX
# backend initialization.
if "xla_gpu_autotune_level" not in os.environ.get("XLA_FLAGS", ""):
    os.environ["XLA_FLAGS"] = f"{os.environ.get('XLA_FLAGS', '')} --xla_gpu_autotune_level=0".strip()

from pathlib import Path  # noqa: E402

import hydra  # noqa: E402

from .core.config import ConfigError, validate  # noqa: E402
from .training.loop import run  # noqa: E402
from .core.utils import directory_lock  # noqa: E402


def train(cfg) -> Path:
    """Validate a composed train config and train into ``paths.output_dir`` under its directory lock; returns the run
    directory."""
    cfg = validate(cfg, "train")
    with directory_lock(cfg.paths.output_dir):
        return run(cfg, Path(cfg.paths.output_dir))


@hydra.main(version_base="1.3", config_path="../../configs", config_name="train")
def main(cfg) -> None:
    try:
        train(cfg)
    except ConfigError as error:
        raise SystemExit(f"invalid configuration: {error}") from None


if __name__ == "__main__":
    main()
