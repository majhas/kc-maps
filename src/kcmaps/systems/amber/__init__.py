"""Pure-JAX Amber energy and score with GBSA-OBC implicit solvent. The force field comes from the system config
(AL2/AL4: amber99sbildn + OBC2; AL6: amber14 + OBC1).
"""

from .energy import build_amber_backend

__all__ = ["build_amber_backend"]
