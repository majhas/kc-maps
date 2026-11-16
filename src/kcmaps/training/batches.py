"""Training inputs: the lag law with its geometric horizon, the per-step input sampler and the lane layout.

All lags are in ps. Every draw is one exact transform of a single uniform variate; nothing is clipped to the horizon.
"""

from __future__ import annotations

import dataclasses
import math

import flax
import jax
import jax.numpy as jnp
import numpy as np

from ..core import constants as C


def log_uniform_probability(*, minimum, maximum, final_maximum, log_uniform_fraction) -> float:
    """Probability of the log-uniform component of ``f LogU[min, final] + (1 - f) U[min, final]`` conditioned on
    ``tau <= maximum``."""
    if maximum == minimum:
        return 1.0
    log_mass = log_uniform_fraction * math.log(maximum / minimum) / math.log(final_maximum / minimum)
    uniform_mass = (1 - log_uniform_fraction) * (maximum - minimum) / (final_maximum - minimum)
    return log_mass / (log_mass + uniform_mass)


@dataclasses.dataclass(frozen=True)
class LagLaw:
    """Lag law of the KC parents (and PF rows) at each step.

    KC parents follow ``f LogU[2 tau_min, tau_max] + (1 - f) U[2 tau_min, tau_max]`` conditioned on
    ``tau <= H(step)``; PF lags are ``LogU[tau_min, min(H(step), pf_tau_max)]``.

    Attributes:
        tau_min_ps, tau_max_ps: lag band (ps).
        horizon_start_ps: H(0) (ps).
        anneal_steps: steps over which H grows geometrically to ``tau_max_ps``.
        log_uniform_fraction: f.
        pf_tau_max_ps: PF lag ceiling (ps), or None without the PF lane.
    """

    tau_min_ps: float
    tau_max_ps: float
    horizon_start_ps: float
    anneal_steps: int
    log_uniform_fraction: float
    pf_tau_max_ps: float | None = None

    @classmethod
    def from_config(cls, run_cfg) -> "LagLaw":
        """The law of a validated run config."""
        trainer, objective = run_cfg.trainer, run_cfg.objective
        return cls(tau_min_ps=float(run_cfg.model.train_tau_min_ps), tau_max_ps=float(run_cfg.model.train_tau_max_ps),
                   horizon_start_ps=float(trainer.horizon_start_ps),
                   anneal_steps=round(trainer.horizon_anneal_fraction * trainer.total_steps),
                   log_uniform_fraction=float(objective.kc.log_uniform_fraction),
                   pf_tau_max_ps=float(objective.pf.tau_max_ps) if objective.aux_loss == "pf" else None)

    def horizon(self, step) -> float:
        """Lag ceiling ``H(step) = start (tau_max / start) ** clip(step / anneal_steps, 0, 1)`` (ps)."""
        if self.anneal_steps <= 0 or step >= self.anneal_steps:
            return float(self.tau_max_ps)
        if step <= 0:
            return float(self.horizon_start_ps)
        return float(self.horizon_start_ps * (self.tau_max_ps / self.horizon_start_ps) ** (step / self.anneal_steps))

    def coefficients(self, step) -> np.ndarray:
        """Float32 vector ``(log_width, p_log, p_uniform, minimum, width, pf_log_width)`` of step ``step``.

        The schedule stays in host float64 until this cast, so one compiled sampler serves every step.
        """
        minimum, maximum = 2.0 * self.tau_min_ps, self.horizon(step)
        p_log = log_uniform_probability(minimum=minimum, maximum=maximum, final_maximum=self.tau_max_ps,
                                        log_uniform_fraction=self.log_uniform_fraction)
        pf_width = (math.log(min(maximum, self.pf_tau_max_ps) / self.tau_min_ps) if self.pf_tau_max_ps is not None
                    else 0.0)
        return np.asarray((math.log(maximum / minimum), p_log, 1.0 - p_log, minimum, maximum - minimum, pf_width),
                          dtype=np.float32)

    @staticmethod
    def sample_kc(key, size, coefficients):
        """(size,) KC parent lags (ps)."""
        log_width, p_log, p_uniform, minimum, width, _ = coefficients
        u = jax.random.uniform(key, (size,))
        # Picks a component with u and reuses the rescaled u inside it; the unselected branch stays finite.
        log = minimum * jnp.exp(jnp.minimum(u / p_log, 1.0) * log_width)
        uniform = minimum + jnp.maximum((u - p_log) / p_uniform, 0.0) * width
        return jnp.where(u < p_log, log, uniform)

    def sample_pf(self, key, size, coefficients):
        """(size,) PF lags (ps)."""
        u = jax.random.uniform(key, (size,))
        return self.tau_min_ps * jnp.exp(u * coefficients[5])


@flax.struct.dataclass
class LaneInputs:
    """Per-row lags ``tau`` (B,) in ps and keys ``keys`` (B, 2) of one lane."""

    tau: jax.Array
    keys: jax.Array


@flax.struct.dataclass
class TrainingInputs:
    """Dynamic inputs of one update; every leaf has the batch as its leading axis.

    Attributes:
        positions: (B, N, 3) centered positions (nm).
        kc: KC lane inputs (DB rows reuse them at their own rows).
        pf: PF lane inputs, or None.
    """

    positions: jax.Array
    kc: LaneInputs
    pf: LaneInputs | None = None


def lane_counts(batch_size: int) -> tuple[int, int]:
    """(KC rows, second-lane rows) of a per-device batch split into blocks of ``LANE_BLOCK`` rows."""
    blocks = batch_size // C.LANE_BLOCK
    return blocks * (C.LANE_BLOCK - C.SECOND_LANE_ROWS), blocks * C.SECOND_LANE_ROWS


def split_lanes(inputs: TrainingInputs, lane: str):
    """``((kc positions, kc inputs), (second positions, second inputs))`` of one device's shard.

    The first ``LANE_BLOCK - SECOND_LANE_ROWS`` rows of every block are KC rows, the rest second-lane rows, so lane
    membership does not depend on how block-aligned shards partition the batch. ``lane`` is ``'pf'`` or ``'kc'``
    (DB rows reuse the KC draws).
    """
    second = getattr(inputs, lane)
    size = inputs.positions.shape[0]
    n_kc = C.LANE_BLOCK - C.SECOND_LANE_ROWS

    def rows(tree, first, count):
        def select(array):
            blocks = array.reshape((size // C.LANE_BLOCK, C.LANE_BLOCK) + array.shape[1:])
            return blocks[:, first:first + count].reshape((-1,) + array.shape[1:])
        return jax.tree.map(select, tree)

    return (rows((inputs.positions, inputs.kc), 0, n_kc),
            rows((inputs.positions, second), n_kc, C.SECOND_LANE_ROWS))


def make_input_sampler(law: LagLaw, dataset, batch_size: int):
    """A sampler ``(key, step) -> (next key, TrainingInputs)`` of ``batch_size`` rows.

    The key splits six ways: slot 0 is the next key, 1 the frames, 2/3 the PF lags/keys and 4/5 the KC
    lags/keys.
    """
    batch_size = int(batch_size)
    pf_lane = law.pf_tau_max_ps is not None

    @jax.jit
    def compiled(key, coefficients):
        next_key, k_x, k_tpf, k_kpf, k_tkc, k_kkc = jax.random.split(key, 6)
        positions = dataset.sample_batch(k_x, batch_size)
        kc = LaneInputs(tau=law.sample_kc(k_tkc, batch_size, coefficients),
                        keys=jax.random.split(k_kkc, batch_size))
        pf = None
        if pf_lane:
            pf = LaneInputs(tau=law.sample_pf(k_tpf, batch_size, coefficients),
                            keys=jax.random.split(k_kpf, batch_size))
        return next_key, TrainingInputs(positions=positions, kc=kc, pf=pf)

    def sample_inputs(key, step: int):
        return compiled(key, law.coefficients(step))

    return sample_inputs
