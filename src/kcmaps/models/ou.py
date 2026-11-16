"""The Gaussian reference: the underdamped OU transition with a mechanical clock and a frozen force, in dimensionless
units, plus the conditioning context it gives the flow."""

from __future__ import annotations

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp

from ..core import constants as C
from .geometry import PhaseConstraints, without_rotation

SERIES_H = 0.1
# Floor inside the sqrt keeps the norm's derivative finite at zero.
NORM_EPS2 = 1e-12

_F_SERIES = (tuple(((-1.0) ** n) * (2.0 - 2.0 ** (n - 1)) / math.factorial(n) for n in range(3, 14)),
             tuple(range(3, 14)))
_G1_SERIES = (tuple(((-1.0) ** n) / math.factorial(n) for n in range(2, 13)), tuple(range(2, 13)))


class Context(NamedTuple):
    """Per-transition conditioning of the flow: source state, reference mean displacement, standardized covariance
    features, kinematic features, OU marginal RMS widths and lag."""

    source: jax.Array
    mean_delta: jax.Array
    covariance_features: jax.Array
    kinematic_features: jax.Array
    marginal_rms: jax.Array
    tau: jax.Array


def _series(h, series):
    coeff, powers = series
    return jnp.sum(jnp.asarray(coeff, dtype=h.dtype) * h[..., None] ** jnp.asarray(powers, dtype=h.dtype), axis=-1)


def _force_coefficients(h_force):
    decay_complement = -jnp.expm1(-h_force)
    g1 = jnp.where(h_force < SERIES_H, _series(h_force, _G1_SERIES), h_force - decay_complement)
    return decay_complement, g1


def kinematic_coefficient(T_std, p_std, masses, x_std):
    """``a_i = T_std p_std,i / (m_i x_std)`` as an ``(N, 1)`` array."""
    return T_std * p_std / (masses.reshape(-1, 1) * x_std)


def friction(gamma_ps_inv, T_std):
    """Dimensionless friction gamma T_std."""
    return jnp.asarray(gamma_ps_inv, dtype=jnp.asarray(T_std).dtype) * T_std


def mechanical_clock(tau, identity, transition):
    """C2 clock s(h): the identity up to ``identity``, then saturating at ``identity + transition``."""
    tau = jnp.asarray(tau)
    identity = jnp.asarray(identity, tau.dtype)
    transition = jnp.asarray(transition, tau.dtype)
    excess = (tau-identity)/transition
    return jnp.where(tau <= identity, tau, identity+transition*jnp.tanh(excess))


def mechanical_clock_ratio(tau, identity, transition):
    """r = s(h)/h, including its removable value r(0) = 1."""
    clock = mechanical_clock(tau, identity, transition)
    return jnp.where(tau <= identity, jnp.ones_like(tau), clock/tau)


def midpoint_force_point(source, mechanical_lag, a, *, grad=False):
    """Ballistic midpoint x + a p s/2 of the frozen force; differentiable in the source only with ``grad``."""
    if not grad:
        source = jax.lax.stop_gradient(source)
    return source[..., :3] + a * source[..., 3:] * (mechanical_lag / 2.0)


def force_response(tau, force, a, gamma):
    """Frozen-force translation (position, momentum) over the lag ``tau``."""
    decay_complement, g1 = _force_coefficients(gamma * tau)
    position = a * force * g1 / gamma**2
    momentum = force * decay_complement / gamma
    return jnp.concatenate([position, momentum], axis=-1)


def free_delta(tau, source, a, gamma):
    """Force-free OU mean displacement."""
    decay_complement = -jnp.expm1(-gamma * tau)
    p = source[..., 3:]
    return jnp.concatenate([a * decay_complement * p / gamma, -decay_complement * p], axis=-1)


def covariance(tau, a, gamma):
    """Per-atom OU covariance blocks ``(Sigma_xx, Sigma_xp, Sigma_pp)``."""
    h = gamma * tau
    one_minus_e1 = -jnp.expm1(-h)
    one_minus_e2 = -jnp.expm1(-2.0 * h)
    f_direct = h - 2.0 * one_minus_e1 + 0.5 * one_minus_e2
    f = jnp.where(h < SERIES_H, _series(h, _F_SERIES), f_direct)
    one3 = jnp.ones((3,), dtype=a.dtype)
    sigma_xx = 2.0 * (a / gamma) ** 2 * f * one3
    sigma_xp = (a / gamma) * one_minus_e1**2 * one3
    sigma_pp = jnp.ones_like(a) * one_minus_e2 * one3
    return sigma_xx, sigma_xp, sigma_pp


def cholesky(tau, a, gamma):
    """Lower-triangular per-atom OU Cholesky factor ``(Lxx, Lpx, Lpp)``."""
    sigma_xx, sigma_xp, sigma_pp = covariance(tau, a, gamma)
    determinant = sigma_xx * sigma_pp - sigma_xp**2
    l_xx = jnp.sqrt(sigma_xx)
    return l_xx, sigma_xp / l_xx, jnp.sqrt(determinant / sigma_xx)


def covariance_features(l_xx, l_px, l_pp):
    """Per-atom ``[log sigma_x, log sigma_p, rho_xp]`` of a Cholesky factor."""
    sigma_x = jnp.sqrt(jnp.mean(l_xx * l_xx, axis=-1, keepdims=True))
    sigma_p_axis = jnp.sqrt(l_px * l_px + l_pp * l_pp)
    sigma_p = jnp.sqrt(jnp.mean(sigma_p_axis * sigma_p_axis, axis=-1, keepdims=True))
    rho_xp = jnp.mean(l_px / sigma_p_axis, axis=-1, keepdims=True)
    return jnp.concatenate([jnp.log(sigma_x), jnp.log(sigma_p), rho_xp], axis=-1)


def standardized_covariance_features(features, lo, hi):
    """Map both log widths to [-1, 1] over the training lag range; rho is O(1) already."""
    log_widths = 2.0*(features[..., :2] - lo)/(hi - lo) - 1.0
    return jnp.concatenate([log_widths, features[..., 2:]], axis=-1)


def soft_gaussian(value, *, num, limit):
    """``num`` equal-width Gaussian bumps interior to ``[0, limit]``.

    Saturating, not clipping: a value far above ``limit`` returns a vanishing vector, so an outlier cannot dominate
    the conditioner, while the low end stays represented because the first bump sits one step inside zero.
    """
    edges = jnp.linspace(0.0, limit, num + 2, dtype=value.dtype)
    step = edges[1] - edges[0]
    centers = edges[1:-1]
    return jnp.exp(-(((value[..., None] - centers) / step) ** 2))


def kinematic_features(source, impulse):
    """Per-atom rotation-invariant kinematics: bumps on the speed and on the mean force impulse (thermal-momentum
    units, each with its own range) and their alignment cosine."""
    p = source[..., 3:]
    speed = jnp.sqrt(jnp.sum(p*p, axis=-1) + NORM_EPS2)
    impulse_norm = jnp.sqrt(jnp.sum(impulse*impulse, axis=-1) + NORM_EPS2)
    alignment = jnp.sum(p*impulse, axis=-1) / (speed*impulse_norm)
    return jnp.concatenate([soft_gaussian(speed, num=C.SPEED_BASIS, limit=C.SPEED_LIMIT),
                            soft_gaussian(impulse_norm, num=C.IMPULSE_BASIS, limit=C.IMPULSE_LIMIT),
                            alignment[..., None]], axis=-1)


def marginal_rms(tau, a, gamma):
    """Per-coordinate OU marginal RMS widths ``[sqrt(Qxx), sqrt(Qpp)]``."""
    sigma_xx, _, sigma_pp = covariance(tau, a, gamma)
    return jnp.concatenate([jnp.sqrt(sigma_xx), jnp.sqrt(sigma_pp)], axis=-1)


def forward_delta(tau, eps, a, gamma):
    """Noise displacement ``L eps``."""
    l_xx, l_px, l_pp = cholesky(tau, a, gamma)
    eps_x, eps_p = eps[..., :3], eps[..., 3:]
    return jnp.concatenate([l_xx * eps_x, l_px * eps_x + l_pp * eps_p], axis=-1)


def inverse_delta(tau, delta, a, gamma):
    """``L^-1 delta``."""
    l_xx, l_px, l_pp = cholesky(tau, a, gamma)
    eps_x = delta[..., :3] / l_xx
    eps_p = (delta[..., 3:] - l_px * eps_x) / l_pp
    return jnp.concatenate([eps_x, eps_p], axis=-1)


def pf_velocity(tau, delta, a, gamma):
    """Probability-flow velocity of the centered homogeneous OU."""
    _, _, l_pp = cholesky(tau, a, gamma)
    eps = inverse_delta(tau, delta, a, gamma)
    delta_p = delta[..., 3:]
    # The momentum block of L^-T eps is eps_p / Lpp.
    sigma_inv_delta_p = eps[..., 3:] / l_pp
    return jnp.concatenate([a * delta_p, gamma * (sigma_inv_delta_p - delta_p)], axis=-1)


def logdet_com_free(tau, a, gamma, p_std):
    """OU square-root volume between the orthonormal center-of-mass-free subspaces; the noise normal is p_std."""
    l_xx, _, l_pp = cholesky(tau, a, gamma)
    normal = p_std.reshape(-1)
    l_x = l_xx[:, 0]
    position = jnp.sum(jnp.log(l_x)) + 0.5*jnp.log(jnp.sum((normal/l_x)**2)) - 0.5*jnp.log(jnp.sum(normal**2))
    # Lpp is atom-independent; use the first atom's value.
    momentum = (normal.size - 1)*jnp.log(l_pp[0, 0])
    return 3.0*(position + momentum)


def torque_free_about_source(force, source, constraints):
    """The force less its rigid rotation about the source lever arm (dimensionless in and out).

    The flow pins Lambda = sum r_s x p' to the reference law. The force is torque-free only about its own evaluation
    point (x + s v / 2), so its impulse would add -(c s / 2) sum v x F to Lambda's mean, where the Langevin law is
    exactly exp(-gamma tau) L_s. Only the three pinned rotational components change.
    """
    p_std = constraints.momentum.functional.reshape(-1, 1)
    masses = (p_std * constraints.momentum.direction.reshape(-1, 1)).reshape(-1)
    lever = source[..., :3] / constraints.position.direction.reshape(-1, 1)
    # p_std * force = T_std * F: the physical impulse direction per unit dimensionless time.
    return without_rotation(p_std * force, lever, masses) / p_std


class GaussianReference:
    """The clocked OU reference of one system: kinematic coefficients ``a`` (N, 1), friction ``gamma``, the clock,
    the context floor and the log-width bounds of the training lag range (all dimensionless).

    It is the frozen-force solution of dX = r P dt, dP = (r F - gamma P) dt + g dW with r = s(h)/h, so its covariance
    is D_r Q_OU(h) D_r, D_r = diag(r I_x, I_p): a Gaussian preconditioner, not the interacting transition. Methods
    take one scalar lag.
    """

    def __init__(self, a, gamma, *, clock_identity, clock_transition, context_floor, tau_min, tau_max):
        self.a, self.gamma = a, gamma
        self.clock_identity, self.clock_transition = clock_identity, clock_transition
        self.context_floor = context_floor

        def features(tau):
            return covariance_features(*cholesky(tau, a*self.clock_ratio(tau), gamma))

        # Widths r sqrt(Qxx) and sqrt(Qpp) grow with the lag while gamma h stays below one.
        lo, hi = features(tau_min), features(tau_max)
        self.log_width_bounds = (jnp.min(lo[..., :2], axis=0), jnp.max(hi[..., :2], axis=0))

    def clock(self, tau):
        return mechanical_clock(tau, self.clock_identity, self.clock_transition)

    def clock_ratio(self, tau):
        return mechanical_clock_ratio(tau, self.clock_identity, self.clock_transition)

    def delta(self, tau, eps):
        """Reference noise displacement L(tau) eps."""
        return forward_delta(tau, eps, self.a * self.clock_ratio(tau), self.gamma)

    def inverse(self, tau, delta):
        return inverse_delta(tau, delta, self.a * self.clock_ratio(tau), self.gamma)

    def logdet(self, tau, p_std):
        """Log-volume of L(tau) on the center-of-mass-free subspaces."""
        return logdet_com_free(tau, self.a * self.clock_ratio(tau), self.gamma, p_std)

    def pf_velocity(self, tau, delta):
        """Probability-flow velocity of the clocked covariance D_r Q_OU(h) D_r, including the r'/r term."""
        ratio, derivative = jax.jvp(self.clock_ratio, (tau,), (jnp.ones_like(tau),))
        velocity = pf_velocity(tau, delta, self.a * ratio, self.gamma)
        return velocity.at[..., :3].add((derivative / ratio) * delta[..., :3])

    def impulse(self, tau, force):
        """Mean momentum impulse of the clocked frozen force."""
        ratio = self.clock_ratio(tau)
        return force_response(tau, force * ratio, self.a * ratio, self.gamma)[..., 3:]

    def context(self, source, tau, force, constraints: PhaseConstraints) -> Context:
        """Context of one transition with the frozen ``force`` (net force removed); its rigid rotation about the source
        lever arm is removed here, so the pinned spin's mean is exactly the Langevin exp(-gamma tau) L_s."""
        force = torque_free_about_source(force, source, constraints)
        ratio = self.clock_ratio(tau)
        context_tau = jnp.maximum(tau, self.context_floor)
        response = force_response(tau, force*ratio, self.a*ratio, self.gamma)
        mean_delta = free_delta(tau, source, self.a*ratio, self.gamma) + response
        # Covariance features: the clocked reference's own D_r Q_OU(h) D_r at the lag floored at the density floor.
        context_ratio = self.clock_ratio(context_tau)
        context_cholesky = cholesky(context_tau, self.a*context_ratio, self.gamma)
        return Context(
            source=source,
            mean_delta=mean_delta,
            covariance_features=standardized_covariance_features(covariance_features(*context_cholesky),
                                                                 *self.log_width_bounds),
            # The momentum block of the response is the mean impulse the force cap bounds (thermal-momentum units).
            kinematic_features=kinematic_features(source, response[..., 3:]),
            # Unclocked widths sqrt(Qxx): the position residual spreads like the transition, not the clocked reference.
            marginal_rms=marginal_rms(tau, self.a, self.gamma),
            tau=tau,
        )
