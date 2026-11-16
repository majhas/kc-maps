"""KCM: the one-step transition model z' = z + displacement, from the Gaussian reference, the flow, the
angular-momentum map and the exact center-of-mass law."""

from __future__ import annotations

import copy
import math

import jax
import jax.numpy as jnp
import numpy as np

from ..core import constants as C
from . import geometry as geo
from . import ou
from .geometry import Scales
from .layers import ATTENTION_BACKENDS
from .normalizing_flow import NormalizingFlow


def build_kcm(model_cfg, system, scales: Scales, *, attention: str = "native") -> "KCM":
    """Construct the KCM (no force cap) of a ``model`` config section for one ``PeptideSystem`` and its scales.

    Raises:
        ValueError: ``attention`` is not 'native' or the inference-only 'bf16_cudnn'.
    """
    if attention not in ATTENTION_BACKENDS:
        raise ValueError(f"attention backend must be one of {ATTENTION_BACKENDS}, got {attention!r}")
    net = NormalizingFlow(
        encodings=system.encodings, scales=scales,
        masses_amu=tuple(float(v) for v in np.asarray(system.masses, np.float64)),
        gamma_ps_inv=float(system.gamma_ps_inv),
        channels=int(model_cfg["channels"]), num_heads=int(model_cfg["num_heads"]),
        num_blocks=int(model_cfg["num_blocks"]), layers_per_block=int(model_cfg["layers_per_block"]),
        source_layers=int(model_cfg["source_layers"]), ffn_hidden=int(model_cfg["ffn_hidden"]),
        conditioner_hidden=int(model_cfg["conditioner_hidden"]),
        time_embedding_max_freq_per_ps=float(model_cfg["time_embedding_max_freq_per_ps"]), attention_backend=attention)
    return KCM(net, system, scales, train_tau_min_ps=float(model_cfg["train_tau_min_ps"]),
               train_tau_max_ps=float(model_cfg["train_tau_max_ps"]))


def _init(kcm: "KCM", key, ctx: ou.Context) -> dict:
    return kcm.net.init(key, jnp.zeros_like(ctx.source)[None], jax.tree.map(lambda v: v[None], ctx), kcm.constraints)


def init_params(kcm: "KCM", positions_nm, key) -> dict:
    """Fresh float32 parameters ``{'params': ...}``; ``key`` seeds the initializers.

    ``positions_nm`` (N, 3) are the centered positions of one training frame; the context at
    sqrt(train_tau_min_ps train_tau_max_ps) with zero momenta is the initialization input.

    Raises:
        ValueError: the context is not finite.
    """
    positions = jnp.asarray(positions_nm, jnp.float32)
    source = kcm.transform.to_dimensionless(jnp.concatenate([positions, jnp.zeros_like(positions)], axis=-1))
    tau = kcm.transform.to_dimensionless_time(
        jnp.asarray(math.sqrt(kcm.train_tau_min_ps * kcm.train_tau_max_ps), jnp.float32))
    ctx = kcm.prepare_context(source, tau)
    if not all(np.isfinite(np.asarray(value)).all() for value in jax.tree.leaves(ctx)):
        raise ValueError("initialization requires a finite context")
    _, ctx = kcm._rotate_to_body_frame(ctx)
    return _init(kcm, key, ctx)


def params_template(kcm: "KCM") -> dict:
    """Shapes and dtypes (``jax.ShapeDtypeStruct``) of the parameter tree, without computing it."""
    n = kcm.masses.shape[0]
    f32 = jnp.float32

    def init():
        zeros = jnp.zeros((n, 6), f32)
        ctx = ou.Context(source=zeros, mean_delta=zeros, covariance_features=jnp.zeros((n, 3), f32),
                         kinematic_features=jnp.zeros((n, C.KINEMATIC_CHANNELS), f32), marginal_rms=zeros,
                         tau=jnp.ones((), f32))
        return _init(kcm, jax.random.PRNGKey(0), ctx)

    return jax.eval_shape(init)


class KCM:
    """The KC map of one system: Gaussian reference + ``NormalizingFlow`` + angular-momentum map + exact
    center-of-mass law (+ optional force cap).

    Physical states are (N, 6) (positions nm, momenta amu nm/ps) and lags ps (``_phys``/``_ps`` names); every other
    method works in dimensionless, center-of-mass-free coordinates with one scalar lag (vmap for batches).

    Attributes:
        net: the ``NormalizingFlow``. system: the ``PeptideSystem``.
        transform: physical <-> dimensionless map. constraints: center-of-mass constraints (dimensionless).
        reference: the ``ou.GaussianReference``.
        masses: (N, 1) float32 amu. kbt_kj_per_mol, gamma_ps_inv: bath.
        d_free: 6N - 6. force_cap: per-atom impulse cap (None; set with ``with_force_cap``).
    """

    def __init__(self, net: NormalizingFlow, system, scales: Scales, *, train_tau_min_ps: float,
                 train_tau_max_ps: float):
        masses = np.asarray(system.masses, np.float64).reshape(-1)
        p_std = np.asarray(scales.p_std, np.float64)
        self.net = net
        self.system = system
        self.train_tau_min_ps = float(train_tau_min_ps)
        self.train_tau_max_ps = float(train_tau_max_ps)
        self.force_cap = None
        self.anchors = tuple(system.anchors)
        self.gamma_ps_inv = float(system.gamma_ps_inv)
        self.kbt_kj_per_mol = float(system.kbt_kj_per_mol)
        self.masses = jnp.asarray(masses[:, None], jnp.float32)
        self.transform = geo.CoordinateTransform(
            x_std_nm=jnp.asarray(float(scales.x_std_nm), jnp.float32),
            p_std=jnp.asarray(list(p_std), jnp.float32).reshape(-1, 1),
            T_std_ps=jnp.asarray(float(scales.T_std_ps), jnp.float32))
        self.constraints = geo.make_constraints(self.masses, self.transform.x_std_nm, self.transform.p_std)
        a = ou.kinematic_coefficient(self.transform.T_std_ps, self.transform.p_std, self.masses,
                                     self.transform.x_std_nm)
        T_std = float(scales.T_std_ps)
        self.reference = ou.GaussianReference(
            a, ou.friction(self.gamma_ps_inv, self.transform.T_std_ps),
            clock_identity=C.CLOCK_IDENTITY_PS / T_std, clock_transition=C.CLOCK_TRANSITION_PS / T_std,
            context_floor=self.transform.to_dimensionless_time(jnp.asarray(C.CONTEXT_LOG_WIDTH_FLOOR_PS, jnp.float32)),
            tau_min=self.transform.to_dimensionless_time(jnp.float32(self.train_tau_min_ps)),
            tau_max=self.transform.to_dimensionless_time(jnp.float32(self.train_tau_max_ps)))
        self.logdet_to_phys = self.transform.restricted_logdet_to_phys()
        self.d_free = 6 * masses.size - 6

    def with_force_cap(self, cap: float | None) -> "KCM":
        """A copy of this model with the per-atom impulse cap ``cap`` (None removes it); parameters are shared."""
        if cap is not None and not (math.isfinite(cap) and cap > 0):
            raise ValueError("force_cap must be finite and positive")
        other = copy.copy(self)
        other.force_cap = None if cap is None else float(cap)
        return other

    # -- center of mass ---------------------------------------------------------------

    def project(self, z):
        """Dimensionless (..., 6) state or displacement less its center-of-mass components."""
        z = jnp.asarray(z)
        return jnp.concatenate([self.constraints.position.project(z[..., :3]),
                                self.constraints.momentum.project(z[..., 3:])], axis=-1)

    def project_displacement(self, displacement, source):
        """Displacement whose endpoint from ``source`` is center-of-mass free."""
        offset = jnp.concatenate([self.constraints.position.offset(source[..., :3]),
                                  self.constraints.momentum.offset(source[..., 3:])], axis=-1)
        return offset + self.project(displacement)

    def center_positions(self, x_phys):
        """Physical positions (..., N, 3) less their mass-weighted center, which the potential does not see."""
        return self.transform.x_std_nm * self.constraints.position.project(x_phys / self.transform.x_std_nm)

    def recenter_state(self, z_phys):
        """(..., N, 6) physical state with positions moved back to a zero center of mass (momenta unchanged); the
        potential is translation invariant, so the samplers pin the position center at zero after each step."""
        return jnp.concatenate([self.center_positions(z_phys[..., :3]), z_phys[..., 3:]], axis=-1)

    def _internal_state(self, z_phys):
        """Physical state less its center-of-mass position and total momentum: the coordinates the internal map
        acts on."""
        return self.transform.to_phys(self.project(self.transform.to_dimensionless(z_phys)))

    def maxwell_momenta_phys(self, key, shape):
        """Maxwell-Boltzmann momenta (amu nm/ps) at the bath temperature, shape (..., N, 3)."""
        return self.transform.p_std * jax.random.normal(key, shape, dtype=self.masses.dtype)

    def initial_state(self, positions_nm, key):
        """(N, 6) float32 chain start: center-of-mass-free ``positions_nm`` (N, 3) and Maxwell momenta from ``key``."""
        position = self.center_positions(jnp.asarray(positions_nm, jnp.float32))
        momentum = self.maxwell_momenta_phys(key, (self.masses.shape[0], 3)).astype(jnp.float32)
        return jnp.concatenate([position, momentum], axis=-1)

    # -- context ----------------------------------------------------------------------

    def force(self, x):
        """Dimensionless force with the uniform net force removed, at dimensionless positions (N, 3)."""
        force = self.kbt_kj_per_mol * self.system.score((self.transform.x_std_nm * x)[None])[0]
        force = force - jnp.mean(force, axis=-2, keepdims=True)
        return self.transform.to_dimensionless_force(force)

    def _cap_force(self, force, tau):
        """Scale each atom's force so its dimensionless mean impulse is at most ``force_cap``; net force removed."""
        impulse = self.reference.impulse(tau, force)
        cap = jnp.asarray(self.force_cap, force.dtype)
        factor = jnp.minimum(cap / jnp.maximum(jnp.linalg.norm(impulse, axis=-1), 1e-20), 1.)[..., None]
        force = self.transform.to_phys_force(force * factor)
        force = force - jnp.mean(force, axis=-2, keepdims=True)
        return self.transform.to_dimensionless_force(force)

    def prepare_context(self, source, tau, *, force_point_grad=False) -> ou.Context:
        """Context of a transition from ``source`` over ``tau``, with the frozen force at the clocked ballistic
        midpoint; ``force_point_grad`` makes that point differentiable in the source (values unchanged).

        This is the model's one lag check: the scalar float32 lag must be finite and positive, else the context's lag
        (``ctx.tau``, which callers reuse) is NaN under tracing and a ValueError is raised eagerly.
        """
        tau = jnp.asarray(tau, jnp.float32)
        if tau.ndim:
            raise ValueError("one scalar lag is required; vmap for batches")
        valid = jnp.isfinite(tau) & (tau > 0)
        if not isinstance(valid, jax.core.Tracer) and not bool(valid):
            raise ValueError("lag must be finite and positive")
        tau = jnp.where(valid, tau, jnp.nan)
        force_point = ou.midpoint_force_point(source, self.reference.clock(tau), self.reference.a,
                                              grad=force_point_grad)
        force = self.force(force_point)
        if self.force_cap is not None:
            force = self._cap_force(force, tau)
        return self.reference.context(source, tau, force, self.constraints)

    # -- flow and angular-momentum map --------------------------------------------------

    def _rotate_to_body_frame(self, ctx):
        """(frame axes, ctx with source and mean displacement rotated into the N-CA-C body frame of the source)."""
        axes = geo.frame_from_source(ctx.source, self.transform.x_std_nm, self.anchors)
        return axes, ctx._replace(source=geo.phase_to_body(ctx.source, axes),
                                  mean_delta=geo.phase_to_body(ctx.mean_delta, axes))

    def _flow(self, params, delta, ctx, reverse):
        axes, ctx = self._rotate_to_body_frame(ctx)
        residual = geo.phase_to_body(delta, axes)
        batched = jax.tree.map(lambda v: v[None], ctx._replace(tau=jnp.asarray(ctx.tau, jnp.float32)))
        mapped, logdet = self.net.apply(params, residual[None], batched, self.constraints, reverse=reverse)
        return delta + geo.phase_to_lab(mapped[0] - residual, axes), logdet[0]

    def _angular_momentum_map(self, source, displacement, inverse=False):
        """Apply T (or T^-1) to a displacement; returns it and log|det T|."""
        source = self.transform.to_phys(source)
        delta = self.transform.to_phys(displacement)
        correction, log_det = geo.angular_momentum_map(
            source[..., :3], delta[..., :3], source[..., 3:] + delta[..., 3:], self.masses.reshape(-1), inverse=inverse)
        correction = self.transform.to_dimensionless(jnp.concatenate([jnp.zeros_like(correction), correction], axis=-1))
        return displacement + correction, log_det

    def angular_momentum_map_valid(self, source, displacement):
        """Whether det T of a displacement is at least ``ANGULAR_MOMENTUM_MAP_MIN_DET_RATIO`` (NaN is valid)."""
        source = self.transform.to_phys(source)
        delta = self.transform.to_phys(displacement)
        det = geo.angular_momentum_map_det(source[..., :3], delta[..., :3], self.masses.reshape(-1))
        return ~(det < C.ANGULAR_MOMENTUM_MAP_MIN_DET_RATIO)

    def forward(self, params, ctx, reference_delta):
        """(displacement, log-det of flow and map) of a reference noise displacement."""
        mapped, logdet = self._flow(params, reference_delta, ctx, reverse=False)
        displacement = ctx.mean_delta + mapped
        mapped_displacement, map_logdet = self._angular_momentum_map(ctx.source, displacement)
        return mapped_displacement, logdet + map_logdet

    def inverse(self, params, ctx, displacement):
        """(eps, inverse log-det of flow and map) of a center-of-mass-free displacement."""
        unmapped, map_logdet = self._angular_momentum_map(ctx.source, displacement, inverse=True)
        reference_delta, flow_logdet = self._flow(params, unmapped - ctx.mean_delta, ctx, reverse=True)
        eps = self.reference.inverse(ctx.tau, reference_delta)
        return eps, flow_logdet - map_logdet

    # -- physical kernel: internal map plus the exact center-of-mass law ------------------------

    def _noise_normal(self):
        """Unit (N, 1) normal of the noise constraint (its direction is p_std)."""
        direction = self.constraints.noise.direction
        return direction / jnp.sqrt(jnp.sum(direction * direction))

    def split_noise(self, eps):
        """(internal noise, center-of-mass noise z (2, 3)) of unprojected (N, 6) noise: the internal part is ``eps``
        projected onto the noise subspace orthogonal to the normal, z its position and momentum normal coefficients."""
        normal = self._noise_normal()
        z = jnp.sum(normal[..., None] * jnp.stack([eps[..., :3], eps[..., 3:]], -2), axis=-3)
        return eps - jnp.concatenate([normal * z[0], normal * z[1]], axis=-1), z

    def _com_units(self):
        total_mass = jnp.sum(self.masses)
        return (total_mass, self.transform.T_std_ps * jnp.sqrt(self.kbt_kj_per_mol / total_mass),
                jnp.sqrt(total_mass * self.kbt_kj_per_mol))

    def _com_law(self, source, tau):
        total_mass, _, p_unit = self._com_units()
        one = jnp.ones((1, 1), jnp.float32)
        momentum = jnp.sum(source[..., 3:], axis=-2, keepdims=True) / p_unit
        mean = ou.free_delta(tau, jnp.concatenate([jnp.zeros_like(momentum), momentum], -1), one,
                             self.reference.gamma)
        return mean, ou.cholesky(tau, one, self.reference.gamma)

    def _com_field(self, d_r, d_p):
        total_mass = jnp.sum(self.masses)
        return jnp.concatenate([jnp.broadcast_to(d_r, (self.masses.shape[0], 3)), self.masses / total_mass * d_p],
                               axis=-1)

    def propose(self, params, z_phys, tau_ps, eps):
        """(physical displacement (N, 6), forward log-det) of one proposal of lag ``tau_ps`` for unprojected standard
        normal ``eps`` (N, 6); the proposal is z_phys + displacement.

        -0.5 |eps|^2 - logdet is the forward log-density up to a constant in the lag only.
        """
        # The map tends to the identity as tau -> 0, but exact zero hits sqrt(0) in the reference Cholesky factor, the
        # 0/0 clock ratio and zero reference widths, so zero lag runs at a valid lag and returns no displacement.
        zero = tau_ps == 0
        tau_ps = jnp.where(zero, 1.0, tau_ps)
        internal_eps, z = self.split_noise(eps)
        source = self.transform.to_dimensionless(self._internal_state(z_phys))
        ctx = self.prepare_context(source, self.transform.to_dimensionless_time(tau_ps))
        displacement, logdet = self.forward(params, ctx, self.reference.delta(ctx.tau, internal_eps))
        displacement = self.transform.to_phys(displacement)
        _, x_unit, p_unit = self._com_units()
        mean, (l_xx, l_px, l_pp) = self._com_law(z_phys, ctx.tau)
        d_r = x_unit * (mean[..., :3] + l_xx * z[0])
        d_p = p_unit * (mean[..., 3:] + l_px * z[0] + l_pp * z[1])
        displacement = self._internal_state(displacement) + self._com_field(d_r, d_p)
        return jnp.where(zero, 0.0, displacement), jnp.where(zero, 0.0, logdet)

    def proposal_inverse(self, params, z_phys, tau_ps, displacement_phys):
        """(unprojected eps, inverse log-det) of a physical displacement from ``z_phys``."""
        total_mass = jnp.sum(self.masses)
        d_r = jnp.sum(self.masses * displacement_phys[..., :3], axis=-2, keepdims=True) / total_mass
        d_p = jnp.sum(displacement_phys[..., 3:], axis=-2, keepdims=True)
        source = self._internal_state(z_phys)
        internal = displacement_phys - self._com_field(d_r, d_p)
        source, tau = self.transform.to_dimensionless(source), self.transform.to_dimensionless_time(tau_ps)
        internal = self.transform.to_dimensionless(internal)
        ctx = self.prepare_context(source, tau)
        eps, logdet = self.inverse(params, ctx, internal)
        _, x_unit, p_unit = self._com_units()
        mean, (l_xx, l_px, l_pp) = self._com_law(z_phys, ctx.tau)
        z_x = (d_r / x_unit - mean[..., :3]) / l_xx
        z_p = (d_p / p_unit - mean[..., 3:] - l_px * z_x) / l_pp
        normal = self._noise_normal()
        return eps + jnp.concatenate([normal * z_x, normal * z_p], axis=-1), logdet
