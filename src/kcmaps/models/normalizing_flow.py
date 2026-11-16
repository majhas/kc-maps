"""NormalizingFlow: the conditional coupling flow on dimensionless residuals of the reference displacement."""

from typing import NamedTuple, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.scipy.special import logsumexp

from ..core import constants as C
from ..systems.peptide import Encodings
from . import ou
from .geometry import Scales
from .layers import (GlobalContextAttention, LayerNorm, Linear, RMSNorm, SelfAttnBlock, SourceMixer, StructuralEmbed,
                     SwiGLUTransition, TransitionBlock, fused_adaln_projections, timestep_embedding)


def rational_gate(chi, order):
    """G_k(chi) = chi^k / (1 + chi^k) for k = 2 or 3."""
    power = chi * chi if order == 2 else chi * chi * chi
    return power / (1.0 + power)


def saturated_position_unit(a, flight, x_std_nm):
    """Dimensionless position-shift unit F / sqrt(1 + (F x_std / L)^2), F = a * flight, L in nm.

    ``flight`` = u_gamma(tau) / T_std with u_gamma = (1 - exp(-gamma tau)) / gamma; shapes broadcast.
    """
    unit = a * flight
    excess = unit * x_std_nm / C.POSITION_UNIT_LENGTH_NM
    return unit / jnp.sqrt(1.0 + excess * excess)


def delivered_log_scale(raw_alpha, g2, n_per_partition):
    """Momentum log-scale: the shared budget divided over the momentum blocks, outside the tanh."""
    return jnp.asarray(g2, raw_alpha.dtype) * (C.MOMENTUM_LOG_SCALE_BUDGET / n_per_partition) * jnp.tanh(raw_alpha)


def spin_features(displacement, ctx, constraints):
    """(B, 6) view of the pinned spin L' = sum r_s x p' of the endpoint, in the frame of the inputs.

    asinh(L' / sqrt(kBT Ibar)) and asinh((L' - a L_s) / sqrt(v kBT Ibar)), Ibar = trace I(x_s) / 3,
    a = exp(-gamma tau), v = 1 - a^2 (the dimensionless OU momentum variance). Momentum steps never change L', so
    every block stays triangular.
    """
    dtype = displacement.dtype
    p_std = jnp.asarray(constraints.momentum.functional, dtype).reshape(1, -1, 1)
    mass = p_std * jnp.asarray(constraints.momentum.direction, dtype).reshape(1, -1, 1)
    kbt = p_std[0, 0, 0]**2 / mass[0, 0, 0]  # p_std^2 = m kBT for every atom
    x = ctx.source[..., :3] / jnp.asarray(constraints.position.direction, dtype).reshape(1, -1, 1)
    r = x - jnp.sum(mass * x, axis=1, keepdims=True) / jnp.sum(mass)
    ibar = (2.0 / 3.0) * jnp.sum(mass[..., 0] * jnp.sum(r * r, axis=-1), axis=-1)
    source_spin = jnp.sum(jnp.cross(r, p_std * ctx.source[..., 3:]), axis=1)
    endpoint_spin = jnp.sum(jnp.cross(r, p_std * (ctx.mean_delta[..., 3:] + displacement[..., 3:])), axis=1)
    v = ctx.marginal_rms[:, 0, 3]**2
    decay = v / (1.0 + jnp.sqrt(1.0 - v))  # 1 - a without cancellation
    scale = jnp.sqrt(kbt * ibar)[:, None]
    noise_scale = jnp.sqrt(v)[:, None] * scale
    return jnp.concatenate([jnp.arcsinh((source_spin + endpoint_spin) / scale),
                            jnp.arcsinh((decay[:, None] * source_spin + endpoint_spin) / noise_scale)], axis=-1)


def constrained_affine_step(part, ell, shift, constraint, reverse=False):
    """One constrained diagonal affine map d' = d + P[(exp(ell) - 1) d + shift] and its restricted log-det.

    Args:
        part, ell, shift: (B, N, 3).
        constraint: ``geometry.LinearConstraint`` (one value per atom).
        reverse: apply the exact analytic inverse instead.

    Returns:
        (mapped part, log-det), the log-det forward-signed forward and inverse-signed in reverse.
    """
    a, b = constraint.normalized_functional(), constraint.direction
    d_inv = jnp.exp(-ell)
    s = jnp.sum(a * b * d_inv, axis=-2, keepdims=True)
    block_logdet = jnp.sum(ell, axis=(1, 2)) + jnp.sum(logsumexp(jnp.log(a * b) - ell, axis=-2), axis=-1)
    if reverse:
        centered = part - constraint.project(shift)
        diagonal_solve = d_inv * centered
        numerator = jnp.sum(a * diagonal_solve, axis=-2, keepdims=True)
        solved = diagonal_solve - (d_inv * b) * (numerator / s)
        return solved, -block_logdet
    update = jnp.expm1(ell) * part + shift
    return part + constraint.project(update), block_logdet


class RigidBasis(NamedTuple):
    """The six rigid momentum columns ``A`` (B,6,N,3), ``A^T A`` and its log-det."""

    basis: jax.Array
    gram: jax.Array
    logdet_gram: jax.Array


def rigid_momentum_basis(functional, positions) -> RigidBasis:
    """Unit-normalized total-momentum and angular-momentum functionals.

    Args:
        functional: per-atom momentum center-of-mass weight (proportional to p_std).
        positions: center-of-mass-free physical source positions (B, N, 3).
    """
    a = jnp.asarray(functional, positions.dtype).reshape(1, -1, 1)
    x, y, z = positions[..., 0], positions[..., 1], positions[..., 2]
    zero = jnp.zeros_like(x)
    rotations = jnp.stack([
        jnp.stack([zero, -z, y], -1),
        jnp.stack([z, zero, -x], -1),
        jnp.stack([-y, x, zero], -1),
    ], axis=1) * a[:, None]
    com = jnp.broadcast_to(jnp.eye(3, dtype=positions.dtype)[None, :, None, :] * a[:, None], rotations.shape)
    basis = jnp.concatenate([com, rotations], axis=1)
    basis = basis / jnp.sqrt(jnp.sum(basis * basis, axis=(2, 3), keepdims=True))
    gram = jnp.einsum("bkni,bmni->bkm", basis, basis, precision=lax.Precision.HIGHEST)
    return RigidBasis(basis, gram, jnp.linalg.slogdet(gram)[1])


def _dot(basis, v):
    return jnp.einsum("bkni,bni->bk", basis, v, precision=lax.Precision.HIGHEST)


def _combine(coef, basis):
    return jnp.einsum("bk,bkni->bni", coef, basis, precision=lax.Precision.HIGHEST)


def _gram(basis, weight):
    return jnp.einsum("bkni,bni,bmni->bkm", basis, weight, basis, precision=lax.Precision.HIGHEST)


def rigid_constrained_affine_step(part, ell, shift, rigid, reverse=False):
    """Momentum affine step that keeps total and angular momentum: ``constrained_affine_step`` with P removing the
    six columns A of ``rigid`` orthogonally.

    With E = exp(ell): log det = sum ell + log det(A^T E^-1 A) - log det(A^T A); the reverse map is
    d = E^-1 c + E^-1 A (A^T E^-1 A)^-1 A^T (c - E^-1 c), c = d' - P shift.
    """
    basis, gram = rigid.basis, rigid.gram
    d_inv = jnp.exp(-ell)
    weighted = _gram(basis, d_inv)
    block_logdet = jnp.sum(ell, axis=(1, 2)) + jnp.linalg.slogdet(weighted)[1] - rigid.logdet_gram

    def project(v):
        return v - _combine(jnp.linalg.solve(gram, _dot(basis, v)[..., None])[..., 0], basis)

    if reverse:
        centered = part - project(shift)
        diagonal_solve = d_inv * centered
        coef = jnp.linalg.solve(weighted, _dot(basis, centered - diagonal_solve)[..., None])[..., 0]
        return diagonal_solve + d_inv * _combine(coef, basis), -block_logdet
    update = jnp.expm1(ell) * part + shift
    return part + project(update), block_logdet


class _CouplingBlock(nn.Module):
    """Attention conditioner of one block: inactive lane + context -> (raw shift, raw log-scale), each (B, N, 3)."""

    channels: int
    layers_per_block: int
    num_heads: int
    ffn_hidden: int
    attention_backend: str = "native"

    def setup(self):
        self.proj_in = Linear(self.channels, weight_init="xavier_uniform", name="proj_in")
        sublayers = []
        for layer in range(self.layers_per_block):
            sublayers.append(SelfAttnBlock(self.channels, self.num_heads, attention_backend=self.attention_backend,
                                           name=f"sa_{layer}"))
            sublayers.append(TransitionBlock(self.channels, self.ffn_hidden, name=f"ffn_{layer}"))
        self.sublayers = sublayers
        self.proj_out = Linear(6, weight_init="zeros", name="proj_out")

    def __call__(self, inactive_state, context, projections):
        hidden = self.proj_in(inactive_state) + context
        for sublayer, projected in zip(self.sublayers, projections):
            hidden = sublayer(hidden, projected)
        output = self.proj_out(hidden)
        return output[..., 0:3], output[..., 3:6]


class NormalizingFlow(nn.Module):
    """Coupling flow on dimensionless residuals of the reference displacement.

    Blocks alternate between the position lane (even blocks) and the momentum lane (odd blocks). Each block is the
    constrained affine map d' = d + P[(exp(ell) - 1) d + shift], where P removes the center-of-mass direction
    (positions) or the six rigid momentum directions (momenta), so the map leaves the total and the angular momentum
    about the source lever arm at their reference law. Heads are bounded and gated on the lag:

        position: ell = 1.25 G3(r h) tanh(alpha),          shift = G3(r h) u(h) 4 tanh(beta / 4)
        momentum: ell = (12 / (K/2)) G2(r h) tanh(alpha),  shift = G3(r h) 32 tanh(beta / 32)

    with G_k(chi) = chi^k / (1 + chi^k), r = 1/(3 fs), K the number of blocks, and u(h) the per-atom free-flight unit
    a_i u_gamma(h) / T_std saturated at ``constants.POSITION_UNIT_LENGTH_NM``.

    Raises:
        ValueError: (at setup) per-atom fields of unequal length.

    Attributes:
        encodings: structural encodings of the peptide (``systems.Encodings``).
        scales: units (``geometry.Scales``): T_std_ps (ps), x_std_nm (nm), per-atom p_std (amu nm/ps).
        masses_amu: per-atom masses (amu).
        gamma_ps_inv: friction (1/ps).
        channels, num_heads, num_blocks, layers_per_block, source_layers, ffn_hidden, conditioner_hidden: capacity.
        time_embedding_max_freq_per_ps: highest angular frequency (1/ps) of the sinusoidal lag embedding; the lag
            features are cos(tau w_k), sin(tau w_k) with w_k geometric from this value down to 1/10000 of it.
        attention_backend: 'native' or the inference-only 'bf16_cudnn'.
    """

    encodings: Encodings
    scales: Scales
    masses_amu: Tuple[float, ...]
    gamma_ps_inv: float
    channels: int
    num_heads: int
    num_blocks: int
    layers_per_block: int
    source_layers: int
    ffn_hidden: int
    conditioner_hidden: int
    time_embedding_max_freq_per_ps: float
    attention_backend: str = "native"

    def setup(self):
        enc = self.encodings
        n_atoms = len(enc.atom_type)
        if any(len(values) != n_atoms for values in (enc.atom_suffix, enc.aa_type, enc.aa_pos, self.scales.p_std,
                                                       self.masses_amu)):
            raise ValueError("per-atom encodings, p_std and masses_amu must have one entry per atom")
        self.gamma, self.a = self._kinetic_constants()
        T_std_ps = self.scales.T_std_ps
        self.omega_max = self.time_embedding_max_freq_per_ps * T_std_ps
        self.position_gate_rate = C.POSITION_GATE_RATE_PS_INV * T_std_ps
        self.momentum_log_scale_rate = C.MOMENTUM_LOG_SCALE_RATE_PS_INV * T_std_ps
        self.momentum_shift_rate = C.MOMENTUM_SHIFT_RATE_PS_INV * T_std_ps
        self.source_mixer = SourceMixer(self.channels, self.a, C.SOURCE_MIXING_TIME_FLOOR_PS / T_std_ps,
                                        name="source_mixer")
        self.cond_in = Linear(self.channels, bias=False, weight_init="xavier_uniform", name="cond_in")
        self.cond_mlp = [SwiGLUTransition(self.channels, self.conditioner_hidden, name=f"cond_mlp_{i}")
                         for i in range(2)]
        self.cond_mlp_norm = [LayerNorm(name=f"cond_mlp_norm_{i}") for i in range(2)]
        self.cond_output_norm = RMSNorm(self.channels, name="cond_output_norm")
        self.context_input_norm = RMSNorm(self.channels, name="context_input_norm")
        self.chemical_attention = GlobalContextAttention(self.channels, self.num_heads,
                                                         attention_backend=self.attention_backend,
                                                         name="chemical_attention")
        self.structural = StructuralEmbed(self.channels, enc.atom_type, enc.aa_type, enc.aa_pos, enc.seq_len,
                                          enc.atom_suffix, name="structural")
        self.source_blocks = [SelfAttnBlock(self.channels, self.num_heads, attention_backend=self.attention_backend,
                                            name=f"source_sa_{layer}") for layer in range(self.source_layers)]
        self.source_ffn = [TransitionBlock(self.channels, self.ffn_hidden, name=f"source_ffn_{layer}")
                           for layer in range(self.source_layers)]
        self.coupling_blocks = [
            _CouplingBlock(channels=self.channels, layers_per_block=self.layers_per_block, num_heads=self.num_heads,
                           ffn_hidden=self.ffn_hidden, attention_backend=self.attention_backend,
                           name=f"block_{block}")
            for block in range(self.num_blocks)]

    def _kinetic_constants(self):
        """(gamma T_std, per-atom a_i = T_std p_std,i / (m_i x_std)) evaluated eagerly in float32, as Python floats."""
        # Eager even inside a jit trace, so the constants are the same float32 values in every trace and under x64.
        with jax.ensure_compile_time_eval():
            T_std = jnp.asarray(self.scales.T_std_ps, jnp.float32)
            x_std = jnp.asarray(self.scales.x_std_nm, jnp.float32)
            p_std = jnp.asarray(self.scales.p_std, jnp.float32).reshape(-1, 1)
            masses = jnp.asarray(np.asarray(self.masses_amu, np.float64), jnp.float32).reshape(-1, 1)
            a = ou.kinematic_coefficient(T_std, p_std, masses, x_std)
            gamma = ou.friction(self.gamma_ps_inv, T_std)
            return float(gamma), tuple(float(value) for value in np.asarray(a).reshape(-1))

    def conditioner(self, tau, covariance_features, kinematic_features, spin):
        """Per-atom conditioning (B, N, C) from the lag sinusoid, width and kinematic features and the spin."""
        tau = timestep_embedding(tau, C.TAU_FEATURES, omega_max=self.omega_max)
        tau = jnp.broadcast_to(tau[:, None, :], covariance_features.shape[:-1] + tau.shape[-1:])
        spin = jnp.broadcast_to(spin[:, None, :], covariance_features.shape[:-1] + spin.shape[-1:])
        hidden = self.cond_in(jnp.concatenate([tau, covariance_features, kinematic_features, spin], axis=-1))
        for norm, layer in zip(self.cond_mlp_norm, self.cond_mlp):
            hidden = hidden + layer(norm(hidden))
        return self.cond_output_norm(hidden)

    def _source_sublayers(self):
        sublayers = []
        for attention, transition in zip(self.source_blocks, self.source_ffn):
            sublayers += [attention, transition]
        return sublayers

    def _adaln_projections(self, condition):
        """Every AdaLN projection of one map call as one matmul: (source-layer list, one list per block)."""
        groups = [self._source_sublayers()] + [block.sublayers for block in self.coupling_blocks]
        flat = iter(fused_adaln_projections(condition, [layer for group in groups for layer in group]))
        projections = [[next(flat) for _ in group] for group in groups]
        return projections[0], projections[1:]

    def _context(self, ctx, projections):
        chemical = self.structural(ctx.source.shape[0])
        context = self.source_mixer(ctx.source, ctx.mean_delta, ctx.tau) + chemical
        context = context + self.chemical_attention(context, chemical)
        context = self.context_input_norm(context)
        for sublayer, projected in zip(self._source_sublayers(), projections):
            context = sublayer(context, projected)
        return context

    def _conditioner_input(self, state, offset, marginal_rms):
        """The inactive lane in units of the OU marginal widths (the updated lane zeroed)."""
        inactive = state.at[..., offset:offset + 3].set(0.0)
        return inactive / marginal_rms

    def __call__(self, displacement, ctx, constraints, reverse=False):
        """Map residuals (B, N, 6) given a batched ``ou.Context`` and the ``PhaseConstraints``; returns (mapped,
        log-det (B,)), the inverse and its inverse log-det with ``reverse``."""
        spin = spin_features(displacement, ctx, constraints)
        condition = self.conditioner(ctx.tau, ctx.covariance_features, ctx.kinematic_features, spin)
        source_projections, block_projections = self._adaln_projections(condition)
        context = self._context(ctx, source_projections)
        dtype = displacement.dtype
        t = ctx.tau[:, None, None]
        g_x = rational_gate(self.position_gate_rate * t, C.POSITION_GATE_ORDER)
        g_p = rational_gate(self.momentum_log_scale_rate * t, C.MOMENTUM_LOG_SCALE_ORDER)
        envelope_p = rational_gate(self.momentum_shift_rate * t, C.MOMENTUM_SHIFT_ORDER)
        flight = -jnp.expm1(-self.gamma * t) / self.gamma
        unit_x = saturated_position_unit(jnp.asarray(self.a, dtype).reshape(1, -1, 1), flight, self.scales.x_std_nm)
        rigid = rigid_momentum_basis(constraints.momentum.functional,
                                     ctx.source[..., :3] / constraints.position.direction)
        state = displacement
        logdet = jnp.zeros((displacement.shape[0],), displacement.dtype)
        order = reversed(range(self.num_blocks)) if reverse else range(self.num_blocks)
        for block_index in order:
            offset = 0 if block_index % 2 == 0 else 3
            inactive = self._conditioner_input(state, offset, ctx.marginal_rms)
            beta, raw_alpha = self.coupling_blocks[block_index](inactive, context, block_projections[block_index])
            part = state[..., offset:offset + 3]
            if offset == 0:
                bound = C.POSITION_SHIFT_BOUND
                log_scale = g_x * C.POSITION_LOG_SCALE_PREFACTOR * jnp.tanh(raw_alpha)
                shift = g_x * unit_x * (bound * jnp.tanh(beta / bound))
                transformed, block_logdet = constrained_affine_step(part, log_scale, shift, constraints.position,
                                                                    reverse=reverse)
            else:
                bounded = C.MOMENTUM_SHIFT_BOUND * jnp.tanh(beta / C.MOMENTUM_SHIFT_BOUND)
                log_scale = delivered_log_scale(raw_alpha, g_p, self.num_blocks // 2)
                shift = envelope_p * bounded
                transformed, block_logdet = rigid_constrained_affine_step(part, log_scale, shift, rigid,
                                                                          reverse=reverse)
            state = state.at[..., offset:offset + 3].set(transformed)
            logdet = logdet + block_logdet
        return state, logdet
