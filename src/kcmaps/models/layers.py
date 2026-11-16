"""Flax layers of the flow network; every submodule has an explicit name, so the parameter tree does not depend on
the Python class names."""

import math
from typing import Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax import lax

from ..systems.peptide import NUM_ATOM_SUFFIXES, NUM_ATOM_TYPES, NUM_RESIDUE_TYPES

ATTENTION_BACKENDS = ("native", "bf16_cudnn")


class Linear(nn.Module):
    """Dense layer with Xavier-uniform or zero weights, zero bias, HIGHEST matmul precision and float32 parameters.

    Attributes:
        out_features: output width.
        bias: whether a bias is added.
        weight_init: 'xavier_uniform' or 'zeros'; the bias starts at zero.
    """

    out_features: int
    bias: bool = True
    weight_init: str = "xavier_uniform"

    def setup(self):
        if self.weight_init == "xavier_uniform":
            weight_initializer = nn.initializers.xavier_uniform()
        elif self.weight_init == "zeros":
            weight_initializer = nn.initializers.zeros
        else:
            raise ValueError(f"invalid weight_init {self.weight_init!r}")
        self._flax_linear = nn.Dense(
            features=self.out_features, use_bias=self.bias, kernel_init=weight_initializer,
            bias_init=nn.initializers.zeros, precision=lax.Precision.HIGHEST, param_dtype=jnp.float32)

    def __call__(self, x):
        return self._flax_linear(x)


class LayerNorm(nn.Module):
    """LayerNorm without scale or bias."""

    eps: float = 1e-6

    def setup(self):
        self._flax_layernorm = nn.LayerNorm(epsilon=self.eps, use_bias=False, use_scale=False)

    def __call__(self, x):
        return self._flax_layernorm(x)


class RMSNorm(nn.Module):
    """RMS normalization with a float32 gain; the arithmetic promotes with the input dtype."""

    hidden_size: int
    eps: float = 1e-6

    @nn.compact
    def __call__(self, x):
        gain = self.param("scale", nn.initializers.ones, (self.hidden_size,), jnp.float32)
        return x * lax.rsqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + self.eps) * gain


def timestep_embedding(t, dim, omega_max=1.0):
    """Sinusoidal embedding [cos(t w), sin(t w)] of floating ``t``, appended as a last axis of even length ``dim``;
    frequencies are geometric from ``omega_max`` down to ``omega_max / 10000``."""
    t = jnp.asarray(t)
    half = dim // 2
    freqs = omega_max * jnp.exp(-math.log(10000) * jnp.arange(start=0, stop=half, dtype=t.dtype) / half)
    args = t[..., None] * freqs
    return jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)


def _split_heads(x, num_heads):
    batch, tokens, channels = x.shape
    return x.reshape(batch, tokens, num_heads, channels // num_heads).transpose(0, 2, 1, 3)


def _merge_heads(x):
    batch, heads, tokens, width = x.shape
    return x.transpose(0, 2, 1, 3).reshape(batch, tokens, heads * width)


def _normed_attention(q, k, v, num_heads, backend):
    """Split heads, RMS-normalize queries and keys per head (``q_norm``, ``k_norm``), attend and merge heads."""
    q, k, v = (_split_heads(value, num_heads) for value in (q, k, v))
    q = RMSNorm(q.shape[-1], name="q_norm")(q)
    k = RMSNorm(k.shape[-1], name="k_norm")(k)
    return _merge_heads(_attend(q, k, v, backend=backend))


def _attend(q, k, v, *, backend="native"):
    """Unmasked scaled dot-product attention on (B, H, T, D); 'bf16_cudnn' runs only the core in bfloat16."""
    dh = q.shape[-1]
    if backend == "bf16_cudnn":
        if any(value.dtype != jnp.float32 for value in (q, k, v)):
            raise TypeError("bf16_cudnn attention requires float32 Q/K/V inputs")
        inputs = [value.transpose(0, 2, 1, 3).astype(jnp.bfloat16) for value in (q, k, v)]
        output = jax.nn.dot_product_attention(*inputs, scale=1.0 / math.sqrt(dh), is_causal=False,
                                              implementation="cudnn")
        return output.transpose(0, 2, 1, 3).astype(jnp.float32)
    logits = jnp.einsum("bhqd,bhkd->bhqk", q, k, precision=lax.Precision.HIGHEST) / math.sqrt(dh)
    weights = nn.softmax(logits, axis=-1)
    return jnp.einsum("bhqk,bhkd->bhqd", weights, v, precision=lax.Precision.HIGHEST)


class MultiHeadSelfAttention(nn.Module):
    channels: int
    num_heads: int
    attention_backend: str = "native"

    @nn.compact
    def __call__(self, x):
        channels = self.channels
        q, k, v = jnp.split(Linear(3 * channels, weight_init="xavier_uniform", name="qkv")(x), 3, axis=-1)
        out = _normed_attention(q, k, v, self.num_heads, self.attention_backend)
        return Linear(channels, weight_init="xavier_uniform", name="out")(out)


class _AdaLNParameters(nn.Module):
    """Zero-initialized kernel and bias of one AdaLN projection."""

    channels: int

    @nn.compact
    def __call__(self):
        kernel = self.param("kernel", nn.initializers.zeros, (self.channels, self.channels), jnp.float32)
        bias = self.param("bias", nn.initializers.zeros, (self.channels,), jnp.float32)
        return kernel, bias


class _AdaLNProjection(nn.Module):
    """Holds one AdaLN kernel and bias under the same parameter path as Linear."""

    channels: int

    @nn.compact
    def __call__(self):
        return _AdaLNParameters(self.channels, name="_flax_linear")()


def project_conditioning(cond, kernel, bias):
    """HIGHEST-precision affine projection; promotes like Dense (float64 inputs give float64 arithmetic)."""
    cond, kernel, bias = nn.dtypes.promote_dtype(cond, kernel, bias)
    return jnp.matmul(cond, kernel, precision=lax.Precision.HIGHEST) + bias


def fused_adaln_projections(cond, sublayers):
    """Every sublayer's AdaLN projection of the shared ``cond`` as one matmul, split per sublayer.

    Args:
        cond: conditioning (B, N, C).
        sublayers: modules with an ``adaln_projection()`` method returning (kernel (C, 3C), bias (3C,)).

    Returns:
        A list with one (B, N, 3C) projection per sublayer.
    """
    kernels, biases = zip(*(layer.adaln_projection() for layer in sublayers))
    projected = project_conditioning(cond, jnp.concatenate(kernels, axis=-1), jnp.concatenate(biases, axis=-1))
    return jnp.split(projected, len(sublayers), axis=-1)


class AdaLNZero(nn.Module):
    """AdaLN-Zero modulation: ``(LN(x) (1 + tanh(W_gamma c)) + W_beta c, W_gate c)``, all projections zero-init.

    ``projected`` is this layer's slice of ``fused_adaln_projections``.
    """

    channels: int

    def setup(self):
        self.norm = LayerNorm(name="norm")
        self.projections = [_AdaLNProjection(self.channels, name=name) for name in ("gamma", "beta", "gate")]

    def projection(self):
        parameters = [projection() for projection in self.projections]
        return (jnp.concatenate([p[0] for p in parameters], axis=-1),
                jnp.concatenate([p[1] for p in parameters], axis=-1))

    def __call__(self, x, projected):
        raw_gamma, beta, gate = jnp.split(projected, 3, axis=-1)
        gamma = 1.0 + jnp.tanh(raw_gamma)
        return self.norm(x) * gamma + beta, gate


class SelfAttnBlock(nn.Module):
    """Residual ``h + gate * attention(AdaLN(h))``."""

    channels: int
    num_heads: int
    attention_backend: str = "native"

    def setup(self):
        self.adaln = AdaLNZero(self.channels, name="adaln")
        self.attn = MultiHeadSelfAttention(self.channels, self.num_heads, attention_backend=self.attention_backend,
                                           name="attn")

    def adaln_projection(self):
        return self.adaln.projection()

    def __call__(self, h, projected):
        mod, gate = self.adaln(h, projected)
        return h + gate * self.attn(mod)


class SwiGLUTransition(nn.Module):
    """``W_out(silu(W_g x) * W_v x)`` without biases."""

    channels: int
    hidden: int

    @nn.compact
    def __call__(self, x):
        value, gate = jnp.split(
            Linear(2 * self.hidden, bias=False, weight_init="xavier_uniform", name="swish_linear")(x), 2, axis=-1)
        return Linear(self.channels, bias=False, weight_init="xavier_uniform", name="linear_out")(nn.silu(gate) * value)


class TransitionBlock(nn.Module):
    """Residual ``h + gate * SwiGLU(AdaLN(h))``."""

    channels: int
    hidden: int

    def setup(self):
        self.adaln = AdaLNZero(self.channels, name="adaln")
        self.transition = SwiGLUTransition(self.channels, self.hidden, name="transition")

    def adaln_projection(self):
        return self.adaln.projection()

    def __call__(self, h, projected):
        mod, gate = self.adaln(h, projected)
        return h + gate * self.transition(mod)


class StructuralEmbed(nn.Module):
    """Per-atom chemical identity: MLP of [atom embedding, residue embedding, residue-index and chain-length
    sinusoids, atom-suffix embedding], RMS-normalized; returns (batch_size, N, C)."""

    channels: int
    atom_type: Tuple[int, ...]
    aa_type: Tuple[int, ...]
    aa_pos: Tuple[int, ...]
    seq_len: int
    atom_suffix: Tuple[int, ...]

    @nn.compact
    def __call__(self, batch_size: int):
        channels = self.channels
        n_atoms = len(self.atom_type)
        atom_type = jnp.asarray(self.atom_type, dtype=jnp.int32)
        aa_type = jnp.asarray(self.aa_type, dtype=jnp.int32)
        aa_pos = jnp.asarray(self.aa_pos, dtype=jnp.float32)
        seq_len = jnp.full((n_atoms,), float(self.seq_len), dtype=jnp.float32)
        atom_emb = nn.Embed(NUM_ATOM_TYPES, channels, param_dtype=jnp.float32, name="atom_embed")(atom_type)
        res_emb = nn.Embed(NUM_RESIDUE_TYPES, channels, param_dtype=jnp.float32, name="aa_embed")(aa_type)
        pos_emb = timestep_embedding(aa_pos, channels)
        seqlen_emb = timestep_embedding(seq_len, channels)
        suffix_emb = nn.Embed(NUM_ATOM_SUFFIXES, channels, param_dtype=jnp.float32, name="suffix_embed")(
            jnp.asarray(self.atom_suffix, jnp.int32))
        features = [atom_emb, res_emb, pos_emb, seqlen_emb, suffix_emb]
        cat = jnp.concatenate(features, axis=-1)
        h = Linear(channels, weight_init="xavier_uniform", name="mlp_fc1")(cat)
        h = nn.gelu(h)
        struct = Linear(channels, weight_init="xavier_uniform", name="mlp_fc2")(h)
        struct = RMSNorm(channels, name="output_norm")(struct)
        return jnp.broadcast_to(struct[None], (batch_size, n_atoms, channels))


class GlobalContextAttention(nn.Module):
    """Cross-attention of per-atom queries over all chemical tokens (B, N, C) -> (B, N, C)."""

    channels: int
    num_heads: int
    attention_backend: str = "native"

    @nn.compact
    def __call__(self, queries, context):
        channels = self.channels
        q = Linear(channels, name="q_proj")(queries)
        k, v = jnp.split(Linear(2 * channels, name="kv_proj")(context), 2, axis=-1)
        value = _normed_attention(q, k, v, self.num_heads, self.attention_backend)
        return Linear(channels, name="out")(value)


def vector_features(vector):
    """(..., 3) -> (..., 4): the vector radially compressed by asinh(|v|)/|v|, and log1p(|v|)."""
    norm = jnp.sqrt(jnp.sum(jnp.square(vector), axis=-1, keepdims=True) + 1e-8)
    return jnp.concatenate([vector * (jnp.arcsinh(norm) / norm), jnp.log1p(norm)], axis=-1)


class SourceMixer(nn.Module):
    """Softmax-weighted mix of four branches: source x, source p, mean x / (a tau) and mean p / tau (dimensionless).

    Attributes:
        channels: output width.
        a: per-atom a_i = T_std p_std,i / (m_i x_std).
        time_floor: lower bound of tau in the rate branches.
    """

    channels: int
    a: Tuple[float, ...]
    time_floor: float

    @nn.compact
    def __call__(self, source, mean_delta, tau):
        a = jnp.asarray(self.a, source.dtype).reshape(1, -1, 1)
        t = jnp.maximum(tau, self.time_floor)[:, None, None]
        vectors = (source[..., :3], source[..., 3:], mean_delta[..., :3] / (a * t), mean_delta[..., 3:] / t)
        hidden = []
        for name, vector in zip(("source_x", "source_p", "mean_x", "mean_p"), vectors):
            h = Linear(self.channels, name=f"{name}_in")(vector_features(vector))
            h = Linear(self.channels, name=f"{name}_out")(nn.silu(h))
            h = RMSNorm(self.channels, name=f"{name}_norm")(h)
            hidden.append(h)
        logits = Linear(4, weight_init="zeros", name="mix_logits")(jnp.concatenate(hidden, axis=-1))
        weights = nn.softmax(logits, axis=-1)
        # Equal weights at initialization; the factor 2 keeps the variance of four unit branches.
        return 2.0 * jnp.sum(jnp.stack(hidden, axis=-2) * weights[..., None], axis=-2)
