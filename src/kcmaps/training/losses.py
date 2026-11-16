"""Training objectives: the KC loss L_KC against the two-step target at the EMA target parameters, the
detailed-balance loss L_DB and the probability-flow loss L_PF, and ``make_loss``, which assembles them over one
shard."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ..core import constants as C
from ..models.geometry import flip_momentum
from . import batches

sg = jax.lax.stop_gradient
# Lag bands of the logged loss means (ps): short < 1 fs <= mid < 10 fs <= long.
LAG_BAND_EDGES_PS = (1e-3, 1e-2)
LAG_BANDS = ("short", "mid", "long")


def ramp_weight(step, *, weight: float, start_step: int, ramp_steps: int):
    """Float32 weight: zero before ``start_step``, linear over ``ramp_steps``, then ``weight``."""
    step = jnp.asarray(step, jnp.float32)
    if ramp_steps == 0:
        ramp = (step >= start_step).astype(jnp.float32)
    else:
        ramp = jnp.clip((step - start_step) / ramp_steps, 0.0, 1.0)
    return jnp.asarray(weight, jnp.float32) * ramp


def flow_nll(eps, logdet, d_free):
    """Per-degree-of-freedom NLL without constants and the reference log-det (the trained quantity)."""
    return (0.5 * jnp.sum(eps**2) - logdet) / d_free


def nll(flow_value, reference_inverse_logdet, d_free):
    """Per-degree-of-freedom NLL of the dimensionless displacement."""
    return flow_value + 0.5 * jnp.log(2.0 * jnp.pi) - reference_inverse_logdet / d_free


def hot_momentum_scale(kcm, key, hot, dtype=jnp.float32):
    """Per-row Maxwell width in units of the bath's: with probability ``hot['fraction']`` sqrt(T'/T) with
    T' = max(N(mean_K, std_K), floor_K), else 1."""
    augment_key, temperature_key = jax.random.split(key)
    kelvin = float(hot["mean_K"]) + float(hot["std_K"]) * jax.random.normal(temperature_key, (), dtype)
    kelvin = jnp.maximum(kelvin, float(hot["floor_K"]))
    scale = jnp.sqrt(kelvin / jnp.asarray(float(kcm.system.temperature_K), dtype))
    augmented = jax.random.uniform(augment_key, (), dtype) < float(hot["fraction"])
    return jnp.where(augmented, scale, jnp.ones_like(scale))


def sample_eps(kcm, key, shape, dtype=jnp.float32):
    """Standard normal noise projected onto the center-of-mass-free noise subspace."""
    return kcm.split_noise(jax.random.normal(key, shape, dtype=dtype))[0]


def source(kcm, positions, key, momentum_scale=1.):
    """Dimensionless center-of-mass-free state: ``positions`` (N, 3) nm with Maxwell momenta from ``key`` at
    ``momentum_scale`` times the bath width."""
    x = jnp.asarray(positions, jnp.float32) / kcm.transform.x_std_nm
    return kcm.project(jnp.concatenate([x, momentum_scale * jax.random.normal(key, x.shape, dtype=jnp.float32)],
                                       axis=-1))


def _hot_source(kcm, positions, tau_ps, key, hot):
    """``(z, tau, key)`` of a KC or PF row: the source with hot-augmented Maxwell momenta, the dimensionless lag and
    the row's remaining key."""
    momentum_key, row_key = jax.random.split(key)
    scale_key, draw_key = jax.random.split(momentum_key)
    z = source(kcm, positions, draw_key, hot_momentum_scale(kcm, scale_key, hot))
    return z, kcm.transform.to_dimensionless_time(tau_ps).astype(jnp.float32), row_key


def split_lag(tau, key, tau_min):
    """The sub-lags (2,) ``(r - s, t - r)`` of a lag ``tau = t - s``: a Beta(2, 2) split restricted so both are
    >= ``tau_min``."""
    tau = jnp.asarray(tau, jnp.float32)
    half = jnp.asarray(0.5, tau.dtype)
    r_min = jnp.minimum(half, jnp.asarray(tau_min) / tau)
    concentration = jnp.asarray(C.KC_SPLIT_BETA, tau.dtype)
    u = jax.random.beta(jax.random.fold_in(key, C.KC_SPLIT_FOLD_IN), concentration, concentration, shape=(),
                        dtype=tau.dtype)
    r = r_min + (1.0 - 2.0 * r_min) * u
    first_product = r * tau
    second_product = (1.0 - r) * tau
    # The larger sub-lag is formed as a difference so the two sum to ``tau`` exactly.
    first_smaller = r <= half
    first = jnp.where(first_smaller, tau - second_product, first_product)
    second = jnp.where(first_smaller, second_product, tau - first_product)
    return jnp.stack([first, second])


def two_step_target(kcm, target_params, source, tau, key, tau_min):
    """Displacement of the two-step target y ~ q_{r,t}(.|x'; eta), x' ~ q_{s,r}(.|x; eta) at the target
    parameters eta, with ``r - s, t - r`` from ``split_lag``."""
    first_key, second_key = jax.random.split(key)
    first_lag, second_lag = split_lag(tau, key, tau_min)
    eps = sample_eps(kcm, first_key, source.shape, dtype=source.dtype)
    ctx = kcm.prepare_context(source, first_lag)
    displacement, _ = kcm.forward(target_params, ctx, kcm.reference.delta(ctx.tau, eps))
    cumulative = kcm.project_displacement(displacement, source)

    current = source + cumulative
    eps = sample_eps(kcm, second_key, current.shape, dtype=current.dtype)
    ctx = kcm.prepare_context(current, second_lag)
    displacement, _ = kcm.forward(target_params, ctx, kcm.reference.delta(ctx.tau, eps))
    return kcm.project_displacement(cumulative + displacement, source)


def kc_loss(kcm, params, target_params, positions, tau_ps, key, *, tau_min, cap, hot):
    """KC loss of one row: the capped flow NLL of the two-step target's displacement under the one-step model.

    Rows whose angular-momentum map is near-singular (``kcm.angular_momentum_map_valid`` False) get zero loss.

    Returns:
        (loss, nll, capped, valid): the loss, the uncapped per-degree-of-freedom NLL of the physical displacement
        (zero on masked rows), the cap flag (float, zero on masked rows) and the not-masked flag (bool).
    """
    z, tau, composition_key = _hot_source(kcm, positions, tau_ps, key, hot)
    target = sg(two_step_target(kcm, target_params, z, tau, composition_key, tau_min))
    # A masked row sees the zero displacement (T = identity) so no inf reaches the masked gradient; non-finite
    # geometry is not masked and still propagates NaN.
    valid = kcm.angular_momentum_map_valid(z, target)
    target = jnp.where(valid, target, jnp.zeros_like(target))
    ctx = kcm.prepare_context(z, tau)
    eps, logdet = kcm.inverse(params, ctx, target)
    flow_value = flow_nll(eps, logdet, kcm.d_free)
    if cap > 0.0:
        capped = flow_value > cap
        loss, capped = jnp.where(capped, cap, flow_value), capped.astype(flow_value.dtype)
    else:
        loss, capped = flow_value, jnp.zeros_like(flow_value)
    loss = jnp.where(valid, loss, 0.0)
    reference_inverse_logdet = -kcm.reference.logdet(ctx.tau, kcm.transform.p_std)
    value = jnp.where(valid, nll(flow_value, reference_inverse_logdet, kcm.d_free) + kcm.logdet_to_phys / kcm.d_free,
                      0.0)
    return loss, value, jnp.where(valid, capped, 0.0), valid


# -- detailed balance -----------------------------------------------------------------------------------------------


def reverse_displacement(displacement):
    """flip(z) - flip(z + displacement), formed from the displacement rather than from rounded endpoints."""
    return jnp.concatenate([-displacement[..., :3], displacement[..., 3:]], axis=-1)


def reduced_hamiltonian(kcm, state):
    """H/kBT of one dimensionless state: U/kBT + |p|^2 / 2."""
    potential = kcm.system.energy((kcm.transform.x_std_nm * state[..., :3])[None])[0]
    return potential + 0.5 * jnp.sum(state[..., 3:] ** 2)


def _displacement(kcm, params, source, tau, eps):
    """(center-of-mass-free displacement, raw displacement) of one model draw."""
    ctx = kcm.prepare_context(source, tau)
    displacement, _ = kcm.forward(params, ctx, kcm.reference.delta(ctx.tau, eps))
    return kcm.project_displacement(displacement, source), displacement


def _log_ratio(kcm, params, source, displacement, tau):
    """``(r~(z, z + displacement), reverse latent)`` through the exact inverse in both directions."""

    def log_weight(start, step):
        ctx = kcm.prepare_context(start, tau, force_point_grad=True)
        eps, logdet = kcm.inverse(params, ctx, step)
        return -0.5 * jnp.sum(eps ** 2) + logdet, eps

    reverse, reverse_eps = log_weight(flip_momentum(source + displacement), reverse_displacement(displacement))
    forward, _ = log_weight(source, displacement)
    energy = reduced_hamiltonian(kcm, source + displacement) - reduced_hamiltonian(kcm, source)
    return -energy + reverse - forward, reverse_eps


def db_loss(kcm, params, positions, tau_ps, key):
    """DB loss of one row.

    With z a center-of-mass-free frame with bath-temperature Maxwell momenta, z' = z + displacement(z, eps), R the
    momentum flip and H = U/kBT + |p|^2 / 2, the generalized-MH log ratio
    r = -[H(z') - H(z)] + log q(Rz | Rz') - log q(z' | z) vanishes on every row exactly in detailed balance, and
    L_DB = KL(q_{s,t}(y|x) pi(x) || q_{s,t}(Rx|Ry) pi(y)) = -E[r]. The value is -r / d_free; its parameter
    gradient is the k2 pathwise estimate
    -[c_fwd . d displacement(z, eps) + c_rev . d displacement(Rz', eps_rev)] / d_free, with c_fwd = d r~/dz' at
    fixed z, c_rev = -R d r~/dz at fixed z' and r~ evaluated with stopped parameters and the frozen-force point
    differentiable in both inverse passes.

    Returns:
        (value, r, valid): the value with the k2 gradient, the log ratio r (zero on masked rows) and the not-masked
        flag (bool). Rows whose forward or reverse angular-momentum map is near-singular get zero value and gradient.
    """
    momentum_key, eps_key = jax.random.split(key)
    z = source(kcm, positions, momentum_key)
    tau = kcm.transform.to_dimensionless_time(tau_ps).astype(jnp.float32)
    eps = sample_eps(kcm, eps_key, z.shape, dtype=z.dtype)
    displacement, raw = _displacement(kcm, params, z, tau, eps)
    displacement_no_grad = sg(displacement)
    valid = (kcm.angular_momentum_map_valid(z, sg(raw))
             & kcm.angular_momentum_map_valid(flip_momentum(z + displacement_no_grad),
                                              reverse_displacement(displacement_no_grad)))
    displacement_no_grad = jnp.where(valid, displacement_no_grad, jnp.zeros_like(displacement_no_grad))
    (r, reverse_eps), (d_source, d_displacement) = jax.value_and_grad(
        lambda s, d: _log_ratio(kcm, sg(params), s, d, tau), argnums=(0, 1), has_aux=True)(z,
                                                                                           displacement_no_grad)
    value = jnp.where(valid, -r, 0.0) / kcm.d_free
    c_fwd = jnp.where(valid, d_displacement, 0.0)
    c_rev = jnp.where(valid, -flip_momentum(d_source - d_displacement), 0.0)
    reverse, _ = _displacement(kcm, params, flip_momentum(z + displacement_no_grad), tau, sg(reverse_eps))
    surrogate = -(jnp.sum(sg(c_fwd) * displacement) + jnp.sum(sg(c_rev) * reverse)) / kcm.d_free
    value = value + (surrogate - sg(surrogate))
    return value, jnp.where(valid, r, 0.0), valid


# -- probability flow -----------------------------------------------------------------------------------------------


def _score(kcm, params, ctx, displacement):
    """Gradient of the model log-density in a displacement from ``source`` (terms constant in it dropped)."""
    def log_prob(value):
        eps, logdet = kcm.inverse(params, ctx, value)
        return -0.5 * jnp.sum(eps**2) + logdet

    return jax.grad(log_prob)(displacement)


def _pf_velocity(kcm, endpoint, score):
    """Langevin probability-flow velocity at an endpoint with the uncapped endpoint force."""
    x, p = endpoint[..., :3], endpoint[..., 3:]
    force = kcm.force(x)
    return kcm.project(jnp.concatenate([kcm.reference.a * p, force - kcm.reference.gamma * (p + score[..., 3:])],
                                       axis=-1))


def _displacement_jvp(kcm, params, source, tau, eps):
    """Mapped displacement at lag ``tau``, its ``tau``-scaled tangent along the reference probability flow, and the
    context."""
    reference_delta = kcm.reference.delta(tau, eps)
    reference_velocity = kcm.reference.pf_velocity(tau, reference_delta)

    def mapped(time, delta):
        ctx = kcm.prepare_context(source, time)
        return kcm.forward(params, ctx, delta)[0], ctx

    (displacement, ctx), (scaled_tangent, _) = jax.jvp(mapped, (tau, reference_delta),
                                                       (tau, tau * reference_velocity))
    return displacement, scaled_tangent, ctx


def _adanorm(mse):
    return mse / (C.PF_ADANORM_EPS + sg(jnp.sqrt(mse)))


def pf_loss(kcm, params, positions, tau_ps, key, *, hot):
    """Probability-flow loss L_PF of one row: AdaNorm of ``tau (d/dtau) delta - tau v(z + delta)`` with gradient only
    through the sampled displacement; the velocity v uses the stopped endpoint, stopped parameters and the uncapped
    endpoint force.

    Returns:
        (loss, valid): the loss and the not-masked flag (bool); masked rows have zero loss.
    """
    z, tau, eps_key = _hot_source(kcm, positions, tau_ps, key, hot)
    eps = sample_eps(kcm, eps_key, z.shape, dtype=z.dtype)
    delta, scaled_tangent, ctx = _displacement_jvp(kcm, params, z, tau, eps)
    delta_no_grad = sg(delta)
    score = _score(kcm, jax.tree.map(sg, params), jax.tree.map(sg, ctx), delta_no_grad)
    velocity = sg(_pf_velocity(kcm, z + delta_no_grad, score))
    valid = kcm.angular_momentum_map_valid(z, delta_no_grad)
    error = jnp.where(valid, kcm.project((delta - sg(delta)) + sg(scaled_tangent - tau * velocity)), 0.0)
    x_mse = jnp.mean(error[..., :3] ** 2)
    p_mse = jnp.mean(error[..., 3:] ** 2)
    x_loss, p_loss = _adanorm(x_mse), _adanorm(p_mse)
    loss = 0.5 * (x_loss + p_loss)
    return loss, valid


# -- assembly -------------------------------------------------------------------------------------------------------


def _mean(values, mask=None):
    """(sum, count) float32 pair of ``values`` over the rows selected by ``mask`` (all rows when None)."""
    values = jnp.asarray(values, jnp.float32)
    if mask is None:
        return jnp.stack([jnp.sum(values), jnp.asarray(values.size, jnp.float32)])
    return jnp.stack([jnp.sum(jnp.where(mask, values, 0.0)), jnp.sum(mask.astype(jnp.float32))])


def _lag_band_means(name, values, valid, tau):
    short, long = tau < LAG_BAND_EDGES_PS[0], tau >= LAG_BAND_EDGES_PS[1]
    masks = (short, ~short & ~long, long)
    return {f"{name}_{band}": _mean(values, mask & valid) for band, mask in zip(LAG_BANDS, masks)}


def scalar_metric(value):
    """(value, 1) float32 pair of a scalar that is the same on every row."""
    return jnp.stack([jnp.asarray(value, jnp.float32), jnp.ones((), jnp.float32)])


def make_loss(kcm, run_cfg, *, db_active: bool):
    """The objective ``loss_fn(params, target_params, step, inputs) -> (loss, metrics)`` of one shard.

    Per-row losses work in dimensionless coordinates and take one physical lag (ps); they are vmapped over the shard.
    Lane rows are fixed inside every block of the shard, so each device holds equal lane counts; each metric is a
    ``(sum, count)`` pair that the train and eval steps ``psum`` over devices into global means.

    ``objective.aux_loss`` selects KC only (null), KC + DB (``db``) or KC + PF (``pf``). With ``db`` and
    ``db_active`` False the loss is plain KC on every row (the updates before ``objective.db.start_step``); with
    ``db_active`` True the KC and DB lanes share ``params`` and one backward pass differentiates the total.

    ``metrics`` maps names to float32 ``(sum, count)`` pairs (mean = sum / count, exact after the cross-device
    ``psum``): ``loss``, ``kc_loss``, ``kc_loss_{short,mid,long}``, ``kc_nll`` (uncapped physical NLL),
    ``kc_frac_capped``, with ``aux_loss`` set ``aux_weight``, and with an active lane ``db_loss``,
    ``db_loss_{short,mid,long}``, ``db_acceptance`` (E[min(1, e^r)]) or ``pf_loss``,
    ``pf_loss_{short,mid,long}``.
    Lag bands and ``kc_nll``/``db_acceptance`` average the rows whose angular-momentum map is valid; the other means
    average every row of the lane.
    """
    objective = run_cfg.objective
    aux_loss = objective.aux_loss
    tau_min = float(run_cfg.model.train_tau_min_ps) / float(run_cfg.model.T_std_ps)
    cap = float(objective.kc.cap)
    kc_weight = float(objective.kc.weight)
    hot = {key: float(objective.kc.hot_source[key]) for key in ("fraction", "mean_K", "std_K", "floor_K")}

    kc_terms = jax.vmap(lambda p, e, x, t, k: kc_loss(kcm, p, e, x, t, k, tau_min=tau_min, cap=cap, hot=hot),
                        in_axes=(None, None, 0, 0, 0))
    if aux_loss == "pf":
        pf = objective.pf
        pf_weight, pf_start, pf_ramp = float(pf.weight), int(pf.start_step), int(pf.ramp_steps)
        pf_terms = jax.vmap(lambda p, x, t, k: pf_loss(kcm, p, x, t, k, hot=hot), in_axes=(None, 0, 0, 0))
    if db_active:
        db = objective.db
        db_weight, db_start, db_ramp = float(db.weight), int(db.start_step), int(db.ramp_steps)
        db_terms = jax.vmap(lambda p, x, t, k: db_loss(kcm, p, x, t, k), in_axes=(None, 0, 0, 0))

    def loss_fn(params, target_params, step, inputs):
        if db_active:
            (kc_positions, kc_lane), (db_positions, db_lane) = batches.split_lanes(inputs, "kc")
        elif aux_loss == "pf":
            (kc_positions, kc_lane), (pf_positions, pf_lane) = batches.split_lanes(inputs, "pf")
        else:
            kc_positions, kc_lane = inputs.positions, inputs.kc
        loss, nll_value, capped, valid = kc_terms(params, target_params, kc_positions, kc_lane.tau, kc_lane.keys)
        total = kc_weight * jnp.mean(loss).astype(jnp.float32)
        metrics = {"kc_loss": _mean(loss), **_lag_band_means("kc_loss", loss, valid, kc_lane.tau),
                   "kc_nll": _mean(nll_value, valid), "kc_frac_capped": _mean(capped)}
        if aux_loss == "db" and not db_active:
            metrics["aux_weight"] = scalar_metric(0.0)
        if aux_loss == "pf":
            values, valid = pf_terms(params, pf_positions, pf_lane.tau, pf_lane.keys)
            weight = ramp_weight(step, weight=pf_weight, start_step=pf_start, ramp_steps=pf_ramp)
            total = total + weight * jnp.mean(values).astype(jnp.float32)
            metrics.update({"aux_weight": scalar_metric(weight), "pf_loss": _mean(values),
                            **_lag_band_means("pf_loss", values, valid, pf_lane.tau)})
        if db_active:
            weight = ramp_weight(step, weight=db_weight, start_step=db_start, ramp_steps=db_ramp)
            values, r, valid = db_terms(params, db_positions, db_lane.tau, db_lane.keys)
            total = total + weight * jnp.mean(values).astype(jnp.float32)
            metrics.update({"aux_weight": scalar_metric(weight), "db_loss": _mean(values),
                            **_lag_band_means("db_loss", values, valid, db_lane.tau),
                            "db_acceptance": _mean(jnp.exp(jnp.minimum(r, 0.0)), valid)})
        metrics["loss"] = scalar_metric(total)
        return total, metrics

    return loss_fn
