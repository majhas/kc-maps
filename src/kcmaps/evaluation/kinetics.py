"""Kinetics on the (phi, psi) torus: metastable cores from the reference free energy, core-set milestoning,
mean first-passage times and state-indicator autocorrelations, with bootstrap intervals over chains.

States: F = -ln p on an ``nbins`` x ``nbins`` torus histogram of the reference, smoothed by a periodic
Gaussian of ``sigma`` cells; steepest-descent watershed over 8 neighbors; basins merged while
(saddle - higher minimum) < ``merge_kbt``; basins below ``min_pop`` merged into their lowest-saddle
neighbor; cores are basin cells with F <= F_min + ``core_kbt``. Basins are named by their minimum
(alphaL: phi > 0; alpha: -120 deg < psi < 50 deg; beta: phi < -110 deg; else ppII).

Milestoning: a frame outside every core keeps the last core visited; frames before the first core hit are
unlabeled. An entry into a state is a label change (the first core hit is not an entry).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

STATE_PARAMS = dict(nbins=120, sigma=1.0, merge_kbt=0.5, min_pop=0.005, core_kbt=2.0)
STATE_ORDER = ("alpha", "beta", "beta/ppII", "ppII", "alphaL")
ACF_CUT = 0.05      # tau_int integrates C(t) up to its first value below this
BOOT_BATCH = 100    # bootstrap replicates per ACF batch
_NB8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def torus_cells(phi: np.ndarray, psi: np.ndarray, nbins: int) -> tuple[np.ndarray, np.ndarray]:
    """Grid cell of each angle pair: ``floor((x + pi) / (2 pi) * nbins) mod nbins``."""
    to = lambda a: np.floor((np.asarray(a, np.float64) + np.pi) / (2 * np.pi) * nbins).astype(np.int64) % nbins
    return to(phi), to(psi)


def _saddles(lab: np.ndarray, F: np.ndarray, nb: int) -> np.ndarray:
    """S[a, b] = min over 8-adjacent cell pairs of basins a, b of max(F_a, F_b); inf if not adjacent."""
    S = np.full((nb, nb), np.inf)
    for di, dj in ((0, 1), (1, 0), (1, 1), (1, -1)):
        l2, F2 = np.roll(lab, (-di, -dj), (0, 1)), np.roll(F, (-di, -dj), (0, 1))
        m = (lab >= 0) & (l2 >= 0) & (lab != l2)
        a, b, h = lab[m], l2[m], np.maximum(F[m], F2[m])
        np.minimum.at(S, (a, b), h)
        np.minimum.at(S, (b, a), h)
    return S


def _relabel(lab: np.ndarray, keep: list[int]) -> np.ndarray:
    new = np.full(lab.max() + 2, -1)
    new[np.asarray(keep, int)] = np.arange(len(keep))
    return np.where(lab >= 0, new[np.maximum(lab, 0)], -1)


def segment_torus(phi: np.ndarray, psi: np.ndarray, *, nbins: int, sigma: float, merge_kbt: float, min_pop: float,
                  core_kbt: float) -> dict:
    """Watershed segmentation of F(phi, psi) (see the module docstring).

    Args:
        phi, psi: reference angles (rad), pooled over trajectories.

    Returns:
        dict with ``F`` (nbins, nbins; +inf on empty cells, min 0), ``hist``, ``basin`` and ``core`` label
        grids (-1 = none), and per basin ``F_min``, ``pop``, ``core_pop``, ``min_deg`` (phi, psi of the
        minimum), plus the ``saddle`` matrix.
    """
    from scipy.ndimage import gaussian_filter

    i, j = torus_cells(phi, psi, nbins)
    h = np.zeros((nbins, nbins))
    np.add.at(h, (i, j), 1)
    p = gaussian_filter(h, sigma, mode="wrap")
    p /= p.sum()
    with np.errstate(divide="ignore"):
        F = np.where(h > 0, -np.log(p), np.inf)
    F -= F[np.isfinite(F)].min()
    idx = np.arange(nbins * nbins).reshape(nbins, nbins)
    nF = np.stack([np.roll(F, (-di, -dj), (0, 1)) for di, dj in _NB8])
    nI = np.stack([np.roll(idx, (-di, -dj), (0, 1)) for di, dj in _NB8])
    k = nF.argmin(0)
    best, bidx = np.take_along_axis(nF, k[None], 0)[0], np.take_along_axis(nI, k[None], 0)[0]
    ptr = np.where(np.isfinite(F) & (best < F), bidx, idx).ravel()
    while True:
        nxt = ptr[ptr]
        if np.array_equal(nxt, ptr):
            break
        ptr = nxt
    fin = np.isfinite(F).ravel()
    roots = np.unique(ptr[fin])
    lab = np.full(nbins * nbins, -1)
    lab[fin] = np.searchsorted(roots, ptr[fin])
    lab = lab.reshape(nbins, nbins)
    fmin = lambda lab, nb: np.array([F[lab == b].min() for b in range(nb)])
    pop = lambda lab, nb: np.array([h[lab == b].sum() for b in range(nb)]) / h.sum()
    cen = lambda c: float(np.degrees(-np.pi + (c + .5) * 2 * np.pi / nbins))
    while True:
        nb = lab.max() + 1
        fm = fmin(lab, nb)
        bar = _saddles(lab, F, nb) - np.maximum(fm[:, None], fm[None, :])
        a, b = np.unravel_index(np.argmin(bar), bar.shape)
        if not bar[a, b] < merge_kbt:
            break
        lo, hi = (a, b) if fm[a] <= fm[b] else (b, a)
        lab = _relabel(np.where(lab == hi, lo, lab), [x for x in range(nb) if x != hi])
    while True:
        nb = lab.max() + 1
        pp = pop(lab, nb)
        small = np.flatnonzero(pp < min_pop)
        if len(small) == 0:
            break
        s = small[np.argmin(pp[small])]
        S = _saddles(lab, F, nb)
        tgt = int(np.argmin(S[s])) if np.isfinite(S[s]).any() else -1
        lab = _relabel(np.where(lab == s, tgt, lab), [x for x in range(nb) if x != s])
    nb = lab.max() + 1
    fm, S, pp = fmin(lab, nb), _saddles(lab, F, nb), pop(lab, nb)
    mins = []
    for b in range(nb):
        ci, cj = np.unravel_index(np.argmin(np.where(lab == b, F, np.inf)), F.shape)
        mins.append((cen(ci), cen(cj)))
    core = np.where(np.isfinite(F) & (F <= fm[np.maximum(lab, 0)] + core_kbt) & (lab >= 0), lab, -1)
    core_pop = np.array([h[core == b].sum() for b in range(nb)]) / h.sum()
    return dict(F=F, hist=h, basin=lab, core=core, F_min=fm, saddle=S, pop=pp, core_pop=core_pop,
                min_deg=np.array(mins))


def _name(min_deg: np.ndarray) -> list[str]:
    out = []
    for f, s in min_deg:
        if f > 0:
            out.append("alphaL")
        elif -120 < s < 50:
            out.append("alpha")
        elif f < -110:
            out.append("beta")
        else:
            out.append("ppII")
    return out


@dataclass(frozen=True)
class States:
    """Named metastable cores on the torus grid.

    Attributes:
        names: state names in output order.
        core: (nbins, nbins) int grid, state index of a core cell or -1.
        nbins: grid size.
        summary: JSON-ready segmentation record (parameters, per-state minimum, F_min, populations, saddles).
    """

    names: tuple[str, ...]
    core: np.ndarray
    nbins: int
    summary: dict = field(compare=False)

    def assign(self, phi: np.ndarray, psi: np.ndarray) -> np.ndarray:
        """Core index of each frame, -1 outside every core."""
        i, j = torus_cells(phi, psi, self.nbins)
        return self.core[i, j]


def define_states(phi: np.ndarray, psi: np.ndarray) -> States:
    """Cores of the reference (phi, psi) samples, named and ordered alpha, beta (or beta/ppII), ppII, alphaL.

    Raises:
        RuntimeError: no beta or ppII basin, or two basins get the same name.
    """
    cfg = dict(STATE_PARAMS)
    seg = segment_torus(phi, psi, **cfg)
    names = _name(seg["min_deg"])
    if "beta" not in names and "ppII" not in names:
        raise RuntimeError(f"no beta/ppII basin: {names}")
    if names.count("beta") + names.count("ppII") == 1:
        names = ["beta/ppII" if n in ("beta", "ppII") else n for n in names]
    if len(set(names)) != len(names):
        raise RuntimeError(f"basin names collide: {names} at {seg['min_deg'].tolist()}")
    order = sorted(range(len(names)), key=lambda b: STATE_ORDER.index(names[b]))
    remap = np.full(len(names), -1)
    remap[order] = np.arange(len(names))
    core = np.where(seg["core"] >= 0, remap[np.maximum(seg["core"], 0)], -1)
    names = [names[b] for b in order]
    o = np.asarray(order)
    sad = seg["saddle"][np.ix_(o, o)]
    fm = seg["F_min"][o]
    summary = dict(
        params=cfg,
        states={n: dict(min_deg=seg["min_deg"][b].tolist(), F_min=float(seg["F_min"][b]),
                        basin_pop=float(seg["pop"][b]), core_pop=float(seg["core_pop"][b]))
                for n, b in zip(names, order)},
        saddle_above_higher_min_kbt={f"{names[a]}|{names[b]}": float(sad[a, b] - max(fm[a], fm[b]))
                                     for a in range(len(names)) for b in range(a + 1, len(names))
                                     if np.isfinite(sad[a, b])})
    return States(names=tuple(names), core=core, nbins=int(cfg["nbins"]), summary=summary)


def milestone(core: np.ndarray) -> np.ndarray:
    """Core-set milestoning labels of one chain's core sequence (-1 before the first core hit)."""
    hit = np.flatnonzero(core >= 0)
    lab = np.full(core.shape, -1)
    if len(hit) == 0:
        return lab
    idx = np.where(core >= 0, np.arange(len(core)), 0)
    lab = core[np.maximum.accumulate(idx)]
    lab[:hit[0]] = -1
    return lab


def _lagged_sums(x: np.ndarray, max_lag: int) -> np.ndarray:
    """For lags t = 0..max_lag: S_t = sum_s x_s x_{s+t}, A_t = sum_s x_s, B_t = sum_s x_{s+t}, n_t pairs."""
    lz = len(x)
    nfft = 1 << int(np.ceil(np.log2(2 * lz)))
    S = np.fft.irfft(np.conj(np.fft.rfft(x, nfft)) * np.fft.rfft(x, nfft), nfft)[:max_lag + 1]
    S = np.pad(S, (0, max_lag + 1 - len(S)))  # chains shorter than max_lag: lags beyond the chain contribute nothing
    t = np.arange(max_lag + 1)
    m = np.clip(np.minimum(lz, lz - t), 0, None)
    pz = np.concatenate([[0], np.cumsum(x)])
    A = pz[m]
    B = pz[np.minimum(t + m, lz)] - pz[np.minimum(t, lz)]
    S = np.where(m > 0, S, 0.0)
    return np.stack([S, A, B, m.astype(float)])


def chain_statistics(lab: np.ndarray, t: np.ndarray, n_states: int, max_lag: int) -> dict:
    """Sufficient statistics of one milestoned chain.

    Args:
        lab: milestoning labels per frame.
        t: frame times (ps).
        n_states: number of states.
        max_lag: largest ACF lag in frames.

    Returns:
        dict of arrays with a leading chain axis of length 1: first-passage sums ``mfpt_sum`` / counts ``mfpt_n``
        (completed), ``mfpt_cens`` / ``mfpt_cens_time`` (right-censored at the chain end), ``mfpt_arrivals``,
        ``exposure`` (labeled frames per state), ``exposure_ps`` (labeled time per state: frame i lasts
        t[i+1] - t[i], the last frame the chain's median interval), ``trans`` (label changes) and ``acf``
        (n_states, 4, max_lag+1).
    """
    nst = n_states
    z = lambda *shape: np.zeros((1,) + shape)
    out = dict(mfpt_sum=z(nst, nst), mfpt_n=z(nst, nst), mfpt_cens=z(nst, nst), mfpt_cens_time=z(nst, nst),
               mfpt_arrivals=z(nst, nst), exposure=z(nst), exposure_ps=z(nst), trans=z(nst, nst))
    unit = lambda a: np.zeros(len(a), np.int64)
    first = np.flatnonzero(lab >= 0)
    if len(first):
        f0 = first[0]
        ch = np.flatnonzero(lab[1:] != lab[:-1]) + 1
        ch = ch[ch > f0]
        out["exposure"][0] = [np.count_nonzero(lab[f0:] == i) for i in range(nst)]
        d = np.diff(t)
        duration = np.r_[d, np.median(d) if len(d) else 0.0][f0:]
        out["exposure_ps"][0] = [duration[lab[f0:] == i].sum() for i in range(nst)]
        np.add.at(out["trans"], (unit(ch), lab[ch - 1], lab[ch]), 1)
        # np.add.at accumulates in passage order, which fixes the floating-point summation order
        for B in range(nst):
            eB = ch[lab[ch] == B]
            for A in range(nst):
                if A == B:
                    continue
                eA = ch[lab[ch] == A]
                if len(eA) == 0:
                    continue
                k = np.searchsorted(eB, eA, side="right")
                done = k < len(eB)
                np.add.at(out["mfpt_sum"], (unit(eA[done]), A, B), t[eB[k[done]]] - t[eA[done]])
                np.add.at(out["mfpt_n"], (unit(eA[done]), A, B), 1)
                np.add.at(out["mfpt_cens"], (unit(eA[~done]), A, B), 1)
                np.add.at(out["mfpt_cens_time"], (unit(eA[~done]), A, B), t[-1] - t[eA[~done]])
                tgt = np.unique(eB[k[done]])
                np.add.at(out["mfpt_arrivals"], (unit(tgt), A, B), 1)
    f0 = first[0] if len(first) else len(lab)
    acf = [_lagged_sums((lab[f0:] == i).astype(float), max_lag) if len(lab) > f0 else np.zeros((4, max_lag + 1))
           for i in range(nst)]
    out["acf"] = np.stack(acf)[None]
    return out


def acf_from(acc: np.ndarray) -> np.ndarray:
    """Normalized autocorrelation C(t) = cov(t) / cov(0) with the pooled mean, from summed lagged sums
    (..., 4, L+1)."""
    S, A, B, n = acc[..., 0, :], acc[..., 1, :], acc[..., 2, :], acc[..., 3, :]
    mu = A[..., :1] / n[..., :1]
    cov = (S - mu * (A + B) + mu ** 2 * n) / np.maximum(n, 1)
    return cov / cov[..., :1]


def acf_times(C: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Integrated time (trapezoid rule, frame units, up to the first C < ACF_CUT; nan if never) and the
    first 1/e crossing (linear interpolation; nan if never)."""
    below = C < ACF_CUT
    has = below.any(-1)
    tc = np.where(has, below.argmax(-1), C.shape[-1] - 1)
    cs = np.cumsum(C, axis=-1)
    tau = (np.take_along_axis(cs, tc[..., None], -1)[..., 0] - 0.5 * C[..., 0]
           - 0.5 * np.take_along_axis(C, tc[..., None], -1)[..., 0])
    tau = np.where(has, tau, np.nan)
    be = C < np.exp(-1)
    he = be.any(-1)
    k = np.where(he, be.argmax(-1), 1)
    c0 = np.take_along_axis(C, (k - 1)[..., None], -1)[..., 0]
    c1 = np.take_along_axis(C, k[..., None], -1)[..., 0]
    te = (k - 1) + (c0 - np.exp(-1)) / np.where(c0 != c1, c0 - c1, 1)
    return tau, np.where(he, te, np.nan)


def _percentile_inf(a: np.ndarray, q: float) -> np.ndarray:
    """Percentile ``q`` over axis 0 with linear interpolation that never forms inf - inf; nan dropped."""
    out = np.full(a.shape[1:], np.nan)
    for idx in np.ndindex(*a.shape[1:]):
        v = np.sort(a[(slice(None),) + idx])
        v = v[~np.isnan(v)]
        if len(v) == 0:
            continue
        pos = q / 100.0 * (len(v) - 1)
        i0, i1 = int(np.floor(pos)), int(np.ceil(pos))
        out[idx] = v[i0] if v[i0] == v[i1] else v[i0] + (v[i1] - v[i0]) * (pos - i0)
    return out


def summarize(acc: Mapping[str, np.ndarray], rng: np.random.Generator, *, n_boot: int) -> dict:
    """Point estimates and 95% percentile bootstrap intervals over chains.

    One multinomial resampling of the U chains per replicate (``n_boot`` replicates drawn from ``rng``).
    MFPT A->B is the exponential MLE (completed times + censored durations) / completed passages; a
    replicate without completions is inf, and a point estimate without completions is a lower bound.

    Returns:
        dict with ``mfpt`` (value, lo, hi, n, censored, arrivals, exposure, lower_bound), ``acf`` (C, tau_int,
        t_1e and their intervals), ``occupancy`` (frame fractions), ``trans``, ``exit_per_ns`` (exits per ns of
        labeled simulated time).
    """
    U = acc["mfpt_sum"].shape[0]
    W = np.vstack([np.ones(U), rng.multinomial(U, np.full(U, 1.0 / U), size=n_boot)])
    ci = lambda a: (np.nanpercentile(a[1:], 2.5, axis=0), np.nanpercentile(a[1:], 97.5, axis=0))
    res = {}
    # an all-nan column (no ACF cut within max_lag) yields a nan interval
    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mn = np.tensordot(W, acc["mfpt_n"], 1)
        num = np.tensordot(W, acc["mfpt_sum"] + acc["mfpt_cens_time"], 1)
        mf = np.where(mn > 0, num / np.where(mn > 0, mn, 1), np.where(num > 0, np.inf, np.nan))
        res["mfpt"] = dict(value=mf[0], lo=_percentile_inf(mf[1:], 2.5), hi=_percentile_inf(mf[1:], 97.5), n=mn[0],
                           censored=acc["mfpt_cens"].sum(0), arrivals=acc["mfpt_arrivals"].sum(0), exposure=num[0],
                           lower_bound=(mn[0] == 0))
        A = acc["acf"]
        nobs, L1 = A.shape[1], A.shape[-1]
        C0 = acf_from(A.sum(0))
        flat = A.reshape(U, -1)
        ti, te = [], []
        for k in range(0, n_boot + 1, BOOT_BATCH):
            a, b = acf_times(acf_from((W[k:k + BOOT_BATCH] @ flat).reshape(-1, nobs, 4, L1)))
            ti.append(a)
            te.append(b)
        ti, te = np.concatenate(ti), np.concatenate(te)
        res["acf"] = dict(C=C0, tau_int=ti[0], tau_int_lo=ci(ti)[0], tau_int_hi=ci(ti)[1], t_1e=te[0],
                          t_1e_lo=ci(te)[0], t_1e_hi=ci(te)[1])
        ex = acc["exposure"].sum(0)
        res["occupancy"] = ex / ex.sum()
        res["trans"] = acc["trans"].sum(0)
        res["exit_per_ns"] = res["trans"].sum(1) / acc["exposure_ps"].sum(0) * 1000.0
    return res


def chain_kinetics(chains: Sequence[Mapping[str, np.ndarray]], states: States, rng: np.random.Generator, *,
                   n_boot: int, max_lag: int) -> tuple[dict, np.ndarray]:
    """Kinetics of a set of chains, the bootstrap resampling chains.

    Args:
        chains: per chain ``phi``, ``psi`` (rad) and ``t`` (ps) of consecutive frames.
        states: from :func:`define_states`.
        rng: bootstrap generator (advanced by this call).
        n_boot: bootstrap replicates.
        max_lag: largest ACF lag in frames.

    Returns:
        (JSON-ready record, ACF array (n_states, max_lag + 1)). MFPTs and times are in ps; ACF times are in
        frames.

    Raises:
        ValueError: ``chains`` is empty.
    """
    if not chains:
        raise ValueError("kinetics needs at least one chain")
    nst = len(states.names)
    parts, n_lab = [], 0
    for c in chains:
        lab = milestone(states.assign(c["phi"], c["psi"]))
        first = np.flatnonzero(lab >= 0)
        if len(first):
            n_lab += len(lab) - first[0]
        parts.append(chain_statistics(lab, np.asarray(c["t"], np.float64), nst, max_lag))
    acc = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    res = summarize(acc, rng, n_boot=n_boot)
    nm = states.names
    f = lambda v: None if not np.isfinite(v) else float(v)
    mf, ac = res["mfpt"], res["acf"]
    record = dict(
        chains=len(chains), frames=int(sum(len(c["phi"]) for c in chains)), milestoned_frames=int(n_lab),
        occupancy={nm[i]: f(res["occupancy"][i]) for i in range(nst)},
        exits={nm[i]: int(res["trans"][i].sum()) for i in range(nst)},
        exits_per_ns={nm[i]: f(res["exit_per_ns"][i]) for i in range(nst)},
        transitions={f"{nm[i]}->{nm[j]}": int(res["trans"][i, j]) for i in range(nst) for j in range(nst) if i != j},
        mfpt_ps={f"{nm[a]}->{nm[b]}": dict(
            value=f(mf["value"][a, b]),
            ci=[None, None] if mf["lower_bound"][a, b] else [f(mf["lo"][a, b]), f(mf["hi"][a, b])],
            ci_hi_unbounded=bool(np.isposinf(mf["hi"][a, b])), lower_bound=bool(mf["lower_bound"][a, b]),
            n_completed=int(mf["n"][a, b]), n_censored=int(mf["censored"][a, b]),
            total_exposure_ps=float(mf["exposure"][a, b]), arrivals=int(mf["arrivals"][a, b]))
            for a in range(nst) for b in range(nst) if a != b},
        relaxation_frames={nm[k]: dict(tau_int=f(ac["tau_int"][k]),
                                       tau_int_ci=[f(ac["tau_int_lo"][k]), f(ac["tau_int_hi"][k])],
                                       t_1e=f(ac["t_1e"][k]), t_1e_ci=[f(ac["t_1e_lo"][k]), f(ac["t_1e_hi"][k])])
                           for k in range(nst)})
    return record, ac["C"]
