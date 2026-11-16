"""Energy filter of one KC-map proposal: rigid and internal momenta, the heat-bath law of the internal kinetic energy,
the energy guard and the momentum rescale. All energies are in kBT."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp

from ..core import constants as C


def filter_max_dh_kbt(n_atoms: int) -> float:
    """Largest |dU + dK_int| (kBT) of a proposal inside the energy guard of an ``n_atoms`` system."""
    return C.FILTER_MAX_DH_KBT * math.sqrt((3 * n_atoms - C.FILTER_DOF_OFFSET) / C.FILTER_REFERENCE_DOF)


def potential_energy(kcm, phase):
    """U/kBT of one (N, 6) state."""
    return kcm.system.energy(phase[None, :, :3])[0]


def max_atom_kinetic(kcm, phase):
    """Largest per-atom kinetic energy / kBT of one (N, 6) state."""
    return jnp.max(jnp.sum(phase[:, 3:]**2 / (2 * kcm.masses * kcm.kbt_kj_per_mol), axis=-1))


def rigid_momenta(kcm, phase):
    """Per-atom momenta of the rigid motion (total-momentum translation plus rotation about the center of mass), the
    angular momentum and the inertia tensor of one (N, 6) state."""
    m = kcm.masses
    r = phase[:, :3] - jnp.sum(m * phase[:, :3], axis=0) / jnp.sum(m)
    angular = jnp.sum(jnp.cross(r, phase[:, 3:]), axis=0)
    inertia = jnp.einsum("i,ijk->jk", m[:, 0],
                         jnp.sum(r * r, -1)[:, None, None] * jnp.eye(3) - r[:, :, None] * r[:, None, :])
    translation = m * jnp.sum(phase[:, 3:], axis=0) / jnp.sum(m)
    return translation + m * jnp.cross(jnp.linalg.solve(inertia, angular), r), angular, inertia


def energy_terms(kcm, source, source_u, proposal, proposal_u, tau_ps, key):
    """Energy-filter terms of one proposal.

    With H = U + K_int (kBT) and n = 3N - 6 internal momenta, the internal kinetic energy after an exact OU at fixed
    positions is K' = (a sqrt(2 K_in) + sqrt(1 - a^2) Z)^2 / 2 + (1 - a^2) Y / 2, a = exp(-gamma tau), Z ~ N(0, 1),
    Y ~ chi2(n - 1). The filter's target K_int of the proposal is U_in + K' + exchange - U_out, where
    exchange = K_rot(L_in, I(x)) - K_rot(L_in, I(x')) is the work done at fixed angular momentum as the inertia
    changes.

    Returns:
        dict with ``momentum`` and ``rigid`` (internal and rigid momenta of the proposal), ``kinetic`` (its K_int),
        ``bath`` (K'), ``exchange``, ``target`` and ``energy_change`` (dU + dK_int).
    """
    def internal(phase):
        rigid, angular, inertia = rigid_momenta(kcm, phase)
        momentum = phase[:, 3:] - rigid
        return momentum, rigid, jnp.sum(momentum**2 / (2 * kcm.masses * kcm.kbt_kj_per_mol)), angular, inertia

    dof = 3 * proposal.shape[0] - C.FILTER_DOF_OFFSET
    _, _, incoming, angular, inertia = internal(source)
    momentum, rigid, kinetic, _, proposal_inertia = internal(proposal)
    rotational = lambda i: .5 * angular @ jnp.linalg.solve(i, angular) / kcm.kbt_kj_per_mol  # noqa: E731
    exchange = rotational(inertia) - rotational(proposal_inertia)
    normal_key, chi2_key = jax.random.split(key)
    gamma = kcm.gamma_ps_inv
    a = jnp.exp(-gamma * tau_ps)
    fresh = -jnp.expm1(-2 * gamma * tau_ps)
    bath = (.5 * (a * jnp.sqrt(2 * incoming) + jnp.sqrt(fresh) * jax.random.normal(normal_key, (), proposal.dtype))**2
            + .5 * fresh * 2 * jax.random.gamma(chi2_key, (dof - 1) / 2, (), proposal.dtype))
    return dict(momentum=momentum, rigid=rigid, kinetic=kinetic, bath=bath, exchange=exchange,
                target=source_u + bath + exchange - proposal_u,
                energy_change=proposal_u + kinetic - source_u - incoming)


def within_guard(terms, n_atoms):
    """|dU + dK_int| <= ``filter_max_dh_kbt`` and a reachable target K_int >= the kinetic floor."""
    return ((jnp.abs(terms["energy_change"]) <= filter_max_dh_kbt(n_atoms))
            & (terms["target"] >= C.FILTER_KINETIC_FLOOR_KBT))


def apply_filter(terms, proposal):
    """Rescale the proposal's internal momenta to the target K_int (positions, total and angular momentum kept).

    An unreachable target (below the kinetic floor) sets K_int to the heat-bath draw K' + exchange instead.
    """
    floor = C.FILTER_KINETIC_FLOOR_KBT
    feasible = terms["target"] >= floor
    goal = jnp.where(feasible, terms["target"], jnp.maximum(terms["bath"] + terms["exchange"], floor))
    scale = jnp.sqrt(jnp.maximum(goal, floor) / jnp.maximum(terms["kinetic"], floor))
    return jnp.concatenate([proposal[:, :3], terms["rigid"] + scale * terms["momentum"]], axis=-1)
