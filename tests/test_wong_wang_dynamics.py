# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Wong-Wang reduced-circuit dynamical invariants

"""Deterministic probes of the maintained Wong-Wang 2006 science boundary."""

from __future__ import annotations

import math

import numpy as np
import pytest

from sc_neurocore.neurons.models.wong_wang import WongWangUnit


def _settle(unit: WongWangUnit, stim1: float, stim2: float, steps: int = 5_000) -> None:
    """Advance a zero-sample deterministic trajectory."""
    for _ in range(steps):
        unit.step_with_gaussian_samples(stim1, stim2, 0.0, 0.0)


def test_transfer_function_is_non_negative_and_monotone() -> None:
    """Pin the reduced population response over its operational current range."""
    currents = np.linspace(-1.0, 3.0, 401)
    rates = np.asarray([WongWangUnit._phi(float(current)) for current in currents])
    assert np.logical_and(np.isfinite(rates), rates >= 0.0).all()
    assert (np.diff(rates) >= -1.0e-14).all()


def test_transfer_function_has_the_lhopital_limit() -> None:
    """Resolve the apparent 0/0 singularity at aI-b=0."""
    pivot = 108.0 / 270.0
    expected = 1.0 / 0.154
    assert WongWangUnit._phi(pivot) == pytest.approx(expected, abs=1.0e-15)
    assert WongWangUnit._phi(pivot - 1.0e-8) == pytest.approx(expected, abs=3.0e-6)
    assert WongWangUnit._phi(pivot + 1.0e-8) == pytest.approx(expected, abs=3.0e-6)


def test_symmetric_deterministic_drive_preserves_exchange_symmetry() -> None:
    """Equal populations and inputs remain exactly exchange symmetric."""
    unit = WongWangUnit(sigma=0.0)
    _settle(unit, 0.05, 0.05)
    assert unit.s1 == unit.s2
    assert unit.noise1 == unit.noise2 == 0.0


@pytest.mark.parametrize(
    ("stim1", "stim2", "winner"),
    ((0.1, 0.0, 1), (0.0, 0.1, 2), (0.2, 0.0, 1), (0.0, 0.2, 2)),
)
def test_biased_drive_selects_the_corresponding_attractor(
    stim1: float,
    stim2: float,
    winner: int,
) -> None:
    """Exercise mutual competition without a stochastic outcome assertion."""
    unit = WongWangUnit(sigma=0.0)
    _settle(unit, stim1, stim2)
    winning, losing = (unit.s1, unit.s2) if winner == 1 else (unit.s2, unit.s1)
    assert winning > 0.75
    assert losing < 0.05


def test_self_coupling_raises_the_driven_population_fixed_point() -> None:
    """Verify the recurrent excitation parameter has its documented direction."""
    values = []
    for coupling in (0.15, 0.2609, 0.35):
        unit = WongWangUnit(j_n=coupling, sigma=0.0)
        _settle(unit, 0.1, 0.0)
        values.append(unit.s1)
    assert values[0] < values[1] < values[2]


def test_cross_coupling_suppresses_the_losing_population() -> None:
    """Verify the cross-population term enters with the published minus sign."""
    losers = []
    for coupling in (0.02, 0.0497, 0.09):
        unit = WongWangUnit(j_cross=coupling, sigma=0.0)
        _settle(unit, 0.15, 0.0)
        losers.append(unit.s2)
    assert losers[0] > losers[1] > losers[2]


def test_pre_update_ou_current_changes_the_corresponding_rate() -> None:
    """Keep AMPA noise as an additive current state, not direct gating noise."""
    unit = WongWangUnit(noise1=0.03, noise2=-0.03, sigma=0.0)
    rate1, rate2 = unit.step_with_gaussian_samples(0.0, 0.0, 0.0, 0.0)
    assert rate1 > rate2
    assert unit.noise1 == pytest.approx(0.03 * (1.0 - unit.dt / unit.tau_ampa))
    assert unit.noise2 == pytest.approx(-0.03 * (1.0 - unit.dt / unit.tau_ampa))


def test_external_samples_follow_the_published_ou_scaling() -> None:
    """Distinguish sqrt(dt/tau_AMPA)*sigma from direct per-step noise."""
    unit = WongWangUnit(sigma=0.02)
    unit.step_with_gaussian_samples(0.0, 0.0, 1.5, -0.25)
    scale = math.sqrt(unit.dt / unit.tau_ampa) * unit.sigma
    assert unit.noise1 == pytest.approx(1.5 * scale, abs=1.0e-15)
    assert unit.noise2 == pytest.approx(-0.25 * scale, abs=1.0e-15)


def test_seeded_stochastic_path_is_reproducible() -> None:
    """Pin the Python caller-visible normal stream without claiming cross-RNG parity."""
    np.random.seed(2026)
    first = WongWangUnit()
    trace_a = [(*first.step(0.02, 0.0), first.s1, first.s2) for _ in range(128)]
    np.random.seed(2026)
    second = WongWangUnit()
    trace_b = [(*second.step(0.02, 0.0), second.s1, second.s2) for _ in range(128)]
    assert trace_a == trace_b


def test_bounded_deterministic_sample_trace_remains_physical() -> None:
    """Run a varied two-second trace through all four state equations."""
    unit = WongWangUnit()
    for index in range(20_000):
        rate1, rate2 = unit.step_with_gaussian_samples(
            0.02 + 0.01 * math.sin(index / 100.0),
            0.01 + 0.01 * math.cos(index / 130.0),
            math.sin(index * 0.37),
            math.cos(index * 0.29),
        )
        assert math.isfinite(rate1) and rate1 >= 0.0
        assert math.isfinite(rate2) and rate2 >= 0.0
    assert 0.0 <= unit.s1 <= 1.0
    assert 0.0 <= unit.s2 <= 1.0
    assert math.isfinite(unit.noise1) and math.isfinite(unit.noise2)
