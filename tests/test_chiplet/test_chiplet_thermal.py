# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Chiplet thermal-network contracts

"""Physics, transient, override, and validation tests for package thermal analysis."""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pytest

from sc_neurocore.chiplet import (
    ChipletDie,
    ChipletTopology,
    DieThermal,
    InterposerLink,
    InterposerTech,
    simulate_thermal,
)


def test_single_die_obeys_steady_state_ohms_law() -> None:
    topology = ChipletTopology(dies=[ChipletDie(0)])
    report = simulate_thermal(topology, {0: 100.0}, ambient_c=25.0)
    assert report.die_temps[0] == pytest.approx(25.15)


def test_link_couples_hot_and_cold_dies_with_power_balance() -> None:
    topology = ChipletTopology(dies=[ChipletDie(0), ChipletDie(1)])
    topology.add_link(InterposerLink.from_tech(0, 1, InterposerTech.UCIE))
    report = simulate_thermal(topology, {0: 10_000.0, 1: 0.0}, ambient_c=25.0)
    hot, cold = report.die_temps[0], report.die_temps[1]
    assert 25.0 < cold < hot < 40.0
    assert ((hot - 25.0) + (cold - 25.0)) / 1.5 == pytest.approx(10.0)


def test_low_resistance_bond_spreads_more_heat() -> None:
    powers = {0: 10_000.0, 1: 0.0}
    reports = {}
    for technology in (InterposerTech.COWOS, InterposerTech.ORGANIC):
        topology = ChipletTopology(dies=[ChipletDie(0), ChipletDie(1)])
        topology.add_link(InterposerLink.from_tech(0, 1, technology))
        reports[technology] = simulate_thermal(topology, powers)
    assert reports[InterposerTech.COWOS].die_temps[1] > reports[InterposerTech.ORGANIC].die_temps[1]


def test_custom_link_resistance_overrides_technology_default() -> None:
    powers = {0: 10_000.0, 1: 0.0}

    def run(resistance: float) -> tuple[float, float]:
        topology = ChipletTopology(dies=[ChipletDie(0), ChipletDie(1)])
        topology.add_link(
            InterposerLink(
                0,
                1,
                technology=InterposerTech.CUSTOM,
                thermal_resistance_k_per_w=resistance,
            )
        )
        report = simulate_thermal(topology, powers)
        return report.die_temps[0], report.die_temps[1]

    weak_hot, weak_cold = run(10.0)
    strong_hot, strong_cold = run(0.2)
    assert strong_cold > weak_cold
    assert strong_hot < weak_hot


def test_conductance_matrix_is_symmetric_with_zero_diagonal() -> None:
    report = simulate_thermal(ChipletTopology.ring(4))
    conductance = report.conductance_matrix
    assert conductance is not None
    np.testing.assert_allclose(conductance, conductance.T, atol=1e-12)
    np.testing.assert_array_equal(np.diag(conductance), np.zeros(4))


def test_missing_link_die_is_ignored_by_thermal_coupling() -> None:
    topology = ChipletTopology(dies=[ChipletDie(0)])
    topology.add_link(InterposerLink(0, 99))
    report = simulate_thermal(topology, {0: 100.0})
    assert report.die_temps[0] == pytest.approx(25.15)


def test_die_state_override_controls_throttle_threshold() -> None:
    topology = ChipletTopology(dies=[ChipletDie(0)])
    state = {0: DieThermal(0, max_temperature_c=25.1)}
    report = simulate_thermal(topology, {0: 100.0}, die_state=state)
    assert report.throttled_dies == [0]
    assert state[0].temperature_c == report.die_temps[0]


def test_transient_converges_and_starts_near_ambient() -> None:
    topology = ChipletTopology.ring(2)
    report = simulate_thermal(
        topology,
        {0: 500.0, 1: 500.0},
        ambient_c=20.0,
        transient_steps=2000,
        transient_dt_s=1e-3,
    )
    assert report.transient_temps is not None
    assert report.transient_times_s is not None
    assert np.all((report.transient_temps[0] > 20.0) & (report.transient_temps[0] < 25.0))
    steady = np.array([report.die_temps[die.die_id] for die in topology.dies])
    np.testing.assert_allclose(report.transient_temps[-1], steady, atol=0.01)
    assert report.transient_times_s[-1] == pytest.approx(2.0)


@pytest.mark.parametrize(
    "call",
    [
        lambda: simulate_thermal(ChipletTopology()),
        lambda: simulate_thermal(ChipletTopology(dies=[ChipletDie(0)]), ambient_c=math.nan),
        lambda: simulate_thermal(ChipletTopology(dies=[ChipletDie(0)]), transient_steps=-1),
        lambda: simulate_thermal(ChipletTopology(dies=[ChipletDie(0)]), transient_dt_s=0.0),
        lambda: simulate_thermal(ChipletTopology(dies=[ChipletDie(0)]), {0: -1.0}),
        lambda: simulate_thermal(ChipletTopology(dies=[ChipletDie(0), ChipletDie(0)])),
        lambda: DieThermal(-1),
        lambda: DieThermal(0, temperature_c=math.nan),
        lambda: DieThermal(0, power_mw=-1.0),
        lambda: DieThermal(0, heat_capacity_j_per_k=0.0),
        lambda: DieThermal(0, r_spread_k_per_w=-1.0),
    ],
)
def test_invalid_thermal_contracts_fail(call: Callable[[], object]) -> None:
    with pytest.raises(ValueError):
        call()
