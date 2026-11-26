# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — AvRonCardiacNeuron physics contracts

"""Model-specific contracts for Av-Ron cardiac ganglion RK4 dynamics."""

from __future__ import annotations

import math

import pytest
from sc_neurocore.neurons.models.av_ron_cardiac import AvRonCardiacNeuron


def test_rk4_reference_point_separates_from_euler_candidate() -> None:
    """Single RK4 step matches the reference trajectory and separates from Euler."""
    neuron = AvRonCardiacNeuron(v=-55.0, h=0.55, n=0.35, s=0.45)

    assert neuron.step(2.0) == 0

    assert math.isclose(neuron.v, -50.0840498399381, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(neuron.h, 0.5506609782132562, rel_tol=0.0, abs_tol=1e-15)
    assert math.isclose(neuron.n, 0.34988677751350306, rel_tol=0.0, abs_tol=1e-15)
    assert math.isclose(neuron.s, 0.4500091998827305, rel_tol=0.0, abs_tol=1e-15)

    euler_v = -49.772441553145036
    assert abs(neuron.v - euler_v) > 1e-8


def test_spike_is_reported_on_upward_threshold_crossing_only() -> None:
    """Spikes fire only on upward threshold crossings."""
    neuron = AvRonCardiacNeuron(v=-20.1, h=0.2, n=0.1, s=0.0, v_threshold=-20.0)

    assert neuron.step(50.0) == 1
    assert neuron.v >= -20.0
    assert neuron.step(50.0) == 0


def test_gates_remain_physical_during_nominal_burst_drive() -> None:
    """Gating variables remain in ``[0, 1]`` under sustained bursting drive."""
    neuron = AvRonCardiacNeuron()

    for _ in range(1000):
        neuron.step(5.0)
        assert 0.0 <= neuron.h <= 1.0
        assert 0.0 <= neuron.n <= 1.0
        assert 0.0 <= neuron.s <= 1.0

    assert math.isfinite(neuron.v)


def test_invalid_runtime_input_raises_and_preserves_state() -> None:
    """Non-finite drive raises ``ValueError`` without mutating state."""
    neuron = AvRonCardiacNeuron(v=-52.0, h=0.4, n=0.2, s=0.8)
    before = (neuron.v, neuron.h, neuron.n, neuron.s)

    with pytest.raises(ValueError, match="current"):
        neuron.step(float("nan"))

    assert (neuron.v, neuron.h, neuron.n, neuron.s) == before


def test_invalid_parameter_raises_and_preserves_state() -> None:
    """Non-positive ``dt`` fails at construction and on mutated ``step`` without mutating state."""
    with pytest.raises(ValueError, match="dt"):
        AvRonCardiacNeuron(v=-52.0, h=0.4, n=0.2, s=0.8, dt=0.0)

    neuron = AvRonCardiacNeuron(v=-52.0, h=0.4, n=0.2, s=0.8)
    neuron.dt = 0.0
    before = (neuron.v, neuron.h, neuron.n, neuron.s)

    with pytest.raises(ValueError, match="dt"):
        neuron.step(1.0)

    assert (neuron.v, neuron.h, neuron.n, neuron.s) == before


def test_corrupted_gate_state_raises_and_preserves_state() -> None:
    """Out-of-range gating state fails at construction and on mutated ``step``."""
    with pytest.raises(ValueError, match="h"):
        AvRonCardiacNeuron(v=-52.0, h=1.2, n=0.2, s=0.8)

    neuron = AvRonCardiacNeuron(v=-52.0, h=0.4, n=0.2, s=0.8)
    neuron.h = 1.2
    before = (neuron.v, neuron.h, neuron.n, neuron.s)

    with pytest.raises(ValueError, match="h"):
        neuron.step(1.0)

    assert (neuron.v, neuron.h, neuron.n, neuron.s) == before


def test_nonfinite_candidate_raises_and_preserves_state() -> None:
    """Non-finite RK4 candidate raises ``ValueError`` without mutating state."""
    neuron = AvRonCardiacNeuron(v=1.0e308, h=0.5, n=0.5, s=0.5, e_l=-1.0e308)
    before = (neuron.v, neuron.h, neuron.n, neuron.s)

    with pytest.raises(ValueError, match="finite|range"):
        neuron.step(1.0e308)

    assert (neuron.v, neuron.h, neuron.n, neuron.s) == before


def test_reset_restores_default_dynamic_state() -> None:
    """Reset restores default membrane and gating values."""
    neuron = AvRonCardiacNeuron(v=-40.0, h=0.1, n=0.9, s=0.2)

    neuron.reset()

    assert (neuron.v, neuron.h, neuron.n, neuron.s) == (-60.0, 0.6, 0.3, 0.5)


def test_boltzmann_inactivation_and_activation_monotonicity() -> None:
    """Steady-state activation and inactivation curves have opposite monotonicity."""
    neuron = AvRonCardiacNeuron()
    low = neuron._rates(-70.0)
    high = neuron._rates(-20.0)

    assert high[0] > low[0]
    assert high[2] > low[2]
    assert high[1] < low[1]
    assert high[3] < low[3]
    assert 1.0 <= low[4] <= 13.0
    assert 1.0 <= high[5] <= 9.0
    assert 200.0 <= high[6] <= 1200.0


def test_candidate_gate_escape_raises_and_preserves_state() -> None:
    """Intermediate RK4 gate escape raises ``ValueError`` without mutating state."""
    neuron = AvRonCardiacNeuron(v=-20.0, h=0.01, n=0.99, s=0.99, dt=100.0)
    before = (neuron.v, neuron.h, neuron.n, neuron.s)

    with pytest.raises(ValueError, match="finite|range"):
        neuron.step(0.0)

    assert (neuron.v, neuron.h, neuron.n, neuron.s) == before


def test_finite_candidate_with_gate_escape_raises_and_preserves_state() -> None:
    """Final RK4 candidate gate escape raises ``ValueError`` without mutating state."""
    neuron = AvRonCardiacNeuron(v=-80.0, h=0.01, n=0.5, s=0.5, dt=1.0)
    before = (neuron.v, neuron.h, neuron.n, neuron.s)

    with pytest.raises(ValueError, match="finite|range"):
        neuron.step(0.0)

    assert (neuron.v, neuron.h, neuron.n, neuron.s) == before
