# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - source-faithful Kobayashi 2009 MAT* tests

from __future__ import annotations

import math

import pytest

from sc_neurocore.neurons.models.mat import MATNeuron
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.network import Network
from sc_neurocore.network.population import Population
from sc_neurocore.network.stimulus import PoissonInput


def test_paper_regular_spiking_defaults() -> None:
    """The public default is the paper's named RS example profile."""
    neuron = MATNeuron()
    assert (
        neuron.v,
        neuron.omega,
        neuron.tau_m,
        neuron.tau_1,
        neuron.tau_2,
        neuron.alpha_1,
        neuron.alpha_2,
        neuron.resistance,
        neuron.refractory_period,
        neuron.dt,
    ) == (0.0, 19.0, 5.0, 10.0, 200.0, 37.0, 2.0, 50.0, 2.0, 0.001)


def test_named_paper_profiles_are_explicit() -> None:
    """RS, IB, and FS examples retain the parameters reported in Figure 1."""
    assert (MATNeuron.regular_spiking().omega, MATNeuron.regular_spiking().alpha_1) == (
        19.0,
        37.0,
    )
    ib = MATNeuron.intrinsically_bursting()
    fs = MATNeuron.fast_spiking()
    assert (ib.omega, ib.alpha_1, ib.alpha_2) == (26.0, 1.7, 2.0)
    assert (fs.omega, fs.alpha_1, fs.alpha_2) == (11.0, 10.0, 0.002)


def test_one_step_matches_independent_mat_star_equations() -> None:
    """One step matches Euler voltage and exact threshold-history decay."""
    neuron = MATNeuron(v=3.0, theta1=7.0, theta2=4.0, refractory_remaining=1.25)
    current = 0.42
    expected_v = 3.0 + neuron.dt * (-3.0 + neuron.resistance * current) / neuron.tau_m
    expected_theta1 = 7.0 * math.exp(-neuron.dt / neuron.tau_1)
    expected_theta2 = 4.0 * math.exp(-neuron.dt / neuron.tau_2)
    assert neuron.step(current) == 0
    assert neuron.v == pytest.approx(expected_v, abs=1.0e-15)
    assert neuron.theta1 == pytest.approx(expected_theta1, abs=1.0e-15)
    assert neuron.theta2 == pytest.approx(expected_theta2, abs=1.0e-15)
    assert neuron.refractory_remaining == pytest.approx(1.249, abs=1.0e-15)


def test_spike_does_not_reset_voltage() -> None:
    """MAT* records threshold history without a membrane reset."""
    neuron = MATNeuron(v=20.0)
    expected_v = neuron.v + neuron.dt * (-neuron.v) / neuron.tau_m
    assert neuron.step(0.0) == 1
    assert neuron.v == pytest.approx(expected_v, abs=1.0e-15)
    assert neuron.theta1 == pytest.approx(neuron.alpha_1)
    assert neuron.theta2 == pytest.approx(neuron.alpha_2)
    assert neuron.refractory_remaining == neuron.refractory_period


def test_persistent_suprathreshold_voltage_reemits_after_two_ms() -> None:
    """A still-high membrane may fire again when absolute refractory expires."""
    neuron = MATNeuron(
        v=25.0,
        omega=1.0,
        alpha_1=0.0,
        alpha_2=0.0,
        tau_m=1.0e9,
        refractory_period=2.0,
        dt=0.5,
    )
    assert neuron.step(0.0) == 1
    assert [neuron.step(0.0) for _ in range(4)] == [0, 0, 0, 1]
    assert neuron.v > neuron.threshold


def test_invalid_candidate_fails_atomically() -> None:
    """An out-of-envelope voltage candidate cannot partially mutate state."""
    neuron = MATNeuron()
    before = (neuron.v, neuron.theta1, neuron.theta2, neuron.refractory_remaining)
    with pytest.raises(ValueError, match="voltage candidate"):
        neuron.step(1.0e9)
    assert (neuron.v, neuron.theta1, neuron.theta2, neuron.refractory_remaining) == before


def test_reset_clears_dynamic_state_but_preserves_profile() -> None:
    """Reset returns to zero-rest state without replacing configured parameters."""
    neuron = MATNeuron.fast_spiking(dt=0.01)
    for _ in range(500):
        neuron.step(0.5)
    neuron.reset()
    assert (neuron.v, neuron.theta1, neuron.theta2, neuron.refractory_remaining) == (
        0.0,
        0.0,
        0.0,
        0.0,
    )
    assert (neuron.omega, neuron.alpha_1, neuron.alpha_2, neuron.dt) == (
        11.0,
        10.0,
        0.002,
        0.01,
    )


def test_source_model_runs_through_public_network_pipeline() -> None:
    """The public MAT identity is usable by the ordinary Python network path."""
    population = Population(MATNeuron, n=4, label="mat-source")
    drive = PoissonInput(n=4, rate_hz=500.0, weight=30.0, dt=0.001, seed=49)
    monitor = SpikeMonitor(population)
    Network(population, drive, monitor).run(duration=0.5, dt=0.001, backend="python")
    assert monitor.count > 0
