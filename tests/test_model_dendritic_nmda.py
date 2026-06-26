# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Dendritic NMDA RK4 model tests

"""Production-path tests for the two-compartment dendritic NMDA neuron."""

from __future__ import annotations

import math

import pytest

from sc_neurocore.neurons.models import DendriticNMDANeuron as PublicDendriticNMDANeuron
from sc_neurocore.neurons.models.dendritic_nmda import DendriticNMDANeuron


def _run_spikes(neuron: DendriticNMDANeuron, steps: int, i_soma: float, glutamate: float) -> int:
    return sum(neuron.step(i_soma, glutamate) for _ in range(steps))


def test_default_integrator_is_rk4() -> None:
    neuron = DendriticNMDANeuron()
    assert neuron.integrator == "rk4"
    assert neuron.v_soma == -65.0
    assert neuron.v_dend == -65.0
    assert PublicDendriticNMDANeuron is DendriticNMDANeuron


def test_rejects_unknown_integrator() -> None:
    with pytest.raises(ValueError, match="integrator"):
        DendriticNMDANeuron(integrator="euler")  # type: ignore[arg-type] # invalid runtime contract


def test_mg_block_formula_matches_jahr_stevens() -> None:
    neuron = DendriticNMDANeuron()
    for voltage in (-80.0, -65.0, -40.0, -20.0, 0.0, 20.0):
        expected = 1.0 / (1.0 + (1.0 / 3.57) * math.exp(-0.062 * voltage))
        assert neuron.mg_block(voltage) == pytest.approx(expected, abs=1e-12)


def test_rk4_cross_backend_anchor() -> None:
    neuron = DendriticNMDANeuron()
    spikes = _run_spikes(neuron, 20_000, 50.0, 0.5)
    assert spikes == 253
    assert math.isfinite(neuron.v_soma)
    assert math.isfinite(neuron.v_dend)


def test_baseline_euler_preserves_historical_comparison_path() -> None:
    rk4 = DendriticNMDANeuron()
    euler = DendriticNMDANeuron(integrator="baseline_euler")
    assert _run_spikes(rk4, 20_000, 30.0, 0.5) == 111
    assert _run_spikes(euler, 20_000, 30.0, 0.5) == 112
    assert (rk4.v_soma, rk4.v_dend) != (euler.v_soma, euler.v_dend)


def test_invalid_runtime_input_preserves_state() -> None:
    neuron = DendriticNMDANeuron()
    for _ in range(10):
        neuron.step(50.0, 0.5)
    previous = (neuron.v_soma, neuron.v_dend)
    with pytest.raises(ValueError, match="i_soma"):
        neuron.step(math.inf, 0.5)
    assert (neuron.v_soma, neuron.v_dend) == previous
    with pytest.raises(ValueError, match="glutamate"):
        neuron.step(50.0, -1.0)
    assert (neuron.v_soma, neuron.v_dend) == previous


def test_invalid_runtime_configuration_preserves_state() -> None:
    neuron = DendriticNMDANeuron()
    for _ in range(10):
        neuron.step(50.0, 0.5)
    previous = (neuron.v_soma, neuron.v_dend)
    neuron.tau_dend = 0.0
    with pytest.raises(ValueError, match="tau_dend"):
        neuron.step(50.0, 0.5)
    assert (neuron.v_soma, neuron.v_dend) == previous


def test_reset_restores_both_compartments() -> None:
    neuron = DendriticNMDANeuron()
    _ = _run_spikes(neuron, 500, 50.0, 0.5)
    neuron.reset()
    assert neuron.v_soma == -65.0
    assert neuron.v_dend == -65.0
