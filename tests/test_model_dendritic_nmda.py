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
    """A new neuron uses RK4, rests at -65 mV and is the publicly exported class."""
    neuron = DendriticNMDANeuron()
    assert neuron.integrator == "rk4"
    assert neuron.v_soma == -65.0
    assert neuron.v_dend == -65.0
    assert PublicDendriticNMDANeuron is DendriticNMDANeuron


def test_rejects_unknown_integrator() -> None:
    """An integrator name other than the two supported ones is refused."""
    with pytest.raises(ValueError, match="integrator"):
        DendriticNMDANeuron(integrator="euler")  # type: ignore[arg-type] # invalid runtime contract


def test_mg_block_formula_matches_jahr_stevens() -> None:
    """The block factor equals the Jahr and Stevens formula at six potentials."""
    neuron = DendriticNMDANeuron()
    for voltage in (-80.0, -65.0, -40.0, -20.0, 0.0, 20.0):
        expected = 1.0 / (1.0 + (1.0 / 3.57) * math.exp(-0.062 * voltage))
        assert neuron.mg_block(voltage) == pytest.approx(expected, abs=1e-12)


def test_rk4_cross_backend_anchor() -> None:
    """The spike count that every lane must reproduce for the anchor drive."""
    neuron = DendriticNMDANeuron()
    spikes = _run_spikes(neuron, 20_000, 50.0, 0.5)
    assert spikes == 266
    assert math.isfinite(neuron.v_soma)
    assert math.isfinite(neuron.v_dend)


def _steady_dendrite_without_soma_current(neuron: DendriticNMDANeuron, glutamate: float) -> float:
    """Solve the resting two-compartment balance for the dendrite by bisection.

    With no soma current and no spike, the soma balance gives
    ``V_s = (g_c * V_d - 65) / (1 + g_c)``. The dendrite balance is then one
    equation in ``V_d``, written here from the model's documented equations and
    not from its integrator.
    """

    def balance(v_dend: float) -> float:
        v_soma = (neuron.g_coupling * v_dend - 65.0) / (1.0 + neuron.g_coupling)
        block = 1.0 / (1.0 + (neuron.mg_conc / 3.57) * math.exp(-0.062 * v_dend))
        i_nmda = neuron.g_nmda * glutamate * block * (v_dend - neuron.e_nmda)
        return -(v_dend + 65.0) - i_nmda + neuron.g_coupling * (v_soma - v_dend)

    low, high = -65.0, -50.0
    assert balance(low) > 0.0 > balance(high)
    for _ in range(80):
        middle = 0.5 * (low + high)
        if balance(middle) > 0.0:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


def test_glutamate_alone_depolarises_the_dendrite_to_its_balance_point() -> None:
    """An inward NMDA current raises the dendrite; it must not lower it."""
    neuron = DendriticNMDANeuron()
    expected = _steady_dendrite_without_soma_current(neuron, 1.0)
    assert expected > -65.0 + 5.0
    assert _run_spikes(neuron, 20_000, 0.0, 1.0) == 0
    assert neuron.v_dend == pytest.approx(expected, abs=1e-6)


def test_glutamate_turns_a_subthreshold_soma_current_into_firing() -> None:
    """Coincidence detection: neither input alone fires the soma, both do."""
    assert _run_spikes(DendriticNMDANeuron(), 20_000, 20.0, 0.0) == 0
    assert _run_spikes(DendriticNMDANeuron(), 20_000, 0.0, 1.0) == 0
    assert 50 <= _run_spikes(DendriticNMDANeuron(), 20_000, 20.0, 1.0) <= 70


def test_glutamate_raises_the_spike_count_at_a_firing_current() -> None:
    """At a current that fires alone, glutamate adds spikes by a stated margin."""
    without = _run_spikes(DendriticNMDANeuron(), 20_000, 40.0, 0.0)
    with_glutamate = _run_spikes(DendriticNMDANeuron(), 20_000, 40.0, 1.0)
    assert 180 <= without <= 200
    assert without + 10 <= with_glutamate <= without + 30


def test_baseline_euler_preserves_historical_comparison_path() -> None:
    """The Euler comparison path differs from RK4 in count and in final state."""
    rk4 = DendriticNMDANeuron()
    euler = DendriticNMDANeuron(integrator="baseline_euler")
    assert _run_spikes(rk4, 20_000, 30.0, 0.5) == 127
    assert _run_spikes(euler, 20_000, 30.0, 0.5) == 128
    assert (rk4.v_soma, rk4.v_dend) != (euler.v_soma, euler.v_dend)


def test_invalid_runtime_input_preserves_state() -> None:
    """A refused drive leaves both potentials as they were."""
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
    """A parameter made invalid after construction is refused without a state change."""
    neuron = DendriticNMDANeuron()
    for _ in range(10):
        neuron.step(50.0, 0.5)
    previous = (neuron.v_soma, neuron.v_dend)
    neuron.tau_dend = 0.0
    with pytest.raises(ValueError, match="tau_dend"):
        neuron.step(50.0, 0.5)
    assert (neuron.v_soma, neuron.v_dend) == previous


def test_reset_restores_both_compartments() -> None:
    """Reset returns the soma and the dendrite to -65 mV."""
    neuron = DendriticNMDANeuron()
    _ = _run_spikes(neuron, 500, 50.0, 0.5)
    neuron.reset()
    assert neuron.v_soma == -65.0
    assert neuron.v_dend == -65.0
