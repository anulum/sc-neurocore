# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Multicompartment MCN model contract tests

"""Real-surface tests for the Spiking-WM multicompartment MCN model."""

from __future__ import annotations

import math

import pytest

from sc_neurocore.neurons.models.multicompartment_mcn import MulticompartmentMCNNeuron


def _spike_count(neuron: MulticompartmentMCNNeuron, current: float, steps: int) -> int:
    return sum(neuron.step(current) for _ in range(steps))


class TestMulticompartmentMCNRK4:
    """Guards the candidate-first RK4 F4 contract for the MCN model."""

    def test_default_integrator_is_rk4(self) -> None:
        """The production MCN path uses candidate-first RK4 by default."""
        assert MulticompartmentMCNNeuron().integrator == "rk4"

    def test_unknown_integrator_rejected(self) -> None:
        """Only the production RK4 and explicit baseline regression paths exist."""
        with pytest.raises(ValueError, match="integrator"):
            MulticompartmentMCNNeuron(integrator="midpoint")  # type: ignore[arg-type]

    def test_rk4_and_baseline_euler_paths_diverge(self) -> None:
        """RK4 must not silently alias the retained Euler regression path."""
        rk4 = MulticompartmentMCNNeuron()
        euler = MulticompartmentMCNNeuron(integrator="baseline_euler")
        rk4_trace = [(rk4.step(3.0), round(rk4.u, 12), round(rk4.v_basal, 12)) for _ in range(40)]
        euler_trace = [
            (euler.step(3.0), round(euler.u, 12), round(euler.v_basal, 12)) for _ in range(40)
        ]
        assert rk4_trace != euler_trace

    def test_cross_backend_spike_anchor(self) -> None:
        """Pin the reference count used by the Rust, Go, Julia, and Mojo mirrors."""
        assert _spike_count(MulticompartmentMCNNeuron(), current=3.2, steps=200_000) == 49_999

    def test_non_finite_current_rejected_without_mutation(self) -> None:
        """Invalid runtime current fails before any compartment state changes."""
        neuron = MulticompartmentMCNNeuron()
        for _ in range(5):
            neuron.step(3.0)
        old_state = (neuron.u, neuron.v_basal, neuron.v_apical)
        with pytest.raises(ValueError, match="current"):
            neuron.step(math.inf)
        assert (neuron.u, neuron.v_basal, neuron.v_apical) == old_state

    def test_non_finite_runtime_state_rejected_before_mutation(self) -> None:
        """Corrupted in-memory state is rejected before candidate commit."""
        neuron = MulticompartmentMCNNeuron()
        neuron.v_basal = math.nan
        with pytest.raises(ValueError, match="v_basal"):
            neuron.step(3.0)
        assert math.isnan(neuron.v_basal)

    def test_apical_drive_increases_firing_anchor(self) -> None:
        """The public three-input API keeps apical gating behavior visible."""
        no_apical = MulticompartmentMCNNeuron()
        with_apical = MulticompartmentMCNNeuron()
        spikes_no = sum(no_apical.step_compartments(2.5, 0.0, 0.0) for _ in range(1_000))
        spikes_yes = sum(with_apical.step_compartments(2.5, 5.0, 0.0) for _ in range(1_000))
        assert spikes_yes >= spikes_no
        assert spikes_yes > 0
