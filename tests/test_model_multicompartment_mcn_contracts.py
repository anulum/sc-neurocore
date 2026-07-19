# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Multicompartment MCN model contracts

"""Module-specific behavioural contracts for ``MulticompartmentMCNNeuron``."""

from __future__ import annotations

import math

import pytest


class TestMulticompartmentMCNNeuron:
    @pytest.fixture()
    def neuron(self):
        from sc_neurocore.neurons.models import MulticompartmentMCNNeuron

        return MulticompartmentMCNNeuron()

    def test_defaults_match_table_ii(self, neuron):
        """Default params from Table II of arXiv:2503.00713."""
        assert neuron.tau == 2.0
        assert neuron.tau_b == 2.0
        assert neuron.tau_a == 2.0
        assert neuron.g_ratio == 1.0
        assert neuron.beta == 1.0
        assert neuron.v_th == 1.0

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"tau": 0.0},
            {"tau_b": 0.0},
            {"tau_a": 0.0},
            {"g_ratio": -0.01},
            {"beta": 0.0},
            {"v_th": 0.0},
            {"dt": 0.0},
            {"u": float("nan")},
            {"v_basal": float("inf")},
            {"v_apical": float("-inf")},
        ],
    )
    def test_rejects_non_physical_multicompartment_parameters(self, kwargs):
        """Compartment dynamics require finite positive constants and finite state."""
        from sc_neurocore.neurons.models import MulticompartmentMCNNeuron

        with pytest.raises(ValueError):
            MulticompartmentMCNNeuron(**kwargs)

    @pytest.mark.parametrize("apical_voltage", [float("nan"), float("inf")])
    def test_rejects_non_finite_sigma_input(self, apical_voltage):
        """Apical sigmoid gate must fail closed on non-finite voltages."""
        from sc_neurocore.neurons.models import MulticompartmentMCNNeuron

        with pytest.raises(ValueError, match="x"):
            MulticompartmentMCNNeuron()._sigma(apical_voltage)

    @pytest.mark.parametrize(
        ("x_basal", "x_apical", "i_soma"),
        [(float("nan"), 0.0, 0.0), (0.0, float("inf"), 0.0), (0.0, 0.0, float("nan"))],
    )
    def test_rejects_non_finite_compartment_drive(self, x_basal, x_apical, i_soma):
        """Basal, apical, and somatic drives must be finite."""
        from sc_neurocore.neurons.models import MulticompartmentMCNNeuron

        with pytest.raises(ValueError, match="finite"):
            MulticompartmentMCNNeuron().step_compartments(x_basal, x_apical, i_soma)

    def test_step_returns_binary(self, neuron):
        s = neuron.step(0.5)
        assert s in (0, 1)

    def test_sigma_gating(self, neuron):
        """sigma(0) = 0.5, sigma(large) -> 1, sigma(-large) -> 0."""
        assert abs(neuron._sigma(0.0) - 0.5) < 1e-10
        assert neuron._sigma(10.0) > 0.99
        assert neuron._sigma(-10.0) < 0.01

    def test_basal_input_produces_spikes(self, neuron):
        spikes = sum(neuron.step(3.0) for _ in range(100))
        assert spikes > 0

    def test_threshold_boundary_accepts_one_ulp_roundoff(self, neuron):
        """The Heaviside equality boundary must survive binary64 RK4 roundoff."""
        assert neuron._threshold_reached(math.nextafter(neuron.v_th, 0.0))
        assert not neuron._threshold_reached(neuron.v_th - 1e-9)

    def test_apical_gating_modulates_firing(self):
        """High apical input (gate open) should increase firing vs no apical."""
        from sc_neurocore.neurons.models import MulticompartmentMCNNeuron

        # No apical: gate = sigma(0) = 0.5.
        n1 = MulticompartmentMCNNeuron()
        s1 = sum(n1.step_compartments(2.0, 0.0, 0.0) for _ in range(200))

        # Strong apical: gate = sigma(V_a) -> high.
        n2 = MulticompartmentMCNNeuron()
        s2 = sum(n2.step_compartments(2.0, 5.0, 0.0) for _ in range(200))

        assert s2 >= s1, "Apical gating should enhance or maintain firing"

    def test_soft_reset_to_zero(self, neuron):
        """After spike: U <- U * (1 - S) = 0."""
        for _ in range(50):
            s = neuron.step_compartments(3.0, 2.0, 0.0)
            if s == 1:
                assert neuron.u == 0.0
                return
        pytest.fail("No spike produced in 50 steps")

    def test_step_compartments_api(self, neuron):
        """step_compartments(x_basal, x_apical, i_soma) must accept 3 args."""
        s = neuron.step_compartments(1.0, 0.5, 0.2)
        assert s in (0, 1)

    def test_reset(self, neuron):
        for _ in range(50):
            neuron.step(3.0)
        neuron.reset()
        assert neuron.u == 0.0
        assert neuron.v_basal == 0.0
        assert neuron.v_apical == 0.0
