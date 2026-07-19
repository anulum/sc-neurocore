# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Dendritic NMDA model contracts

"""Module-specific behavioural contracts for ``DendriticNMDANeuron``."""

from __future__ import annotations

import math

import pytest


class TestDendriticNMDANeuron:
    @pytest.fixture()
    def neuron(self):
        from sc_neurocore.neurons.models import DendriticNMDANeuron

        return DendriticNMDANeuron()

    def test_defaults(self, neuron):
        assert neuron.v_soma == -65.0
        assert neuron.v_dend == -65.0
        assert neuron.mg_conc == 1.0

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"g_nmda": -0.01},
            {"e_nmda": float("nan")},
            {"mg_conc": -0.01},
            {"g_coupling": -0.01},
            {"tau_soma": 0.0},
            {"tau_dend": 0.0},
            {"theta": float("inf")},
            {"dt": 0.0},
            {"v_soma": float("nan")},
            {"v_dend": float("inf")},
        ],
    )
    def test_rejects_non_physical_nmda_parameters(self, kwargs):
        """NMDA compartment parameters must be finite and biophysically bounded."""
        from sc_neurocore.neurons.models import DendriticNMDANeuron

        with pytest.raises(ValueError):
            DendriticNMDANeuron(**kwargs)

    @pytest.mark.parametrize("voltage", [float("nan"), float("inf")])
    def test_rejects_non_finite_mg_block_voltage(self, voltage):
        """Voltage-dependent magnesium block must reject non-finite voltage."""
        from sc_neurocore.neurons.models import DendriticNMDANeuron

        with pytest.raises(ValueError, match="voltage"):
            DendriticNMDANeuron().mg_block(voltage)

    @pytest.mark.parametrize(
        ("i_soma", "glutamate"),
        [(float("nan"), 0.0), (0.0, float("inf")), (0.0, -0.01)],
    )
    def test_rejects_non_physical_nmda_drive(self, i_soma, glutamate):
        """Somatic current must be finite and glutamate must be finite non-negative."""
        from sc_neurocore.neurons.models import DendriticNMDANeuron

        with pytest.raises(ValueError):
            DendriticNMDANeuron().step(i_soma, glutamate)

    def test_step_returns_binary(self, neuron):
        s = neuron.step(10.0, 0.5)
        assert s in (0, 1)

    def test_mg_block_at_rest(self, neuron):
        """At -65 mV, Mg block should be strong (~0.06)."""
        b = neuron.mg_block(-65.0)
        assert 0.0 < b < 0.15, f"Mg block at -65mV = {b}, expected ~0.06"

    def test_mg_block_at_depolarised(self, neuron):
        """At 0 mV, Mg block should be relieved (~0.78)."""
        b = neuron.mg_block(0.0)
        assert b > 0.5, f"Mg block at 0mV = {b}, expected >0.5"

    def test_mg_block_formula(self, neuron):
        """B(V) = 1/(1 + [Mg]/3.57 * exp(-0.062*V)) — exact from Jahr & Stevens."""
        for v in [-80.0, -65.0, -40.0, -20.0, 0.0, 20.0]:
            expected = 1.0 / (1.0 + (1.0 / 3.57) * math.exp(-0.062 * v))
            actual = neuron.mg_block(v)
            assert abs(actual - expected) < 1e-12, f"Mg block at {v}mV: {actual} != {expected}"

    def test_spikes_with_strong_input(self, neuron):
        """Strong somatic current must produce spikes."""
        spikes = sum(neuron.step(50.0, 0.0) for _ in range(2000))
        assert spikes > 0

    def test_coincidence_detection(self):
        """NMDA requires BOTH glutamate AND depolarisation for full effect."""
        from sc_neurocore.neurons.models import DendriticNMDANeuron

        # Only soma current, no glutamate.
        n1 = DendriticNMDANeuron()
        for _ in range(500):
            n1.step(30.0, 0.0)
        v_no_glut = n1.v_dend

        # Soma current + glutamate.
        n2 = DendriticNMDANeuron()
        for _ in range(500):
            n2.step(30.0, 1.0)
        v_with_glut = n2.v_dend
        # With glutamate, dendrite should differ due to NMDA current.
        assert v_no_glut != v_with_glut

    def test_reset(self, neuron):
        for _ in range(100):
            neuron.step(30.0, 0.5)
        neuron.reset()
        assert neuron.v_soma == -65.0
        assert neuron.v_dend == -65.0
