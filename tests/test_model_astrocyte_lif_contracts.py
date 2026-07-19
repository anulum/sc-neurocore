# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Astrocyte LIF model contracts

"""Module-specific behavioural contracts for ``AstrocyteLIFNeuron``."""

from __future__ import annotations

import pytest


class TestAstrocyteLIFNeuron:
    @pytest.fixture()
    def neuron(self):
        from sc_neurocore.neurons.models import AstrocyteLIFNeuron

        return AstrocyteLIFNeuron()

    def test_defaults(self, neuron):
        assert neuron.tau_ca == 500.0
        assert neuron.ca_thresh == 0.5
        assert neuron.g_glio == 2.0

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"tau_m": 0.0},
            {"tau_ca": 0.0},
            {"e_l": float("nan")},
            {"theta": float("nan")},
            {"theta": -70.0},
            {"v_reset": float("inf")},
            {"ca_delta": -0.01},
            {"ca_thresh": -0.01},
            {"g_glio": -0.01},
            {"dt": 0.0},
            {"v": float("nan")},
            {"ca": -0.01},
        ],
    )
    def test_rejects_non_physical_tripartite_parameters(self, kwargs):
        """Tripartite LIF parameters must be finite and physically bounded."""
        from sc_neurocore.neurons.models import AstrocyteLIFNeuron

        with pytest.raises(ValueError):
            AstrocyteLIFNeuron(**kwargs)

    @pytest.mark.parametrize("current", [float("nan"), float("inf")])
    def test_rejects_non_finite_external_current(self, current):
        """Membrane integration must fail closed on non-finite drive."""
        from sc_neurocore.neurons.models import AstrocyteLIFNeuron

        with pytest.raises(ValueError, match="i_ext"):
            AstrocyteLIFNeuron().step_with_pre(current, pre_spike=False)

    @pytest.mark.parametrize("pre_spike", [0, 1, "yes", None])
    def test_rejects_non_boolean_presynaptic_spike_flag(self, pre_spike):
        """Presynaptic event input must be an explicit boolean contract."""
        from sc_neurocore.neurons.models import AstrocyteLIFNeuron

        with pytest.raises(TypeError, match="pre_spike"):
            AstrocyteLIFNeuron().step_with_pre(0.0, pre_spike=pre_spike)

    def test_step_returns_binary(self, neuron):
        s = neuron.step(5.0)
        assert s in (0, 1)

    def test_calcium_rises_with_pre_spikes(self, neuron):
        """Presynaptic spikes must increase calcium."""
        ca_before = neuron.ca
        neuron.step_with_pre(0.0, pre_spike=True)
        assert neuron.ca > ca_before

    def test_calcium_decays_without_spikes(self, neuron):
        """Without pre_spikes, calcium decays toward 0 (tau_ca=500ms)."""
        neuron.ca = 1.0
        # 500ms / dt=0.1 = 5000 steps for one time constant.
        for _ in range(10000):
            neuron.step_with_pre(0.0, pre_spike=False)
        assert neuron.ca < 0.2

    def test_gliotransmitter_threshold(self):
        """I_glio = g_glio only when Ca > Ca_thresh."""
        from sc_neurocore.neurons.models import AstrocyteLIFNeuron

        n = AstrocyteLIFNeuron()
        # Build up calcium with sustained pre_spikes.
        for _ in range(100):
            n.step_with_pre(0.0, pre_spike=True)
        assert n.ca > n.ca_thresh, f"Ca={n.ca} should exceed thresh={n.ca_thresh}"

    def test_glial_feedback_increases_firing(self):
        """Gliotransmitter feedback should increase spike rate vs no feedback."""
        from sc_neurocore.neurons.models import AstrocyteLIFNeuron

        # Strong enough current to be near threshold.
        n_no = AstrocyteLIFNeuron()
        s_no = sum(n_no.step_with_pre(14.0, pre_spike=False) for _ in range(1000))

        n_glio = AstrocyteLIFNeuron()
        s_glio = sum(n_glio.step_with_pre(14.0, pre_spike=True) for _ in range(1000))

        assert s_glio >= s_no, "Glial feedback should not decrease firing"

    def test_reset(self, neuron):
        for _ in range(100):
            neuron.step_with_pre(10.0, pre_spike=True)
        neuron.reset()
        assert neuron.v == neuron.e_l
        assert neuron.ca == 0.0
