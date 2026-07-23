# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSuperSpikeLIFDynamics from former test_model_superspike_neuron.py

"""Focused suite: TestSuperSpikeLIFDynamics from former test_model_superspike_neuron.py."""

from __future__ import annotations

from tests.model_superspike_neuron_support import *  # noqa: F403

class TestSuperSpikeLIFDynamics:
    def test_voltage_leaky_integration(self):
        """v = alpha_m · v + I. Standard LIF with precomputed alpha."""
        n = SuperSpikeNeuron(v_threshold=100.0)
        n.step(0.5)
        assert abs(n.v - 0.5) < 1e-10  # v = alpha*0 + 0.5 = 0.5

    def test_spike_at_threshold(self):
        n = SuperSpikeNeuron()
        n.v = 0.9
        s = n.step(0.2)  # v = alpha*0.9 + 0.2 ≈ 1.014 ≥ 1.0
        assert s == 1

    def test_reset_on_spike(self):
        n = SuperSpikeNeuron()
        n.v = 0.9
        n.step(0.2)  # spike
        assert n.v == n.v_reset
