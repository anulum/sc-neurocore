# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSuperSpikeNeuron from former test_model_superspike_neuron.py

"""Focused suite: TestSuperSpikeNeuron from former test_model_superspike_neuron.py."""

from __future__ import annotations

from tests.model_superspike_neuron_support import *  # noqa: F403


class TestSuperSpikeNeuron:
    def test_fires(self):
        from sc_neurocore.neurons.models.superspike_neuron import SuperSpikeNeuron

        n = SuperSpikeNeuron()
        assert sum(n.step(30.0) for _ in range(200)) > 0
