# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSpiNNaker2Refractory from former test_model_spinnaker2.py

"""Focused suite: TestSpiNNaker2Refractory from former test_model_spinnaker2.py."""

from __future__ import annotations

from tests.model_spinnaker2_support import *  # noqa: F403

class TestSpiNNaker2Refractory:
    def test_refractory_blocks(self):
        n = SpiNNaker2Neuron()
        for _ in range(1000):
            if n.step(500) == 1:
                assert n._refrac_count == n.refrac_steps
                s1 = n.step(500)
                s2 = n.step(500)
                assert s1 == 0 and s2 == 0
                return
        raise AssertionError("No spike")

    def test_refrac_count_decrements(self):
        n = SpiNNaker2Neuron()
        n._refrac_count = 2
        n.step(0)
        assert n._refrac_count == 1

    def test_max_rate_limited_by_refrac(self):
        """With refrac_steps=2: max rate = 1/(1+2) = 0.33 spikes/step."""
        n = SpiNNaker2Neuron(refrac_steps=2)
        outputs = [n.step(2000) for _ in range(3000)]
        spikes = outputs.count(1)
        max_rate = 3000 / (1 + 2)
        assert spikes <= max_rate + 10
