# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestAkidaDynamics from former test_model_akida_neuron.py

"""Focused suite: TestAkidaDynamics from former test_model_akida_neuron.py."""

from __future__ import annotations

from tests.model_akida_neuron_support import *  # noqa: F403

class TestAkidaDynamics:
    def test_fires_with_large_weight(self):
        n = AkidaNeuron()
        assert n.step(100) == 1

    def test_accumulation_to_threshold(self):
        """Multiple small weights accumulate to threshold."""
        n = AkidaNeuron(threshold=100)
        spikes = 0
        for _ in range(50):
            spikes += n.step(30)
        assert spikes == 1  # fires once

    def test_never_fires_with_tiny_input(self):
        """Weight too small → int truncation → V never reaches threshold."""
        n = AkidaNeuron(threshold=100)
        # weight=1: ranks 0→0=1, 1→0, 2→0, ... V maxes at 1
        for _ in range(1000):
            n.step(1)
        assert n._spiked is False

    @pytest.mark.parametrize("weight", [20, 50, 100, 200])
    def test_weight_sweep(self, weight: int):
        n = AkidaNeuron()
        for _ in range(100):
            n.step(weight)
        assert isinstance(n.v, int)
