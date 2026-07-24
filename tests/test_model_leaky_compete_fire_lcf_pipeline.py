# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestLCFPipeline from former test_model_leaky_compete_fire.py

"""Focused suite: TestLCFPipeline from former test_model_leaky_compete_fire.py."""

from __future__ import annotations

from tests.model_leaky_compete_fire_support import *  # noqa: F403


class TestLCFPipeline:
    def test_population_incompatible(self):
        """v is list (multi-unit WTA) → Population._sync_voltages fails."""
        with pytest.raises((ValueError, TypeError)):
            Population(LeakyCompeteFireNeuron, n=5, label="lcf")

    def test_analysis_isolation(self):
        """Run in isolation, flatten spikes for analysis."""
        n = LeakyCompeteFireNeuron()
        total_spikes = 0
        for _ in range(5000):
            spikes = n.step(5.0)
            total_spikes += sum(spikes)
        assert total_spikes > 0
