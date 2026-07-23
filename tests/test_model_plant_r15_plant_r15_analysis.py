# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPlantR15Analysis from former test_model_plant_r15.py

"""Focused suite: TestPlantR15Analysis from former test_model_plant_r15.py."""

from __future__ import annotations

from tests.model_plant_r15_support import *  # noqa: F403

class TestPlantR15Analysis:
    def test_spike_count(self):
        """At least 1 transient spike in a long run."""
        n = PlantR15Neuron()
        train = np.array([float(n.step(0.0)) for _ in range(50000)])
        assert spike_count(train) >= 1

    def test_spike_count_consistency(self):
        n = PlantR15Neuron()
        train = np.array([float(n.step(0.0)) for _ in range(50000)])
        assert spike_count(train) == int(train.sum())
