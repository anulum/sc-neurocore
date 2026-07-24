# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPlantR15Determinism from former test_model_plant_r15.py

"""Focused suite: TestPlantR15Determinism from former test_model_plant_r15.py."""

from __future__ import annotations

from tests.model_plant_r15_support import *  # noqa: F403


class TestPlantR15Determinism:
    def test_bit_exact_reproducibility(self):
        traces = []
        for _ in range(2):
            n = PlantR15Neuron()
            trace = [(n.step(1.0), n.v) for _ in range(200)]
            traces.append(trace)
        assert traces[0] == traces[1]
