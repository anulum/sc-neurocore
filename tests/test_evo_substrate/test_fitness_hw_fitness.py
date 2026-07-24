# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestHWFitness from former test_fitness.py

"""Focused suite: TestHWFitness from former test_fitness.py."""

from __future__ import annotations

from tests.test_evo_substrate.fitness_support import *  # noqa: F403


class TestHWFitness:
    def test_report(self) -> None:
        r = HWFitnessReport("test_id", fpga_accuracy=0.9, fmax_mhz=200.0)
        assert r.hw_composite > 0

    def test_collector(self) -> None:
        col = HWFitnessCollector()
        col.submit(HWFitnessReport("g1", fpga_accuracy=0.8))
        assert col.total_reports == 1
        assert col.get("g1") is not None
        assert col.get("nonexistent") is None
