# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestComplexityMetric from former test_statistics.py

"""Focused suite: TestComplexityMetric from former test_statistics.py."""

from __future__ import annotations

from tests.test_evo_substrate.statistics_support import *  # noqa: F403

class TestComplexityMetric:
    def test_complexity_positive(self) -> None:
        g = Genome()
        assert genome_complexity(g) > 0

    def test_bigger_is_more_complex(self) -> None:
        small = Genome()
        small.topology.num_neurons = 4
        big = Genome()
        big.topology.num_neurons = 512
        big.topology.num_layers = 8
        assert genome_complexity(big) > genome_complexity(small)

    def test_tracker(self) -> None:
        ct = ComplexityTracker()
        pop = [Organism(genome=Genome()) for _ in range(5)]
        ct.record(0, pop)
        ct.record(1, pop)
        assert len(ct.mean_trajectory) == 2

    def test_tracker_ignores_empty_population(self) -> None:
        ct = ComplexityTracker()
        ct.record(0, [])
        assert ct.mean_trajectory == []

    def test_tracker_requires_three_records_before_complexifying(self) -> None:
        ct = ComplexityTracker()
        low = Genome()
        low.topology.num_neurons = 4
        mid = Genome()
        mid.topology.num_neurons = 32
        high = Genome()
        high.topology.num_neurons = 128

        ct.record(0, [Organism(genome=low)])
        ct.record(1, [Organism(genome=mid)])
        assert not ct.is_complexifying
        ct.record(2, [Organism(genome=high)])

        assert ct.is_complexifying
