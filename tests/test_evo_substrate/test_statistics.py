# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Evolutionary statistics and genome-comparison tests

"""Evolutionary statistics and genome-comparison tests."""

from __future__ import annotations

import pytest

from sc_neurocore.evo_substrate.genome import Genome
from sc_neurocore.evo_substrate.organism import Organism
from sc_neurocore.evo_substrate.statistics import (
    ComplexityTracker,
    EvoStatisticsTracker,
    GenerationStats,
    genome_complexity,
    genome_diff,
)


class TestEvoStatistics:
    def test_record(self) -> None:
        est = EvoStatisticsTracker()
        est.record(GenerationStats(1, 10, 0.7, 0.5, 0.3))
        est.record(GenerationStats(2, 12, 0.8, 0.6, 0.25))
        assert est.generations_tracked == 2

    def test_trajectory(self) -> None:
        est = EvoStatisticsTracker()
        est.record(GenerationStats(1, 10, 0.5, 0.3, 0.2))
        est.record(GenerationStats(2, 10, 0.8, 0.5, 0.3))
        assert est.fitness_trajectory == [0.5, 0.8]
        assert est.improvement_rate() == pytest.approx(0.3)

    def test_single_record_has_diversity_trajectory_and_zero_improvement(self) -> None:
        est = EvoStatisticsTracker()
        est.record(GenerationStats(1, 10, 0.5, 0.3, 0.2))

        assert est.diversity_trajectory == [0.2]
        assert est.improvement_rate() == 0.0


# ── Genome Diff Tests (Gap 19) ────────────────────────────────────────


class TestGenomeDiff:
    def test_identical(self) -> None:
        g = Genome()
        d = genome_diff(g, g)
        assert d.is_identical
        assert d.neuron_delta == 0

    def test_different(self) -> None:
        a = Genome()
        b = Genome()
        b.topology.num_neurons = 64
        d = genome_diff(a, b)
        assert not d.is_identical
        assert d.neuron_delta == 48


# ── Complexity Metric Tests (Gap 20) ──────────────────────────────────


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
