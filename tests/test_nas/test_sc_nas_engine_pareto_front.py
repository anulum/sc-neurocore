# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestParetoFront from former test_sc_nas_engine.py

"""Focused suite: TestParetoFront from former test_sc_nas_engine.py."""

from __future__ import annotations

from sc_neurocore.nas.sc_nas_engine import (
    DecorrelationStrategy,
    LayerConfig,
    NeuronType,
    pareto_front,
    SCCandidate,
)


class TestParetoFront:
    """Pareto fronts and crowding distances of candidates."""

    def test_empty_input(self) -> None:
        """No candidates give an empty front."""
        assert pareto_front([]) == []

    def test_single_candidate(self) -> None:
        """One candidate is its own front."""
        c = SCCandidate(
            layers=[LayerConfig(32, NeuronType.LIF, 256, DecorrelationStrategy.LFSR)],
            accuracy=0.9,
            total_luts=1000,
        )
        front = pareto_front([c])
        assert len(front) == 1

    def test_dominated_candidate_excluded(self) -> None:
        """A dominated candidate leaves the front."""
        a = SCCandidate(layers=[], accuracy=0.95, total_luts=500, total_power_mw=10)
        b = SCCandidate(layers=[], accuracy=0.90, total_luts=600, total_power_mw=15)
        front = pareto_front([a, b])
        assert len(front) == 1
        assert front[0] is a

    def test_non_dominated_both_kept(self) -> None:
        """Two candidates that do not dominate each other both stay."""
        a = SCCandidate(layers=[], accuracy=0.95, total_luts=1000, total_power_mw=20)
        b = SCCandidate(layers=[], accuracy=0.90, total_luts=500, total_power_mw=10)
        front = pareto_front([a, b])
        assert len(front) == 2

    def test_crowding_distance_assigned(self) -> None:
        """The front's boundary candidates get an infinite crowding distance."""
        candidates = [
            SCCandidate(layers=[], accuracy=0.90, total_luts=100, total_power_mw=5),
            SCCandidate(layers=[], accuracy=0.93, total_luts=200, total_power_mw=10),
            SCCandidate(layers=[], accuracy=0.96, total_luts=300, total_power_mw=15),
            SCCandidate(layers=[], accuracy=0.99, total_luts=400, total_power_mw=20),
        ]
        front = pareto_front(candidates)
        assert any(c.crowding_distance == float("inf") for c in front)

    def test_crowding_distance_interior(self) -> None:
        """Interior candidates get a positive crowding distance."""
        candidates = [
            SCCandidate(layers=[], accuracy=0.90, total_luts=100, total_power_mw=5),
            SCCandidate(layers=[], accuracy=0.93, total_luts=200, total_power_mw=10),
            SCCandidate(layers=[], accuracy=0.96, total_luts=300, total_power_mw=15),
            SCCandidate(layers=[], accuracy=0.99, total_luts=400, total_power_mw=20),
        ]
        front = pareto_front(candidates)
        interior = [c for c in front if c.crowding_distance != float("inf")]
        for c in interior:
            assert c.crowding_distance > 0
