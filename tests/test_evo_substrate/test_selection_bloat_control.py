# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestBloatControl from former test_selection.py

"""Focused suite: TestBloatControl from former test_selection.py."""

from __future__ import annotations

from tests.test_evo_substrate.selection_support import *  # noqa: F403

class TestBloatControl:
    def test_compute_bloat(self) -> None:
        g = Genome()
        bm = compute_bloat(g)
        assert bm.total_params > 0
        assert bm.bloat_score > 0

    def test_penalizer_no_penalty(self) -> None:
        bp = BloatPenalizer(threshold=100.0)
        g = Genome()
        assert bp.penalize(0.9, g) == 0.9

    def test_penalizer_reduces(self) -> None:
        bp = BloatPenalizer(threshold=0.01)
        g = Genome()
        assert bp.penalize(0.9, g) < 0.9

    def test_bloat_metrics_marks_large_genome_bloated(self) -> None:
        g = Genome()
        g.topology.num_neurons = 512
        g.topology.num_layers = 8
        metrics = compute_bloat(g, baseline_neurons=4)
        assert metrics.is_bloated
