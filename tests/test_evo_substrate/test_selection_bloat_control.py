# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestBloatControl from former test_selection.py

"""Focused suite: TestBloatControl from former test_selection.py."""

from __future__ import annotations

from sc_neurocore.evo_substrate.genome import (
    Genome,
)
from sc_neurocore.evo_substrate.selection import (
    BloatPenalizer,
    compute_bloat,
)


class TestBloatControl:
    """Measuring and penalising genome bloat."""

    def test_compute_bloat(self) -> None:
        """A genome has parameters and a bloat score."""
        g = Genome()
        bm = compute_bloat(g)
        assert bm.total_params > 0
        assert bm.bloat_score > 0

    def test_penalizer_no_penalty(self) -> None:
        """A lean genome keeps its fitness."""
        bp = BloatPenalizer(threshold=100.0)
        g = Genome()
        assert bp.penalize(0.9, g) == 0.9

    def test_penalizer_reduces(self) -> None:
        """A bloated genome loses fitness."""
        bp = BloatPenalizer(threshold=0.01)
        g = Genome()
        assert bp.penalize(0.9, g) < 0.9

    def test_bloat_metrics_marks_large_genome_bloated(self) -> None:
        """A large genome is marked bloated."""
        g = Genome()
        g.topology.num_neurons = 512
        g.topology.num_layers = 8
        metrics = compute_bloat(g, baseline_neurons=4)
        assert metrics.is_bloated
