# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestLiveAnalyzer from former test_sc_scope.py

"""Focused suite: TestLiveAnalyzer from former test_sc_scope.py."""

from __future__ import annotations

from sc_neurocore.debug.sc_scope import (
    LiveAnalyzer,
)
from sc_scope_support import (
    _sample,
)


class TestLiveAnalyzer:
    """Live per-layer analysis of samples."""

    def test_ingest(self) -> None:
        """Ingested samples are counted."""
        la = LiveAnalyzer(num_layers=2)
        la.ingest(_sample(layer=0))
        la.ingest(_sample(layer=1))
        assert la.total_samples == 2

    def test_layer_stats(self) -> None:
        """A layer reports its mean density and sample count."""
        la = LiveAnalyzer(num_layers=1)
        for _ in range(10):
            la.ingest(_sample(layer=0, density=0.5))
        stats = la.layer_stats(0)
        assert "mean_density" in stats
        assert stats["sample_count"] == 10

    def test_all_stats(self) -> None:
        """Statistics are reported for every layer."""
        la = LiveAnalyzer(num_layers=3)
        for lid in range(3):
            la.ingest(_sample(layer=lid))
        all_s = la.all_stats()
        assert len(all_s) == 3

    def test_unknown_layer(self) -> None:
        """An unknown layer has no statistics."""
        la = LiveAnalyzer(num_layers=1)
        assert la.layer_stats(99) == {}
