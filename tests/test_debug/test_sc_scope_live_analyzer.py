# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestLiveAnalyzer from former test_sc_scope.py

"""Focused suite: TestLiveAnalyzer from former test_sc_scope.py."""

from __future__ import annotations

from sc_scope_support import *  # noqa: F403

class TestLiveAnalyzer:
    def test_ingest(self):
        la = LiveAnalyzer(num_layers=2)
        la.ingest(_sample(layer=0))
        la.ingest(_sample(layer=1))
        assert la.total_samples == 2

    def test_layer_stats(self):
        la = LiveAnalyzer(num_layers=1)
        for _ in range(10):
            la.ingest(_sample(layer=0, density=0.5))
        stats = la.layer_stats(0)
        assert "mean_density" in stats
        assert stats["sample_count"] == 10

    def test_all_stats(self):
        la = LiveAnalyzer(num_layers=3)
        for lid in range(3):
            la.ingest(_sample(layer=lid))
        all_s = la.all_stats()
        assert len(all_s) == 3

    def test_unknown_layer(self):
        la = LiveAnalyzer(num_layers=1)
        assert la.layer_stats(99) == {}
