# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestWaveguideRouter from former test_bridges_photonic_noc.py

"""Focused suite: TestWaveguideRouter from former test_bridges_photonic_noc.py."""

from __future__ import annotations

from tests.bridges_photonic_noc_support import *  # noqa: F403

class TestWaveguideRouter:
    def test_route_two_nodes(self):
        router = WaveguideRouter()
        adj = np.array([[0, 1], [0, 0]], dtype=float)
        segments = router.route(adj)
        assert isinstance(segments, list)
        assert len(segments) >= 1
        assert all(isinstance(s, WaveguideSegment) for s in segments)

    def test_route_triangle(self):
        router = WaveguideRouter()
        adj = np.array([[0, 1, 1], [0, 0, 1], [0, 0, 0]], dtype=float)
        segments = router.route(adj)
        assert len(segments) >= 3

    def test_no_self_loops(self):
        router = WaveguideRouter()
        adj = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float)
        segments = router.route(adj)
        for seg in segments:
            assert seg.source != seg.target

    def test_route_empty_graph(self):
        router = WaveguideRouter()
        adj = np.zeros((5, 5), dtype=float)
        segments = router.route(adj)
        assert len(segments) == 0

    def test_segment_positive_length(self):
        router = WaveguideRouter()
        adj = np.array([[0, 1], [0, 0]], dtype=float)
        segments = router.route(adj)
        for seg in segments:
            assert seg.length_um > 0
