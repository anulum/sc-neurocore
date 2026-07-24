# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestWaveguideRouter from former test_photonic_noc.py

"""Focused suite: TestWaveguideRouter from former test_photonic_noc.py."""

from __future__ import annotations

from photonic_noc_support import *  # noqa: F403


class TestWaveguideRouter:
    """Waveguide routing tests."""

    def test_route_produces_segments(self, simple_adjacency: np.ndarray) -> None:
        router = WaveguideRouter()
        segments = router.route(simple_adjacency)
        assert len(segments) > 0
        assert all(isinstance(s, WaveguideSegment) for s in segments)

    def test_loss_positive(self, simple_adjacency: np.ndarray) -> None:
        router = WaveguideRouter()
        segments = router.route(simple_adjacency)
        for s in segments:
            assert s.loss_db >= 0

    def test_no_self_loops(self, simple_adjacency: np.ndarray) -> None:
        router = WaveguideRouter()
        segments = router.route(simple_adjacency)
        for s in segments:
            assert s.source != s.target

    def test_custom_pitch(self) -> None:
        adj = np.array([[0.0, 1.0], [1.0, 0.0]])
        r1 = WaveguideRouter(pitch_um=100.0).route(adj)
        r2 = WaveguideRouter(pitch_um=500.0).route(adj)
        assert r1[0].length_um < r2[0].length_um
