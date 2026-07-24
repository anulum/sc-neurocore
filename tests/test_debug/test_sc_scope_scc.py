# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSCC from former test_sc_scope.py

"""Focused suite: TestSCC from former test_sc_scope.py."""

from __future__ import annotations

from sc_scope_support import *  # noqa: F403


class TestSCC:
    def test_identical_bitstreams(self):
        words = np.array([0xAAAA_AAAA] * 4, dtype=np.uint32)
        scc = compute_scc(words, words)
        assert abs(scc - 1.0) < 0.01

    def test_empty_bitstreams(self):
        scc = compute_scc(np.array([], dtype=np.uint32), np.array([], dtype=np.uint32))
        assert scc == 0.0

    def test_scc_range(self):
        rng = np.random.default_rng(42)
        a = rng.integers(0, 0xFFFF_FFFF, size=16, dtype=np.uint32)
        b = rng.integers(0, 0xFFFF_FFFF, size=16, dtype=np.uint32)
        scc = compute_scc(a, b)
        assert -1.0 <= scc <= 1.0
