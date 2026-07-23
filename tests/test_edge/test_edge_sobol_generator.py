# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSobolGenerator from former test_edge.py

"""Focused suite: TestSobolGenerator from former test_edge.py."""

from __future__ import annotations

from edge_support import *  # noqa: F403

class TestSobolGenerator:
    def test_deterministic(self):
        a = SobolGenerator(seed=0)
        b = SobolGenerator(seed=0)
        for _ in range(100):
            assert a.step() == b.step()

    def test_unique_values(self):
        s = SobolGenerator()
        values = {s.step() for _ in range(1000)}
        assert len(values) > 500

    def test_encode_probability(self):
        s = SobolGenerator()
        bs = s.encode(32768, 10000)  # ~50% threshold
        popcount = sum(bin(int(w)).count("1") for w in bs)
        p = popcount / 10000
        assert abs(p - 0.5) < 0.05

    def test_different_seeds(self):
        a = SobolGenerator(seed=0x1234)
        b = SobolGenerator(seed=0x5678)
        assert a.step() != b.step()

    def test_reset(self):
        s = SobolGenerator()
        first = [s.step() for _ in range(10)]
        s.reset()
        second = [s.step() for _ in range(10)]
        assert first == second
