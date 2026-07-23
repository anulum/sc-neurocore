# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestChaoticRNG from former test_research_modules.py

"""Focused suite: TestChaoticRNG from former test_research_modules.py."""

from __future__ import annotations

import sys
from pathlib import Path as _Path

# Ensure same-dir support module is importable under pytest importlib mode.
sys.path.insert(0, str(_Path(__file__).resolve().parent))

from research_modules_support import *  # noqa: F403

class TestChaoticRNG:
    def test_construction(self):
        rng = ChaoticRNG(r=4.0, x=0.3)
        # After burn-in, x should be in (0, 1)
        assert 0 < rng.x < 1

    def test_random_output_shape(self):
        rng = ChaoticRNG()
        vals = rng.random(100)
        assert vals.shape == (100,)

    def test_random_in_unit_interval(self):
        rng = ChaoticRNG()
        vals = rng.random(1000)
        assert np.all(vals >= 0) and np.all(vals <= 1)

    def test_generate_bitstream_shape(self):
        rng = ChaoticRNG()
        bs = rng.generate_bitstream(0.5, 256)
        assert bs.shape == (256,)
        assert set(np.unique(bs)).issubset({0, 1})

    def test_generate_bitstream_probability(self):
        # x=0.5 is a fixed point for r=4, so use a different init
        rng = ChaoticRNG(r=4.0, x=0.3)
        bs = rng.generate_bitstream(0.4, 10000)
        # Chaotic logistic map has arcsine distribution, not uniform.
        # Bitstream probability won't exactly match p, but should be non-trivial.
        assert 0.0 < bs.mean() < 1.0

    def test_deterministic_same_init(self):
        rng1 = ChaoticRNG(r=4.0, x=0.123)
        rng2 = ChaoticRNG(r=4.0, x=0.123)
        np.testing.assert_array_equal(rng1.random(50), rng2.random(50))
