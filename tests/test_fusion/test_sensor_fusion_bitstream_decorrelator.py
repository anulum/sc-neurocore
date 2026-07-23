# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestBitstreamDecorrelator from former test_sensor_fusion.py

"""Focused suite: TestBitstreamDecorrelator from former test_sensor_fusion.py."""

from __future__ import annotations

from sensor_fusion_support import *  # noqa: F403

class TestBitstreamDecorrelator:
    def test_decorrelate_produces_different_streams(self):
        dec = BitstreamDecorrelator(seed=42)
        a = np.ones((8, 64), dtype=np.uint8)
        b = np.ones((8, 64), dtype=np.uint8)
        result = dec.decorrelate([a, b])
        assert not np.array_equal(result[0], result[1])

    def test_decorrelate_preserves_shape(self):
        dec = BitstreamDecorrelator(seed=42)
        a = np.ones((16, 128), dtype=np.uint8)
        result = dec.decorrelate([a])
        assert result[0].shape == (16, 128)

    def test_sobol_method(self):
        dec = BitstreamDecorrelator(seed=42)
        a = np.ones((4, 32), dtype=np.uint8)
        result = dec.decorrelate([a], method="sobol")
        assert result[0].shape == (4, 32)

    def test_scc_returns_bounded_value(self):
        dec = BitstreamDecorrelator(seed=42)
        rng = np.random.default_rng(0)
        a = rng.integers(0, 2, 100, dtype=np.uint8)
        b = rng.integers(0, 2, 100, dtype=np.uint8)
        scc = dec.measure_scc(a, b)
        assert -1.0 <= scc <= 1.0

    def test_scc_identical_streams(self):
        dec = BitstreamDecorrelator()
        a = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.uint8)
        scc = dec.measure_scc(a, a)
        assert abs(scc - 1.0) < 0.01

    def test_seed_zero_collides_and_resets_to_one(self):
        # base_seed 0 makes the i=0 mask seed land on 0, which the generator
        # must bump to 1 (a zero LFSR seed produces a degenerate all-zero mask).
        dec = BitstreamDecorrelator(seed=0)
        stream = np.ones((2, 4), dtype=np.uint8)
        result = dec.decorrelate([stream])
        assert result[0].shape == (2, 4)

    def test_scc_independent_streams_hit_numerator_floor(self):
        # Two all-zero streams give pa=pb=p_and=0, so the numerator collapses
        # to the |num|<eps floor and the coefficient is defined as 0.
        dec = BitstreamDecorrelator()
        zeros = np.zeros(8, dtype=np.float64)
        assert dec.measure_scc(zeros, zeros) == 0.0

    def test_scc_degenerate_denominator_returns_zero(self):
        # A non-binary stream breaks the bitstream invariant p_and<=min(pa,pb):
        # for a=[1.5,0.5] (pa=1.0) the denominator min(pa,pb)-pa*pb is exactly 0
        # while the numerator stays positive, exercising the |denom|<eps floor.
        dec = BitstreamDecorrelator()
        degenerate = np.array([1.5, 0.5], dtype=np.float64)
        assert dec.measure_scc(degenerate, degenerate) == 0.0
