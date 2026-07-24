# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestCrossModalAttention from former test_sensor_fusion.py

"""Focused suite: TestCrossModalAttention from former test_sensor_fusion.py."""

from __future__ import annotations

from sensor_fusion_support import *  # noqa: F403


class TestCrossModalAttention:
    def test_attend_preserves_shape(self):
        attn = CrossModalAttention(num_channels=8, seed=42)
        q = np.ones((8, 32), dtype=np.uint8)
        k = np.ones((8, 32), dtype=np.uint8)
        v = np.ones((8, 32), dtype=np.uint8)
        result = attn.attend(q, k, v)
        assert result.shape == (8, 32)

    def test_attend_zero_query_zero_output(self):
        attn = CrossModalAttention(num_channels=4, seed=42)
        q = np.zeros((4, 16), dtype=np.uint8)
        k = np.ones((4, 16), dtype=np.uint8)
        v = np.ones((4, 16), dtype=np.uint8)
        result = attn.attend(q, k, v)
        assert np.sum(result) == 0

    def test_sc_and_multiplication(self):
        attn = CrossModalAttention(num_channels=4)
        a = np.array([[1, 0, 1, 1]], dtype=np.uint8)
        b = np.array([[1, 1, 0, 1]], dtype=np.uint8)
        result = attn._sc_and(a, b)
        np.testing.assert_array_equal(result, [[1, 0, 0, 1]])
