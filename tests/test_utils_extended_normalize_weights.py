# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestNormalizeWeights from former test_utils_extended.py

"""Focused suite: TestNormalizeWeights from former test_utils_extended.py."""

from __future__ import annotations

from tests.utils_extended_support import *  # noqa: F403


class TestNormalizeWeights:
    def test_basic_normalization(self):
        w = np.array([0.0, 5.0, 10.0])
        result = normalize_weights(w)
        np.testing.assert_allclose(result, [0.0, 0.5, 1.0])

    def test_negative_weights(self):
        w = np.array([-10.0, 0.0, 10.0])
        result = normalize_weights(w)
        np.testing.assert_allclose(result, [0.0, 0.5, 1.0])

    def test_uniform_weights(self):
        """All equal weights should map to 0.5."""
        w = np.array([3.0, 3.0, 3.0])
        result = normalize_weights(w)
        np.testing.assert_allclose(result, [0.5, 0.5, 0.5])

    def test_single_element(self):
        w = np.array([7.0])
        result = normalize_weights(w)
        assert result[0] == 0.5

    def test_2d_array(self):
        w = np.array([[0.0, 1.0], [2.0, 3.0]])
        result = normalize_weights(w)
        assert result.min() == 0.0
        assert result.max() == 1.0
        assert result.shape == (2, 2)
