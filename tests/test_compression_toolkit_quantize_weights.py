# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestQuantizeWeights from former test_compression_toolkit.py

"""Focused suite: TestQuantizeWeights from former test_compression_toolkit.py."""

from __future__ import annotations

from tests.compression_toolkit_support import *  # noqa: F403

class TestQuantizeWeights:
    def test_8bit_symmetric(self):
        w = [np.array([[0.123456, -0.789012, 0.5]])]
        q = quantize_weights(w, bits=8, symmetric=True)
        assert len(q) == 1
        assert q[0].shape == w[0].shape
        assert not np.array_equal(q[0], w[0])

    def test_quantization_reduces_unique_values(self):
        rng = np.random.RandomState(42)
        w = [rng.randn(100, 100)]
        q = quantize_weights(w, bits=4)
        assert len(np.unique(q[0])) < len(np.unique(w[0]))

    def test_asymmetric_quantization(self):
        w = [np.array([[0.1, 0.5, 0.9]])]
        q = quantize_weights(w, bits=8, symmetric=False)
        assert q[0].shape == w[0].shape

    def test_bits_clamped(self):
        w = [np.array([[1.0]])]
        q_low = quantize_weights(w, bits=1)
        q_high = quantize_weights(w, bits=32)
        assert len(q_low) == 1
        assert len(q_high) == 1

    def test_multiple_layers(self):
        w = [np.random.randn(5, 5), np.random.randn(3, 5)]
        q = quantize_weights(w, bits=8)
        assert len(q) == 2
        assert q[0].shape == (5, 5)
        assert q[1].shape == (3, 5)
