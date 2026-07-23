# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestFastBernoulli from former test_engine_v3_bitstream_kernels.py

"""Focused suite: TestFastBernoulli from former test_engine_v3_bitstream_kernels.py."""

from __future__ import annotations

from tests.engine_v3_bitstream_kernels_support import *  # noqa: F403

class TestFastBernoulli:
    """Tests for byte-threshold Bernoulli in forward_fast and batch_encode_numpy."""

    def test_forward_fast_deterministic(self) -> None:
        layer = v3.DenseLayer(16, 8, 1024, seed=42)
        inputs = [0.5] * 16
        out1 = layer.forward_fast(inputs, seed=100)
        out2 = layer.forward_fast(inputs, seed=100)
        np.testing.assert_array_equal(out1, out2)

    def test_forward_fast_output_range(self) -> None:
        layer = v3.DenseLayer(8, 4, 1024, seed=42)
        inputs = [0.3] * 8
        out = layer.forward_fast(inputs, seed=42)
        assert all(v >= 0.0 for v in out)

    def test_forward_fast_statistical_sanity(self) -> None:
        """forward_fast output should correlate with input probability."""
        layer = v3.DenseLayer(8, 4, 2048, seed=42)
        low_out = np.mean(layer.forward_fast([0.1] * 8, seed=42))
        high_out = np.mean(layer.forward_fast([0.9] * 8, seed=42))
        assert high_out > low_out, "Higher input probs should give higher output"

    def test_batch_encode_numpy_deterministic(self) -> None:
        probs = np.array([0.5, 0.5], dtype=np.float64)
        p1 = v3.batch_encode_numpy(probs, length=256, seed=42)
        p2 = v3.batch_encode_numpy(probs, length=256, seed=42)
        np.testing.assert_array_equal(p1, p2)

    def test_batch_encode_numpy_statistics(self) -> None:
        """Encoded bitstreams should have popcount proportional to probability."""
        probs = np.array([0.25, 0.75], dtype=np.float64)
        packed = v3.batch_encode_numpy(probs, length=10_000, seed=42)
        pc0 = sum(int(w).bit_count() for w in packed[0])
        pc1 = sum(int(w).bit_count() for w in packed[1])
        assert abs(pc0 / 10_000 - 0.25) < 0.04
        assert abs(pc1 / 10_000 - 0.75) < 0.04
