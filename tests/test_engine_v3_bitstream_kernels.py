# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — v3 bitstream-kernel contracts

"""Contracts for v3 bitstream packing, encoding and pseudo-random kernels."""

from __future__ import annotations

import numpy as np
import pytest
import sc_neurocore_engine as v3


class TestSIMDPack:
    """Test SIMD-accelerated pack_bitstream_numpy correctness."""

    def test_pack_numpy_matches_list_pack(self) -> None:
        """SIMD pack must produce identical output to list pack."""
        rng = np.random.RandomState(42)
        bits = rng.randint(0, 2, 10_000).astype(np.uint8)
        packed_list = v3.pack_bitstream(bits.tolist())
        packed_numpy = np.asarray(v3.pack_bitstream_numpy(bits))
        np.testing.assert_array_equal(packed_list, packed_numpy)

    @pytest.mark.parametrize("length", [1, 63, 64, 65, 127, 128, 256, 1024, 4096])
    def test_pack_numpy_various_lengths(self, length: int) -> None:
        """SIMD pack handles all lengths including non-aligned."""
        rng = np.random.RandomState(42)
        bits = rng.randint(0, 2, length).astype(np.uint8)
        packed_list = v3.pack_bitstream(bits.tolist())
        packed_numpy = np.asarray(v3.pack_bitstream_numpy(bits))
        np.testing.assert_array_equal(packed_list, packed_numpy)

    def test_pack_numpy_deterministic(self) -> None:
        """Same input -> same output."""
        bits = np.array([1, 0, 1, 1, 0, 0, 1, 0] * 128, dtype=np.uint8)
        a = np.asarray(v3.pack_bitstream_numpy(bits))
        b = np.asarray(v3.pack_bitstream_numpy(bits))
        np.testing.assert_array_equal(a, b)

    def test_pack_unpack_roundtrip(self) -> None:
        """Pack->unpack roundtrip preserves bits."""
        rng = np.random.RandomState(42)
        bits = rng.randint(0, 2, 2048).astype(np.uint8)
        packed = v3.pack_bitstream_numpy(bits)
        unpacked = v3.unpack_bitstream_numpy(packed, len(bits))
        np.testing.assert_array_equal(bits, np.asarray(unpacked))


class TestSIMDBernoulliEncode:
    """Verify SIMD Bernoulli encoder statistical correctness and determinism."""

    def test_batch_encode_statistics(self) -> None:
        probs = np.array([0.25, 0.75], dtype=np.float64)
        packed = v3.batch_encode_numpy(probs, length=10_000, seed=42)
        pc0 = sum(int(w).bit_count() for w in packed[0])
        pc1 = sum(int(w).bit_count() for w in packed[1])
        assert abs(pc0 / 10_000 - 0.25) < 0.03
        assert abs(pc1 / 10_000 - 0.75) < 0.03

    def test_batch_encode_determinism(self) -> None:
        probs = np.array([0.15, 0.35, 0.55, 0.75], dtype=np.float64)
        a = v3.batch_encode_numpy(probs, length=1024, seed=1234)
        b = v3.batch_encode_numpy(probs, length=1024, seed=1234)
        np.testing.assert_array_equal(a, b)

    def test_dense_fast_correctness(self) -> None:
        layer = v3.DenseLayer(16, 8, 1024, seed=42)
        low = np.mean(layer.forward_fast([0.1] * 16, seed=22))
        high = np.mean(layer.forward_fast([0.9] * 16, seed=22))
        assert high > low


class TestFastPRNG:
    """Verify xoshiro-backed fast paths remain deterministic and statistically sane."""

    def test_xoshiro_determinism(self) -> None:
        probs = np.array([0.2, 0.4, 0.6, 0.8], dtype=np.float64)
        a = v3.batch_encode_numpy(probs, length=1024, seed=2026)
        b = v3.batch_encode_numpy(probs, length=1024, seed=2026)
        np.testing.assert_array_equal(a, b)

    def test_xoshiro_statistical_quality(self) -> None:
        probs = np.array([0.35], dtype=np.float64)
        packed = v3.batch_encode_numpy(probs, length=10_000, seed=1337)
        count = sum(int(w).bit_count() for w in packed[0])
        measured = count / 10_000
        assert abs(measured - 0.35) < 0.03

    def test_forward_fast_determinism_new(self) -> None:
        layer = v3.DenseLayer(12, 6, 1024, seed=42)
        inputs = np.linspace(0.05, 0.95, 12, dtype=np.float64)
        a = layer.forward_fast(inputs.tolist(), seed=98765)
        b = layer.forward_fast(inputs.tolist(), seed=98765)
        np.testing.assert_array_equal(a, b)


class TestParallelBatchEncodeNumpy:
    """Tests for parallel batch_encode_numpy."""

    def test_shape_and_dtype(self) -> None:
        probs = np.array([0.3, 0.5, 0.7], dtype=np.float64)
        packed = v3.batch_encode_numpy(probs, length=1024, seed=42)
        assert packed.shape == (3, 16)
        assert packed.dtype == np.uint64

    def test_deterministic(self) -> None:
        probs = np.array([0.5, 0.5], dtype=np.float64)
        p1 = v3.batch_encode_numpy(probs, length=256, seed=42)
        p2 = v3.batch_encode_numpy(probs, length=256, seed=42)
        np.testing.assert_array_equal(p1, p2)

    def test_different_seed(self) -> None:
        probs = np.array([0.5], dtype=np.float64)
        p1 = v3.batch_encode_numpy(probs, length=1024, seed=1)
        p2 = v3.batch_encode_numpy(probs, length=1024, seed=2)
        assert not np.array_equal(p1, p2)

    def test_popcount_statistics(self) -> None:
        """Encoded bitstreams should have popcount proportional to probability."""
        probs = np.array([0.25, 0.75], dtype=np.float64)
        packed = v3.batch_encode_numpy(probs, length=10_000, seed=42)
        pc0 = sum(int(w).bit_count() for w in packed[0])
        pc1 = sum(int(w).bit_count() for w in packed[1])
        assert abs(pc0 / 10_000 - 0.25) < 0.03
        assert abs(pc1 / 10_000 - 0.75) < 0.03

    def test_pipeline_encode_then_forward(self) -> None:
        """batch_encode_numpy -> forward_prepacked remains valid."""
        layer = v3.DenseLayer(4, 2, 1024, seed=42)
        probs = np.array([0.3, 0.5, 0.7, 0.9], dtype=np.float64)
        packed = v3.batch_encode_numpy(probs, length=1024, seed=55)
        out = layer.forward_prepacked(packed)
        assert len(out) == 2
        assert all(0.0 <= v <= 4.0 for v in out)

    def test_empty_probs(self) -> None:
        probs = np.array([], dtype=np.float64)
        packed = v3.batch_encode_numpy(probs, length=64, seed=42)
        assert packed.shape[0] == 0


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
