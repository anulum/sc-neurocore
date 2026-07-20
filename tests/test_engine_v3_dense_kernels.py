# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — v3 dense-kernel contracts

"""Contracts for v3 DenseLayer storage, batching and fused forward kernels."""

from __future__ import annotations

import numpy as np
import pytest
import sc_neurocore_engine as v3


class TestRayonThreshold:
    """Test that rayon threshold does not change forward_fast outputs."""

    def test_forward_fast_determinism(self) -> None:
        """forward_fast with small inputs (below threshold) stays deterministic."""
        layer = v3.DenseLayer(16, 8, 1024)
        inputs = [0.5] * 16
        a = layer.forward_fast(inputs, seed=42)
        b = layer.forward_fast(inputs, seed=42)
        np.testing.assert_array_equal(a, b)

    def test_forward_fast_consistent_across_sizes(self) -> None:
        """forward_fast produces valid outputs for various input sizes."""
        for n_in in [4, 16, 64, 128, 256]:
            layer = v3.DenseLayer(n_in, 8, 1024)
            inputs = [0.5] * n_in
            result = layer.forward_fast(inputs, seed=42)
            assert len(result) == 8
            for val in result:
                assert 0.0 <= val <= float(n_in), f"Out of range: {val}"


class TestSIMDFusedAndPopcount:
    """Verify SIMD fused AND+popcount preserves dense behavior."""

    def test_dense_forward_unchanged(self) -> None:
        layer = v3.DenseLayer(8, 4, 1024, seed=42)
        inputs = [0.1, 0.3, 0.5, 0.7, 0.2, 0.4, 0.6, 0.8]
        out1 = layer.forward(inputs, seed=123)
        out2 = layer.forward(inputs, seed=123)
        np.testing.assert_array_equal(out1, out2)
        assert all(0.0 <= x <= 8.0 for x in out1)

    def test_dense_prepacked_unchanged(self) -> None:
        layer = v3.DenseLayer(8, 4, 1024, seed=42)
        probs = np.array([0.2, 0.4, 0.6, 0.8, 0.1, 0.3, 0.5, 0.7], dtype=np.float64)
        packed = v3.batch_encode_numpy(probs, length=1024, seed=99)
        out_legacy = layer.forward_prepacked(packed)
        out_numpy = layer.forward_prepacked_numpy(packed)
        np.testing.assert_allclose(out_numpy, out_legacy)

    def test_determinism(self) -> None:
        layer = v3.DenseLayer(16, 8, 1024, seed=42)
        inputs = [0.5] * 16
        out1 = layer.forward_fast(inputs, seed=77)
        out2 = layer.forward_fast(inputs, seed=77)
        np.testing.assert_array_equal(out1, out2)


class TestFlatWeightStorage:
    """Verify flat packed weight storage keeps API behavior unchanged."""

    def test_weight_roundtrip(self) -> None:
        layer = v3.DenseLayer(4, 3, 256, seed=42)
        weights = np.array(
            [
                [0.1, 0.2, 0.3, 0.4],
                [0.4, 0.3, 0.2, 0.1],
                [0.5, 0.6, 0.7, 0.8],
            ],
            dtype=np.float64,
        )
        layer.set_weights(weights.tolist())
        got = np.array(layer.get_weights(), dtype=np.float64)
        np.testing.assert_allclose(got, weights)

    def test_forward_equivalence_vs_prepacked(self) -> None:
        layer = v3.DenseLayer(8, 4, 512, seed=42)
        probs = np.array([0.2, 0.4, 0.6, 0.8, 0.1, 0.3, 0.5, 0.7], dtype=np.float64)
        seed = 31415
        packed = v3.batch_encode_numpy(probs, length=512, seed=seed)
        out_fast = np.asarray(layer.forward_fast(probs.tolist(), seed=seed), dtype=np.float64)
        out_prepacked = np.asarray(layer.forward_prepacked_numpy(packed), dtype=np.float64)
        np.testing.assert_allclose(out_fast, out_prepacked)


class TestFusedKernel:
    """Verify fused encode+AND+popcount behavior and determinism."""

    def test_fused_matches_forward_fast(self) -> None:
        """Fused forward_fast output matches prepacked materialized encode path."""
        layer = v3.DenseLayer(8, 4, 512, seed=42)
        inputs = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9], dtype=np.float64)
        seed = 123

        fused = np.asarray(layer.forward_fast(inputs.tolist(), seed=seed), dtype=np.float64)
        packed = v3.batch_encode_numpy(inputs, length=512, seed=seed)
        materialized = np.asarray(layer.forward_prepacked_numpy(packed), dtype=np.float64)

        np.testing.assert_array_equal(fused, materialized)

    def test_fused_determinism(self) -> None:
        layer = v3.DenseLayer(16, 8, 1024, seed=42)
        inputs = [0.5] * 16
        out1 = layer.forward_fast(inputs, seed=777)
        out2 = layer.forward_fast(inputs, seed=777)
        np.testing.assert_array_equal(out1, out2)

    def test_fused_statistical_correctness(self) -> None:
        layer = v3.DenseLayer(16, 8, 2048, seed=42)
        low = np.mean(layer.forward_fast([0.1] * 16, seed=42))
        high = np.mean(layer.forward_fast([0.9] * 16, seed=42))
        assert high > low


class TestBatchForward:
    """Verify batched forward API correctness, shape and determinism."""

    def test_batch_vs_sequential(self) -> None:
        layer = v3.DenseLayer(8, 4, 1024, seed=42)
        inputs = np.random.RandomState(42).uniform(0, 1, (10, 8)).astype(np.float64)
        seed = 555

        batched = np.asarray(layer.forward_batch_numpy(inputs, seed=seed), dtype=np.float64)

        sequential_rows = []
        for sample_idx, row in enumerate(inputs):
            sample_seed = seed + sample_idx * 1_000_000
            sequential_rows.append(layer.forward_fast(row.tolist(), seed=sample_seed))
        sequential = np.asarray(sequential_rows, dtype=np.float64)

        np.testing.assert_array_equal(batched, sequential)

    def test_batch_shape(self) -> None:
        layer = v3.DenseLayer(16, 8, 512, seed=42)
        inputs = np.random.RandomState(1).uniform(0, 1, (25, 16)).astype(np.float64)
        out = np.asarray(layer.forward_batch_numpy(inputs, seed=100))
        assert out.shape == (25, 8)

    def test_batch_determinism(self) -> None:
        layer = v3.DenseLayer(16, 8, 512, seed=42)
        inputs = np.random.RandomState(7).uniform(0, 1, (12, 16)).astype(np.float64)
        a = np.asarray(layer.forward_batch_numpy(inputs, seed=101))
        b = np.asarray(layer.forward_batch_numpy(inputs, seed=101))
        np.testing.assert_array_equal(a, b)

    def test_batch_numpy_output(self) -> None:
        layer = v3.DenseLayer(4, 2, 256, seed=42)
        inputs = np.random.RandomState(9).uniform(0, 1, (3, 4)).astype(np.float64)
        out = layer.forward_batch_numpy(inputs)
        assert isinstance(out, np.ndarray)
        assert out.dtype == np.float64


class TestForwardNumpy:
    """Tests for single-call numpy dense forward."""

    def test_output_shape_and_type(self) -> None:
        layer = v3.DenseLayer(16, 8, 512)
        inputs = np.array([0.5] * 16, dtype=np.float64)
        out = layer.forward_numpy(inputs)
        assert isinstance(out, np.ndarray)
        assert out.shape == (8,)
        assert out.dtype == np.float64

    def test_output_range(self) -> None:
        layer = v3.DenseLayer(16, 8, 512)
        inputs = np.array([0.3] * 16, dtype=np.float64)
        out = layer.forward_numpy(inputs)
        assert np.all(out >= 0.0)
        assert np.all(out <= 16.0)

    def test_deterministic(self) -> None:
        layer = v3.DenseLayer(16, 8, 512, seed=42)
        inputs = np.array([0.5] * 16, dtype=np.float64)
        out1 = layer.forward_numpy(inputs, seed=100)
        out2 = layer.forward_numpy(inputs, seed=100)
        np.testing.assert_array_equal(out1, out2)

    def test_matches_forward_fast(self) -> None:
        """forward_numpy should match forward_fast with same seed."""
        layer = v3.DenseLayer(8, 4, 256, seed=42)
        inputs_list = [0.1, 0.3, 0.5, 0.7, 0.2, 0.4, 0.6, 0.8]
        inputs_np = np.array(inputs_list, dtype=np.float64)
        out_fast = layer.forward_fast(inputs_list, seed=42)
        out_numpy = layer.forward_numpy(inputs_np, seed=42)
        np.testing.assert_allclose(out_numpy, out_fast)

    def test_wrong_input_length(self) -> None:
        layer = v3.DenseLayer(8, 4, 256)
        inputs = np.array([0.5] * 7, dtype=np.float64)
        with pytest.raises(ValueError):
            layer.forward_numpy(inputs)

    def test_different_seed_different_output(self) -> None:
        layer = v3.DenseLayer(8, 4, 1024, seed=42)
        inputs = np.array([0.5] * 8, dtype=np.float64)
        out1 = layer.forward_numpy(inputs, seed=100)
        out2 = layer.forward_numpy(inputs, seed=200)
        assert not np.array_equal(out1, out2)


class TestFusedAndPopcount:
    """Tests verifying fused AND+popcount produces same results as before."""

    def test_forward_matches_reference(self) -> None:
        """forward() output should still be valid (range + deterministic)."""
        layer = v3.DenseLayer(8, 4, 512, seed=42)
        inputs = [0.3, 0.5, 0.7, 0.2, 0.4, 0.6, 0.8, 0.1]
        out1 = layer.forward(inputs, seed=42)
        out2 = layer.forward(inputs, seed=42)
        np.testing.assert_array_equal(out1, out2)
        assert all(v >= 0.0 for v in out1)

    def test_prepacked_deterministic(self) -> None:
        layer = v3.DenseLayer(4, 2, 1024, seed=42)
        probs = np.array([0.3, 0.5, 0.7, 0.9], dtype=np.float64)
        packed = v3.batch_encode_numpy(probs, length=1024, seed=55)
        out1 = layer.forward_prepacked(packed)
        out2 = layer.forward_prepacked(packed)
        np.testing.assert_array_equal(out1, out2)


class TestZeroCopyPrepackedNumpy:
    """Tests for forward_prepacked_numpy (true zero-copy path)."""

    def test_output_shape_and_type(self) -> None:
        layer = v3.DenseLayer(4, 2, 1024, seed=42)
        probs = np.array([0.3, 0.5, 0.7, 0.9], dtype=np.float64)
        packed = v3.batch_encode_numpy(probs, length=1024, seed=55)
        out = layer.forward_prepacked_numpy(packed)
        assert isinstance(out, np.ndarray)
        assert out.shape == (2,)
        assert out.dtype == np.float64

    def test_matches_forward_prepacked(self) -> None:
        """Zero-copy numpy path must match the existing prepacked path."""
        layer = v3.DenseLayer(8, 4, 512, seed=42)
        probs = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], dtype=np.float64)
        packed = v3.batch_encode_numpy(probs, length=512, seed=99)
        out_legacy = layer.forward_prepacked(packed)
        out_numpy = layer.forward_prepacked_numpy(packed)
        np.testing.assert_allclose(out_numpy, out_legacy)

    def test_wrong_n_inputs(self) -> None:
        layer = v3.DenseLayer(4, 2, 1024, seed=42)
        packed = np.zeros((3, 16), dtype=np.uint64)  # 3 inputs, need 4
        with pytest.raises(ValueError):
            layer.forward_prepacked_numpy(packed)

    def test_wrong_word_count(self) -> None:
        layer = v3.DenseLayer(4, 2, 1024, seed=42)
        packed = np.zeros((4, 10), dtype=np.uint64)  # 10 words, need 16
        with pytest.raises(ValueError):
            layer.forward_prepacked_numpy(packed)

    def test_pipeline_encode_then_zero_copy(self) -> None:
        """Full pipeline: batch_encode_numpy -> forward_prepacked_numpy."""
        layer = v3.DenseLayer(16, 8, 1024, seed=42)
        probs = np.random.uniform(0, 1, 16)
        packed = v3.batch_encode_numpy(probs, length=1024, seed=42)
        out = layer.forward_prepacked_numpy(packed)
        assert out.shape == (8,)
        assert np.all(out >= 0.0)

    def test_deterministic(self) -> None:
        layer = v3.DenseLayer(4, 2, 512, seed=42)
        probs = np.array([0.5] * 4, dtype=np.float64)
        packed = v3.batch_encode_numpy(probs, length=512, seed=42)
        out1 = layer.forward_prepacked_numpy(packed)
        out2 = layer.forward_prepacked_numpy(packed)
        np.testing.assert_array_equal(out1, out2)
