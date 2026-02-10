"""Tests for Phase 12: fused kernel, fast PRNG, and batched dense forward."""

from __future__ import annotations

import numpy as np

import sc_neurocore_engine as v3


class TestFusedKernel:
    """Verify fused encode+AND+popcount behavior and determinism."""

    def test_fused_matches_forward_fast(self):
        """Fused forward_fast output matches prepacked materialized encode path."""
        layer = v3.DenseLayer(8, 4, 512, seed=42)
        inputs = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9], dtype=np.float64)
        seed = 123

        fused = np.asarray(layer.forward_fast(inputs.tolist(), seed=seed), dtype=np.float64)
        packed = v3.batch_encode_numpy(inputs, length=512, seed=seed)
        materialized = np.asarray(layer.forward_prepacked_numpy(packed), dtype=np.float64)

        np.testing.assert_array_equal(fused, materialized)

    def test_fused_determinism(self):
        layer = v3.DenseLayer(16, 8, 1024, seed=42)
        inputs = [0.5] * 16
        out1 = layer.forward_fast(inputs, seed=777)
        out2 = layer.forward_fast(inputs, seed=777)
        np.testing.assert_array_equal(out1, out2)

    def test_fused_statistical_correctness(self):
        layer = v3.DenseLayer(16, 8, 2048, seed=42)
        low = np.mean(layer.forward_fast([0.1] * 16, seed=42))
        high = np.mean(layer.forward_fast([0.9] * 16, seed=42))
        assert high > low


class TestFastPRNG:
    """Verify xoshiro-backed fast paths remain deterministic and statistically sane."""

    def test_xoshiro_determinism(self):
        probs = np.array([0.2, 0.4, 0.6, 0.8], dtype=np.float64)
        a = v3.batch_encode_numpy(probs, length=1024, seed=2026)
        b = v3.batch_encode_numpy(probs, length=1024, seed=2026)
        np.testing.assert_array_equal(a, b)

    def test_xoshiro_statistical_quality(self):
        probs = np.array([0.35], dtype=np.float64)
        packed = v3.batch_encode_numpy(probs, length=10_000, seed=1337)
        count = sum(int(w).bit_count() for w in packed[0])
        measured = count / 10_000
        assert abs(measured - 0.35) < 0.03

    def test_forward_fast_determinism_new(self):
        layer = v3.DenseLayer(12, 6, 1024, seed=42)
        inputs = np.linspace(0.05, 0.95, 12, dtype=np.float64)
        a = layer.forward_fast(inputs.tolist(), seed=98765)
        b = layer.forward_fast(inputs.tolist(), seed=98765)
        np.testing.assert_array_equal(a, b)


class TestBatchForward:
    """Verify batched forward API correctness, shape and determinism."""

    def test_batch_vs_sequential(self):
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

    def test_batch_shape(self):
        layer = v3.DenseLayer(16, 8, 512, seed=42)
        inputs = np.random.RandomState(1).uniform(0, 1, (25, 16)).astype(np.float64)
        out = np.asarray(layer.forward_batch_numpy(inputs, seed=100))
        assert out.shape == (25, 8)

    def test_batch_determinism(self):
        layer = v3.DenseLayer(16, 8, 512, seed=42)
        inputs = np.random.RandomState(7).uniform(0, 1, (12, 16)).astype(np.float64)
        a = np.asarray(layer.forward_batch_numpy(inputs, seed=101))
        b = np.asarray(layer.forward_batch_numpy(inputs, seed=101))
        np.testing.assert_array_equal(a, b)

    def test_batch_numpy_output(self):
        layer = v3.DenseLayer(4, 2, 256, seed=42)
        inputs = np.random.RandomState(9).uniform(0, 1, (3, 4)).astype(np.float64)
        out = layer.forward_batch_numpy(inputs)
        assert isinstance(out, np.ndarray)
        assert out.dtype == np.float64


class TestPhase12Version:
    def test_version(self):
        assert v3.__version__ == "3.6.0"
