# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Source/config provenance header

# Tests for sc_neurocore.compression (pruning + quantization)

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.compression import (
    prune_weights,
    prune_neurons,
    prune_stochastic,
    PruningReport,
    quantize_delays,
    quantize_weights,
)


class TestPruneWeights:
    def test_magnitude_pruning(self):
        w = [np.array([[0.5, 0.001, -0.8], [-0.002, 0.3, 0.0]])]
        pruned, report = prune_weights(w, threshold=0.01)
        assert pruned[0][0, 1] == 0.0
        assert pruned[0][1, 0] == 0.0
        assert pruned[0][0, 0] == 0.5

    def test_sparsity_calculation(self):
        w = [np.array([[0.1, 0.001], [0.002, 0.5]])]
        _, report = prune_weights(w, threshold=0.01)
        assert isinstance(report, PruningReport)
        assert report.original_params == 4
        assert report.pruned_params == 2
        assert report.remaining_params == 2
        assert report.sparsity == pytest.approx(0.5)

    def test_percentile_pruning(self):
        rng = np.random.RandomState(42)
        w = [rng.randn(10, 10)]
        pruned, report = prune_weights(w, threshold=50.0, method="percentile")
        assert report.sparsity > 0.3

    def test_no_pruning_below_threshold(self):
        w = [np.array([[1.0, 2.0], [3.0, 4.0]])]
        pruned, report = prune_weights(w, threshold=0.01)
        assert report.pruned_params == 0
        np.testing.assert_array_equal(pruned[0], w[0])

    def test_multiple_layers(self):
        w = [np.ones((3, 3)) * 0.5, np.ones((2, 3)) * 0.001]
        pruned, report = prune_weights(w, threshold=0.01)
        assert report.original_params == 15
        assert report.pruned_params == 6
        np.testing.assert_array_equal(pruned[0], w[0])
        np.testing.assert_array_equal(pruned[1], np.zeros((2, 3)))

    def test_does_not_modify_original(self):
        w = [np.array([[0.5, 0.001]])]
        original_copy = w[0].copy()
        prune_weights(w, threshold=0.01)
        np.testing.assert_array_equal(w[0], original_copy)


class TestPruneNeurons:
    def test_structural_pruning_by_weight_norm(self):
        w1 = np.array([[1.0, 0.5], [0.0001, 0.0001], [0.8, 0.3]])
        w2 = np.array([[0.5, 0.3, 0.1]])
        pruned, report = prune_neurons([w1, w2], activity_threshold=0.001)
        assert report.pruned_neurons == 1
        assert pruned[0].shape[0] == 2
        assert pruned[1].shape[1] == 2

    def test_no_pruning_when_all_active(self):
        w = [np.ones((3, 3))]
        pruned, report = prune_neurons(w, activity_threshold=0.001)
        assert report.pruned_neurons == 0
        np.testing.assert_array_equal(pruned[0], w[0])

    def test_with_firing_rates(self):
        w = [np.ones((4, 3))]
        rates = [np.array([0.1, 0.0, 0.05, 0.0002])]
        pruned, report = prune_neurons(w, firing_rates=rates, activity_threshold=0.001)
        # Neurons with rate <= 0.001 are pruned: index 1 (0.0) and index 3 (0.0002)
        assert report.pruned_neurons == 2
        assert pruned[0].shape[0] == 2

    def test_report_fields(self):
        w = [np.eye(3)]
        _, report = prune_neurons(w)
        assert report.original_neurons == 3
        assert report.original_params > 0


class TestPruneStochastic:
    def test_near_zero_pruned(self):
        """Weights near 0 produce deterministic 0-bitstreams → pruned."""
        w = [np.array([[0.001, 0.5], [0.999, 0.01]])]
        pruned, report = prune_stochastic(w, bitstream_length=256, min_popcount_bits=1.0)
        # 0.001 → contribution = 0.001 * 256 = 0.256 < 1.0 → pruned
        # 0.01 → contribution = 0.01 * 256 = 2.56 > 1.0 → kept
        assert pruned[0][0, 0] == 0.0
        assert pruned[0][0, 1] == 0.5

    def test_near_one_pruned(self):
        """Weights near 1 produce deterministic 1-bitstreams → pruned."""
        w = [np.array([[0.998, 0.5]])]
        pruned, _ = prune_stochastic(w, bitstream_length=256, min_popcount_bits=1.0)
        # 0.998 → min(0.998, 0.002) * 256 = 0.512 < 1.0 → pruned
        assert pruned[0][0, 0] == 0.0
        assert pruned[0][0, 1] == 0.5

    def test_half_weight_never_pruned(self):
        """Weight 0.5 has maximum entropy → always kept."""
        w = [np.array([[0.5]])]
        pruned, report = prune_stochastic(w, bitstream_length=256, min_popcount_bits=1.0)
        # 0.5 → min(0.5, 0.5) * 256 = 128 >> 1.0 → kept
        assert pruned[0][0, 0] == 0.5
        assert report.pruned_params == 0

    def test_sparsity_increases_with_threshold(self):
        rng = np.random.RandomState(42)
        w = [rng.uniform(0, 1, (20, 20))]
        _, r1 = prune_stochastic(w, bitstream_length=256, min_popcount_bits=1.0)
        _, r2 = prune_stochastic(w, bitstream_length=256, min_popcount_bits=10.0)
        assert r2.sparsity >= r1.sparsity

    def test_longer_bitstream_less_pruning(self):
        """Longer bitstreams → higher contribution → less pruning."""
        w = [np.array([[0.01, 0.5]])]
        _, r_short = prune_stochastic(w, bitstream_length=64, min_popcount_bits=1.0)
        _, r_long = prune_stochastic(w, bitstream_length=1024, min_popcount_bits=1.0)
        assert r_long.sparsity <= r_short.sparsity

    def test_does_not_modify_original(self):
        w = [np.array([[0.001, 0.5]])]
        original = w[0].copy()
        prune_stochastic(w, bitstream_length=256, min_popcount_bits=1.0)
        np.testing.assert_array_equal(w[0], original)

    def test_report_fields(self):
        w = [np.array([[0.001, 0.5, 0.999]])]
        _, report = prune_stochastic(w, bitstream_length=256, min_popcount_bits=1.0)
        assert report.original_params == 3
        assert report.pruned_params >= 1
        assert report.remaining_params == report.original_params - report.pruned_params
        assert 0.0 <= report.sparsity <= 1.0


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


class TestQuantizeDelays:
    def test_basic(self):
        d = np.array([0.5, 1.7, 3.2, 5.0])
        q = quantize_delays(d, resolution=1)
        np.testing.assert_array_equal(q, np.array([0, 2, 3, 5]))

    def test_resolution_2(self):
        d = np.array([1.0, 2.5, 3.8, 7.0])
        q = quantize_delays(d, resolution=2)
        assert np.all(q % 2 == 0)

    def test_max_delay(self):
        d = np.array([1.0, 5.0, 10.0, 100.0])
        q = quantize_delays(d, resolution=1, max_delay=8)
        assert q.max() <= 8

    def test_negative_clamped(self):
        d = np.array([-1.0, 0.0, 1.0])
        q = quantize_delays(d, resolution=1)
        assert q[0] == 0

    def test_dtype(self):
        d = np.array([1.5, 2.5])
        q = quantize_delays(d, resolution=1)
        assert q.dtype == np.int64
