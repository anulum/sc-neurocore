# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPruneStochastic from former test_compression_toolkit.py

"""Focused suite: TestPruneStochastic from former test_compression_toolkit.py."""

from __future__ import annotations

from tests.compression_toolkit_support import *  # noqa: F403


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
