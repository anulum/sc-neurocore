# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestFederatedAggregator from former test_federated_sc.py

"""Focused suite: TestFederatedAggregator from former test_federated_sc.py."""

from __future__ import annotations

from federated_sc_support import *  # noqa: F403

class TestFederatedAggregator:
    def test_majority_vote(self):
        agg = FederatedAggregator(num_clients=3, bitstream_length=8)
        bs1 = [np.array([1, 1, 1, 0, 0, 0, 1, 0], dtype=np.uint8)]
        bs2 = [np.array([1, 1, 0, 0, 0, 1, 1, 0], dtype=np.uint8)]
        bs3 = [np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=np.uint8)]
        result = agg.aggregate_bitstreams([bs1, bs2, bs3])
        assert result[0][0] == 1
        assert result[0][3] == 0

    def test_aggregation_preserves_probability(self):
        rng = np.random.default_rng(42)
        agg = FederatedAggregator(num_clients=5, bitstream_length=1024)
        client_bs = []
        for i in range(5):
            bs = lfsr_encode(0.5, 0xACE1 + i * 1000, 1024)
            client_bs.append([bs])
        result = agg.aggregate_bitstreams(client_bs)
        p = bitstream_probability(result[0])
        assert abs(p - 0.5) < 0.1

    def test_weighted_aggregation(self):
        agg = FederatedAggregator(num_clients=2, bitstream_length=8)
        bs1 = [np.ones(8, dtype=np.uint8)]  # all 1s
        bs2 = [np.zeros(8, dtype=np.uint8)]  # all 0s
        # Heavy weight on client 0
        result = agg.aggregate_bitstreams([bs1, bs2], weights=[0.9, 0.1])
        assert np.sum(result[0]) == 8  # client 0 dominates

    def test_uniform_vs_weighted(self):
        agg = FederatedAggregator(num_clients=3, bitstream_length=8)
        bs1 = [np.ones(8, dtype=np.uint8)]
        bs2 = [np.zeros(8, dtype=np.uint8)]
        bs3 = [np.zeros(8, dtype=np.uint8)]
        # Uniform: 1 vs 2 → 0 wins
        uniform = agg.aggregate_bitstreams([bs1, bs2, bs3])
        assert np.sum(uniform[0]) == 0
        # Weighted: client 0 gets 0.8 → 1 wins
        weighted = agg.aggregate_bitstreams([bs1, bs2, bs3], weights=[0.8, 0.1, 0.1])
        assert np.sum(weighted[0]) == 8

    def test_outlier_detection_normal(self):
        agg = FederatedAggregator(num_clients=3, bitstream_length=128)
        rng = np.random.default_rng(42)
        similar = [
            [lfsr_encode(0.5, 0xACE1, 128)],
            [lfsr_encode(0.5, 0xBEEF, 128)],
            [lfsr_encode(0.5, 0xCAFE, 128)],
        ]
        outliers = agg.detect_outliers(similar, threshold=0.1)
        assert not any(outliers)

    def test_outlier_detection_catches_adversary(self):
        agg = FederatedAggregator(num_clients=3, bitstream_length=128)
        normal = lfsr_encode(0.5, 0xACE1, 128)
        adversary = np.ones(128, dtype=np.uint8)
        outliers = agg.detect_outliers(
            [[normal], [normal.copy()], [adversary]],
            threshold=0.9,
        )
        assert outliers[2] is True  # adversary flagged

    def test_single_client_no_outlier(self):
        agg = FederatedAggregator(num_clients=1)
        outliers = agg.detect_outliers([[np.ones(10, dtype=np.uint8)]])
        assert outliers == [False]

    def test_outlier_detection_zero_norm_client(self):
        # A client whose update is all zeros has zero norm, so its cosine
        # similarity is undefined and defined as 0 rather than dividing by zero;
        # below any positive threshold it is flagged as an outlier.
        agg = FederatedAggregator(num_clients=2, bitstream_length=16)
        empty = [np.zeros(16, dtype=np.uint8)]
        active = [np.ones(16, dtype=np.uint8)]
        outliers = agg.detect_outliers([empty, active], threshold=0.1)
        assert outliers[0] is True

    def test_verify_commitments_matches_and_rejects(self):
        agg = FederatedAggregator(num_clients=2, bitstream_length=4)
        bs_a = [np.array([1, 0, 1, 1], dtype=np.uint8)]
        bs_b = [np.array([0, 1, 0, 0], dtype=np.uint8)]
        good = CommitmentScheme.commit(np.concatenate(bs_a))
        results = agg.verify_commitments([bs_a, bs_b], [good, "deadbeef"])
        assert results == [True, False]
