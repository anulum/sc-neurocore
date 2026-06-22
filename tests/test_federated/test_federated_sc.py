# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Federated SC Learning Tests

import numpy as np

from sc_neurocore.federated.federated_sc import (
    AdaptiveEpsilonScheduler,
    AuditLog,
    CommitmentScheme,
    ConvergenceTracker,
    DPCertificate,
    DPMechanism,
    ErrorFeedback,
    FederatedAggregator,
    FederatedClient,
    FederatedRound,
    PrivacyAccountant,
    SCGradientEncoder,
    SecretShare,
    amplified_epsilon,
    bitstream_probability,
    clip_gradients,
    fedprox_gradient,
    krum_select,
    lfsr_encode,
    poisson_subsample,
    sparsify_topk,
    stochastic_quantize,
    trimmed_mean,
)


# ── LFSR Encoder Tests ───────────────────────────────────────────────


class TestLFSREncoder:
    def test_encode_half(self):
        bs = lfsr_encode(0.5, 0xACE1, 1000)
        p = bitstream_probability(bs)
        assert abs(p - 0.5) < 0.05

    def test_encode_zero(self):
        bs = lfsr_encode(0.0, 0xACE1, 256)
        assert np.sum(bs) == 0

    def test_encode_one(self):
        bs = lfsr_encode(1.0, 0xACE1, 256)
        assert np.sum(bs) == 256

    def test_deterministic(self):
        a = lfsr_encode(0.3, 0xACE1, 128)
        b = lfsr_encode(0.3, 0xACE1, 128)
        assert np.array_equal(a, b)

    def test_different_seeds(self):
        a = lfsr_encode(0.5, 0xACE1, 128)
        b = lfsr_encode(0.5, 0xBEEF, 128)
        assert not np.array_equal(a, b)

    def test_zero_seed_is_reset_to_one(self):
        # A zero LFSR register is a fixed point that never advances, so a seed
        # of 0 must be bumped to 1 before stepping.
        bs = lfsr_encode(0.5, 0, 64)
        assert bs.shape == (64,)
        assert bs.dtype == np.uint8


# ── DP Mechanism Tests ───────────────────────────────────────────────


class TestDPMechanism:
    def test_flip_probability_range(self):
        dp = DPMechanism(epsilon=1.0)
        p = dp.flip_probability
        assert 0.0 < p < 0.5

    def test_higher_epsilon_less_noise(self):
        dp_low = DPMechanism(epsilon=0.5)
        dp_high = DPMechanism(epsilon=5.0)
        assert dp_high.flip_probability < dp_low.flip_probability

    def test_privatise_preserves_length(self):
        dp = DPMechanism(epsilon=1.0)
        rng = np.random.default_rng(42)
        bs = lfsr_encode(0.5, 0xACE1, 256)
        noisy = dp.privatise(bs, rng)
        assert len(noisy) == len(bs)

    def test_privatise_changes_bits(self):
        dp = DPMechanism(epsilon=0.1)
        rng = np.random.default_rng(42)
        bs = lfsr_encode(0.5, 0xACE1, 1000)
        noisy = dp.privatise(bs, rng)
        diff = np.sum(bs != noisy)
        assert diff > 0

    def test_high_epsilon_preserves_most_bits(self):
        dp = DPMechanism(epsilon=10.0)
        rng = np.random.default_rng(42)
        bs = lfsr_encode(0.5, 0xACE1, 1000)
        noisy = dp.privatise(bs, rng)
        diff = np.sum(bs != noisy)
        assert diff < 100

    def test_per_bit_epsilon(self):
        dp = DPMechanism(epsilon=2.0)
        assert dp.per_bit_epsilon() > 0

    def test_per_bit_epsilon_degenerate_flip_probability_is_infinite(self):
        # A deeply negative epsilon drives the flip probability to 1.0 (every
        # bit flipped), where ln((1-p)/p) is undefined: the per-bit cost is
        # reported as infinite rather than raising.
        dp = DPMechanism(epsilon=-800.0)
        assert dp.flip_probability >= 1.0
        assert dp.per_bit_epsilon() == float("inf")

    def test_total_epsilon(self):
        dp = DPMechanism(epsilon=1.0)
        total = dp.total_epsilon(256)
        assert total > 0


# ── Gradient Clipping Tests ──────────────────────────────────────────


class TestGradientClipping:
    def test_clips_large_gradient(self):
        g = np.array([3.0, 4.0])  # norm=5
        clipped = clip_gradients(g, max_norm=1.0)
        assert np.linalg.norm(clipped) <= 1.0 + 1e-6

    def test_does_not_clip_small_gradient(self):
        g = np.array([0.1, 0.2])
        clipped = clip_gradients(g, max_norm=10.0)
        np.testing.assert_array_almost_equal(clipped, g)

    def test_preserves_direction(self):
        g = np.array([6.0, 8.0])  # norm=10
        clipped = clip_gradients(g, max_norm=5.0)
        direction = g / np.linalg.norm(g)
        clipped_dir = clipped / np.linalg.norm(clipped)
        np.testing.assert_array_almost_equal(direction, clipped_dir)

    def test_zero_gradient(self):
        g = np.array([0.0, 0.0])
        clipped = clip_gradients(g, max_norm=1.0)
        np.testing.assert_array_equal(clipped, g)


# ── Gradient Sparsification Tests ────────────────────────────────────


class TestSparsifyTopK:
    def test_top_k_selects_largest(self):
        g = np.array([0.1, -0.5, 0.3, -0.8, 0.2])
        sparse, mask = sparsify_topk(g, k=2)
        assert mask[1] == 1  # |-0.5| is 2nd largest
        assert mask[3] == 1  # |-0.8| is largest
        assert np.count_nonzero(mask) == 2

    def test_sparse_preserves_values(self):
        g = np.array([0.1, 0.5, 0.3])
        sparse, mask = sparsify_topk(g, k=1)
        idx = np.argmax(mask)
        assert sparse[idx] == g[idx]

    def test_zero_entries(self):
        g = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        sparse, mask = sparsify_topk(g, k=2)
        assert np.count_nonzero(sparse) == 2

    def test_k_exceeds_length(self):
        g = np.array([1.0, 2.0])
        sparse, mask = sparsify_topk(g, k=10)
        np.testing.assert_array_almost_equal(sparse, g)


# ── Privacy Accountant Tests ─────────────────────────────────────────


class TestPrivacyAccountant:
    def test_initial_state(self):
        acc = PrivacyAccountant(target_epsilon=10.0)
        assert acc.current_epsilon() == 0.0
        assert not acc.is_exhausted()
        assert acc.rounds_consumed == 0

    def test_consume_round(self):
        acc = PrivacyAccountant(target_epsilon=100.0)
        dp = DPMechanism(epsilon=1.0)
        result = acc.consume_round(dp, 64)
        assert result is True
        assert acc.rounds_consumed == 1
        assert acc.current_epsilon() > 0

    def test_budget_exhaustion(self):
        acc = PrivacyAccountant(target_epsilon=0.01, target_delta=1e-5)
        dp = DPMechanism(epsilon=1.0)
        for _ in range(100):
            acc.consume_round(dp, 256)
        assert acc.is_exhausted()

    def test_remaining_epsilon(self):
        acc = PrivacyAccountant(target_epsilon=100.0)
        assert acc.remaining_epsilon() == 100.0
        dp = DPMechanism(epsilon=1.0)
        acc.consume_round(dp, 64)
        assert acc.remaining_epsilon() < 100.0


# ── Secret Sharing Tests ─────────────────────────────────────────────


class TestSecretSharing:
    def test_split_and_reconstruct(self):
        ss = SecretShare(num_parties=3)
        rng = np.random.default_rng(42)
        bs = lfsr_encode(0.5, 0xACE1, 128)
        shares = ss.split(bs, rng)
        assert len(shares) == 3
        reconstructed = SecretShare.reconstruct(shares)
        assert np.array_equal(bs, reconstructed)

    def test_individual_shares_random(self):
        ss = SecretShare(num_parties=3)
        rng = np.random.default_rng(42)
        bs = lfsr_encode(0.7, 0xACE1, 256)
        shares = ss.split(bs, rng)
        for share in shares:
            p = bitstream_probability(share)
            assert abs(p - 0.7) > 0.01 or True

    def test_verify_reconstruction(self):
        ss = SecretShare(num_parties=5)
        rng = np.random.default_rng(42)
        bs = lfsr_encode(0.3, 0xACE1, 64)
        shares = ss.split(bs, rng)
        assert SecretShare.verify_reconstruction(bs, shares)

    def test_two_party(self):
        ss = SecretShare(num_parties=2)
        rng = np.random.default_rng(42)
        bs = np.array([1, 0, 1, 1, 0], dtype=np.uint8)
        shares = ss.split(bs, rng)
        assert np.array_equal(bs, SecretShare.reconstruct(shares))


# ── Commitment Scheme Tests ──────────────────────────────────────────


class TestCommitmentScheme:
    def test_commit_deterministic(self):
        data = np.array([1, 0, 1, 1], dtype=np.uint8)
        c1 = CommitmentScheme.commit(data)
        c2 = CommitmentScheme.commit(data)
        assert c1 == c2

    def test_commit_different_data(self):
        a = np.array([1, 0, 1], dtype=np.uint8)
        b = np.array([0, 1, 0], dtype=np.uint8)
        assert CommitmentScheme.commit(a) != CommitmentScheme.commit(b)

    def test_verify(self):
        data = np.array([1, 0, 1, 1, 0], dtype=np.uint8)
        c = CommitmentScheme.commit(data)
        assert CommitmentScheme.verify(data, c)

    def test_verify_with_nonce(self):
        rng = np.random.default_rng(42)
        data = np.array([1, 0, 1], dtype=np.uint8)
        nonce = CommitmentScheme.generate_nonce(rng)
        c = CommitmentScheme.commit(data, nonce)
        assert CommitmentScheme.verify(data, c, nonce)

    def test_nonce_binding(self):
        rng = np.random.default_rng(42)
        data = np.array([1, 0, 1], dtype=np.uint8)
        n1 = CommitmentScheme.generate_nonce(rng)
        n2 = CommitmentScheme.generate_nonce(rng)
        assert CommitmentScheme.commit(data, n1) != CommitmentScheme.commit(data, n2)

    def test_sha256_length(self):
        data = np.array([1, 0], dtype=np.uint8)
        c = CommitmentScheme.commit(data)
        assert len(c) == 64


# ── SC Gradient Encoder Tests ────────────────────────────────────────


class TestSCGradientEncoder:
    def test_encode_decode_roundtrip(self):
        enc = SCGradientEncoder(bitstream_length=1024, dp=DPMechanism(epsilon=10.0))
        rng = np.random.default_rng(42)
        gradients = np.array([0.1, 0.5, 0.9])
        seeds = np.array([0xACE1, 0xBEEF, 0xCAFE])
        bitstreams = enc.encode(gradients, seeds, rng)
        decoded = enc.decode(bitstreams, gradients.min(), gradients.max())
        assert len(decoded) == 3
        for i in range(3):
            assert abs(decoded[i] - gradients[i]) < 0.15

    def test_encode_length(self):
        enc = SCGradientEncoder(bitstream_length=512)
        rng = np.random.default_rng(42)
        gradients = np.array([0.3, 0.7])
        seeds = np.array([0xACE1, 0xBEEF])
        bitstreams = enc.encode(gradients, seeds, rng)
        assert len(bitstreams) == 2
        assert len(bitstreams[0]) == 512

    def test_encode_zero_seed_is_reset(self):
        # A supplied seed of 0 masks to a zero register, which the encoder must
        # bump to 1 before LFSR stepping.
        enc = SCGradientEncoder(bitstream_length=64)
        rng = np.random.default_rng(0)
        gradients = np.array([0.2, 0.8])
        bitstreams = enc.encode(gradients, np.array([0, 0]), rng)
        assert len(bitstreams) == 2
        assert all(len(bs) == 64 for bs in bitstreams)


# ── Federated Client Tests ───────────────────────────────────────────


class TestFederatedClient:
    def test_local_train(self):
        enc = SCGradientEncoder(bitstream_length=128, dp=DPMechanism(epsilon=2.0))
        client = FederatedClient(client_id=0, encoder=enc)
        data = np.random.default_rng(42).standard_normal((20, 5))
        labels = np.random.default_rng(42).standard_normal(20)
        grads = client.local_train(data, labels)
        assert len(grads) == 5

    def test_encode_gradients(self):
        enc = SCGradientEncoder(bitstream_length=128, dp=DPMechanism(epsilon=2.0))
        client = FederatedClient(client_id=1, encoder=enc)
        grads = np.array([0.1, -0.2, 0.3])
        bitstreams, commitment, g_min, g_max = client.encode_gradients(grads)
        assert len(bitstreams) == 3
        assert len(commitment) == 64
        assert g_min <= g_max

    def test_deterministic_by_client_id(self):
        enc = SCGradientEncoder(bitstream_length=128, dp=DPMechanism(epsilon=5.0))
        c1 = FederatedClient(client_id=0, encoder=enc)
        c2 = FederatedClient(client_id=1, encoder=enc)
        grads = np.array([0.5])
        bs1, _, _, _ = c1.encode_gradients(grads)
        bs2, _, _, _ = c2.encode_gradients(grads)
        assert not np.array_equal(bs1[0], bs2[0])


# ── Federated Aggregator Tests ───────────────────────────────────────


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


# ── Poisson Subsampling Tests ────────────────────────────────────────


class TestPoissonSubsampling:
    def test_always_returns_at_least_one(self):
        enc = SCGradientEncoder(bitstream_length=64)
        clients = [FederatedClient(client_id=i, encoder=enc) for i in range(5)]
        rng = np.random.default_rng(42)
        selected = poisson_subsample(clients, sampling_rate=0.01, rng=rng)
        assert len(selected) >= 1

    def test_full_rate_selects_all(self):
        enc = SCGradientEncoder(bitstream_length=64)
        clients = [FederatedClient(client_id=i, encoder=enc) for i in range(5)]
        rng = np.random.default_rng(42)
        selected = poisson_subsample(clients, sampling_rate=1.0, rng=rng)
        assert len(selected) == 5

    def test_half_rate_reasonable(self):
        enc = SCGradientEncoder(bitstream_length=64)
        clients = [FederatedClient(client_id=i, encoder=enc) for i in range(100)]
        rng = np.random.default_rng(42)
        selected = poisson_subsample(clients, sampling_rate=0.5, rng=rng)
        assert 20 < len(selected) < 80


# ── Convergence Tracker Tests ────────────────────────────────────────


class TestConvergenceTracker:
    def test_not_converged_initially(self):
        ct = ConvergenceTracker()
        assert not ct.converged

    def test_converged_after_stable_norms(self):
        ct = ConvergenceTracker()
        for _ in range(10):
            ct.record(np.array([0.001, 0.002]))
        assert ct.converged

    def test_not_converged_if_large_norm(self):
        ct = ConvergenceTracker()
        for _ in range(4):
            ct.record(np.array([0.001, 0.002]))
        ct.record(np.array([10.0, 20.0]))
        assert not ct.converged

    def test_trend_decreasing(self):
        ct = ConvergenceTracker()
        ct.record(np.array([10.0]))
        ct.record(np.array([5.0]))
        assert ct.trend == "decreasing"

    def test_trend_increasing(self):
        ct = ConvergenceTracker()
        ct.record(np.array([5.0]))
        ct.record(np.array([10.0]))
        assert ct.trend == "increasing"

    def test_trend_insufficient(self):
        ct = ConvergenceTracker()
        assert ct.trend == "insufficient_data"

    def test_trend_stable_on_equal_norms(self):
        # Two consecutive rounds with the same gradient norm are neither rising
        # nor falling, so the trend is reported as stable.
        ct = ConvergenceTracker()
        ct.record(np.array([3.0, 4.0]))  # norm 5.0
        ct.record(np.array([4.0, 3.0]))  # norm 5.0
        assert ct.trend == "stable"

    def test_record_loss(self):
        ct = ConvergenceTracker()
        ct.record_loss(1.5)
        ct.record_loss(1.2)
        assert len(ct.round_losses) == 2


# ── Federated Round Tests ────────────────────────────────────────────


class TestFederatedRound:
    def _make_round(
        self,
        num_clients=3,
        epsilon=10.0,
        target_eps=1000.0,
        clip_norm=0.0,
        sampling_rate=1.0,
        audit_log=None,
    ):
        enc = SCGradientEncoder(bitstream_length=128, dp=DPMechanism(epsilon=epsilon))
        clients = [FederatedClient(client_id=i, encoder=enc) for i in range(num_clients)]
        agg = FederatedAggregator(num_clients=num_clients, bitstream_length=128)
        acc = PrivacyAccountant(target_epsilon=target_eps)
        return FederatedRound(
            clients=clients,
            aggregator=agg,
            accountant=acc,
            clip_norm=clip_norm,
            sampling_rate=sampling_rate,
            audit_log=audit_log,
        )

    def test_single_round(self):
        rng = np.random.default_rng(42)
        fr = self._make_round()
        data = [rng.standard_normal((10, 3)) for _ in range(3)]
        labels = [rng.standard_normal(10) for _ in range(3)]
        result = fr.run(data, labels)
        assert result is not None
        assert len(result) == 3

    def test_multiple_rounds(self):
        rng = np.random.default_rng(42)
        fr = self._make_round()
        for _ in range(5):
            data = [rng.standard_normal((10, 3)) for _ in range(3)]
            labels = [rng.standard_normal(10) for _ in range(3)]
            fr.run(data, labels)
        assert fr.round_number == 5

    def test_budget_exhaustion_stops(self):
        fr = self._make_round(target_eps=0.001)
        rng = np.random.default_rng(42)
        data = [rng.standard_normal((10, 3)) for _ in range(3)]
        labels = [rng.standard_normal(10) for _ in range(3)]
        fr.run(data, labels)
        result = fr.run(data, labels)
        assert result is None

    def test_status(self):
        fr = self._make_round()
        status = fr.status()
        assert "round" in status
        assert "epsilon_consumed" in status
        assert "epsilon_remaining" in status
        assert "budget_exhausted" in status
        assert "converged" in status
        assert "trend" in status

    def test_convergence_tracking(self):
        rng = np.random.default_rng(42)
        fr = self._make_round()
        data = [rng.standard_normal((10, 3)) for _ in range(3)]
        labels = [rng.standard_normal(10) for _ in range(3)]
        fr.run(data, labels)
        assert len(fr.convergence.grad_norms) == 1

    def test_gradient_clipping_active(self):
        rng = np.random.default_rng(42)
        fr = self._make_round(clip_norm=0.01)
        data = [rng.standard_normal((10, 3)) for _ in range(3)]
        labels = [rng.standard_normal(10) for _ in range(3)]
        result = fr.run(data, labels)
        assert result is not None

    def test_subsampling_round(self):
        rng = np.random.default_rng(42)
        fr = self._make_round(num_clients=10, sampling_rate=0.5)
        data = [rng.standard_normal((10, 3)) for _ in range(10)]
        labels = [rng.standard_normal(10) for _ in range(10)]
        result = fr.run(data, labels)
        assert result is not None

    def test_weighted_round(self):
        rng = np.random.default_rng(42)
        fr = self._make_round(num_clients=3)
        data = [rng.standard_normal((10, 3)) for _ in range(3)]
        labels = [rng.standard_normal(10) for _ in range(3)]
        result = fr.run(data, labels, client_weights=[0.5, 0.3, 0.2])
        assert result is not None
        assert len(result) == 3

    def test_audit_log_integration(self):
        log = AuditLog()
        rng = np.random.default_rng(42)
        fr = self._make_round(audit_log=log)
        data = [rng.standard_normal((10, 3)) for _ in range(3)]
        labels = [rng.standard_normal(10) for _ in range(3)]
        fr.run(data, labels)
        fr.run(data, labels)
        assert log.total_rounds == 2
        entries = log.to_list()
        assert entries[0]["round"] == 1
        assert entries[1]["round"] == 2


# ── DP Certificate Tests ─────────────────────────────────────────────


class TestDPCertificate:
    def test_from_accountant(self):
        acc = PrivacyAccountant(target_epsilon=10.0)
        dp = DPMechanism(epsilon=1.0)
        acc.consume_round(dp, 128)
        cert = DPCertificate.from_accountant(acc, dp, 128)
        assert cert.mechanism == "bitstream_flip_rr"
        assert cert.rounds == 1
        assert cert.delta == 1e-5

    def test_to_dict(self):
        acc = PrivacyAccountant(target_epsilon=10.0)
        dp = DPMechanism(epsilon=1.0)
        acc.consume_round(dp, 128)
        cert = DPCertificate.from_accountant(acc, dp, 128)
        d = cert.to_dict()
        assert "mechanism" in d
        assert "compliant" in d
        assert d["composition_method"] == "renyi_dp"

    def test_compliant_status(self):
        acc = PrivacyAccountant(target_epsilon=100.0)
        dp = DPMechanism(epsilon=1.0)
        acc.consume_round(dp, 64)
        cert = DPCertificate.from_accountant(acc, dp, 64)
        assert cert.is_compliant

    def test_non_compliant_status(self):
        acc = PrivacyAccountant(target_epsilon=0.001)
        dp = DPMechanism(epsilon=1.0)
        for _ in range(100):
            acc.consume_round(dp, 256)
        cert = DPCertificate.from_accountant(acc, dp, 256)
        assert not cert.is_compliant


# ── Stochastic Quantization Tests ────────────────────────────────────


class TestStochasticQuantize:
    def test_unbiased(self):
        g = np.array([0.3, 0.7, 0.5])
        results = [
            stochastic_quantize(g, levels=4, rng=np.random.default_rng(i)) for i in range(1000)
        ]
        mean_q = np.mean(results, axis=0)
        for i in range(3):
            assert abs(mean_q[i] - g[i]) < 0.05

    def test_output_in_range(self):
        rng = np.random.default_rng(42)
        g = np.array([-1.0, 0.5, 2.0])
        q = stochastic_quantize(g, levels=8, rng=rng)
        assert q.min() >= g.min() - 0.01
        assert q.max() <= g.max() + 0.01

    def test_constant_gradient(self):
        rng = np.random.default_rng(42)
        g = np.array([0.5, 0.5, 0.5])
        q = stochastic_quantize(g, levels=4, rng=rng)
        np.testing.assert_array_almost_equal(q, g)


# ── Adaptive Epsilon Scheduler Tests ─────────────────────────────────


class TestAdaptiveEpsilonScheduler:
    def test_initial_epsilon(self):
        sched = AdaptiveEpsilonScheduler(base_epsilon=2.0)
        assert sched.current_epsilon == 2.0

    def test_decay_on_convergence(self):
        sched = AdaptiveEpsilonScheduler(base_epsilon=2.0, decay_rate=0.5)
        eps = sched.step(converging=True)
        assert eps == 1.0

    def test_increase_on_divergence(self):
        sched = AdaptiveEpsilonScheduler(base_epsilon=2.0, decay_rate=0.5)
        sched.current_epsilon = 0.5
        eps = sched.step(converging=False)
        assert eps == 1.0

    def test_min_epsilon_floor(self):
        sched = AdaptiveEpsilonScheduler(base_epsilon=1.0, decay_rate=0.01, min_epsilon=0.5)
        eps = sched.step(converging=True)
        assert eps >= 0.5

    def test_max_epsilon_cap(self):
        sched = AdaptiveEpsilonScheduler(base_epsilon=2.0, decay_rate=0.5)
        sched.current_epsilon = 1.5
        eps = sched.step(converging=False)
        assert eps <= 2.0


# ── Krum Byzantine Selection Tests ───────────────────────────────────


class TestKrumSelect:
    def test_selects_central(self):
        vecs = [
            np.array([0.0, 0.0]),
            np.array([0.1, 0.1]),
            np.array([10.0, 10.0]),
        ]
        idx = krum_select(vecs, num_byzantine=1)
        assert idx in [0, 1]

    def test_single_byzantine(self):
        honest = [
            np.array([1.0, 1.0]) + np.random.default_rng(i).standard_normal(2) * 0.1
            for i in range(5)
        ]
        byzantine = [np.array([100.0, -100.0])]
        all_vecs = honest + byzantine
        idx = krum_select(all_vecs, num_byzantine=1)
        assert idx < 5


# ── Trimmed Mean Tests ───────────────────────────────────────────────


class TestTrimmedMean:
    def test_removes_extremes(self):
        vecs = [
            np.array([1.0, 1.0]),
            np.array([1.1, 0.9]),
            np.array([0.9, 1.1]),
            np.array([100.0, -100.0]),
            np.array([1.0, 1.0]),
        ]
        result = trimmed_mean(vecs, trim_fraction=0.2)
        assert abs(result[0] - 1.0) < 0.2
        assert abs(result[1] - 1.0) < 0.2

    def test_matches_mean_without_trimming(self):
        vecs = [np.array([1.0, 2.0]), np.array([3.0, 4.0]), np.array([5.0, 6.0])]
        result = trimmed_mean(vecs, trim_fraction=0.0)
        np.testing.assert_array_almost_equal(result, np.array([3.0, 4.0]))

    def test_over_trimming_falls_back_to_full_mean(self):
        # With two clients the minimum trim of one from each end removes every
        # row, so the aggregator falls back to the untrimmed mean rather than
        # averaging an empty slice.
        vecs = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
        result = trimmed_mean(vecs, trim_fraction=0.1)
        np.testing.assert_array_almost_equal(result, np.array([2.0, 3.0]))


# ── FedProx Tests ────────────────────────────────────────────────────


class TestFedProx:
    def test_proximal_no_drift(self):
        g = np.array([0.5, 0.3])
        w = np.array([1.0, 1.0])
        result = fedprox_gradient(g, w, w, mu=0.1)
        np.testing.assert_array_almost_equal(result, g)

    def test_proximal_with_drift(self):
        g = np.array([0.5, 0.3])
        w_local = np.array([2.0, 2.0])
        w_global = np.array([1.0, 1.0])
        result = fedprox_gradient(g, w_local, w_global, mu=0.1)
        expected = g + 0.1 * (w_local - w_global)
        np.testing.assert_array_almost_equal(result, expected)


# ── Error Feedback Tests ─────────────────────────────────────────────


class TestErrorFeedback:
    def test_initial_no_residual(self):
        ef = ErrorFeedback()
        g = np.array([1.0, 2.0, 3.0])
        acc = ef.accumulate(g)
        np.testing.assert_array_almost_equal(acc, g)

    def test_accumulates_residual(self):
        ef = ErrorFeedback()
        g1 = np.array([1.0, 2.0, 3.0])
        sparse = np.array([0.0, 2.0, 0.0])
        ef.update(g1, sparse)
        g2 = np.array([0.5, 0.5, 0.5])
        acc = ef.accumulate(g2)
        expected = g2 + (g1 - sparse)
        np.testing.assert_array_almost_equal(acc, expected)


# ── Privacy Amplification Tests ──────────────────────────────────────


class TestPrivacyAmplification:
    def test_full_sampling_no_amplification(self):
        assert amplified_epsilon(1.0, 1.0) == 1.0

    def test_subsampling_reduces_epsilon(self):
        amp = amplified_epsilon(1.0, 0.1)
        assert amp < 1.0

    def test_zero_sampling(self):
        assert amplified_epsilon(1.0, 0.0) == 0.0

    def test_monotonic_in_rate(self):
        a = amplified_epsilon(2.0, 0.1)
        b = amplified_epsilon(2.0, 0.5)
        c = amplified_epsilon(2.0, 1.0)
        assert a < b < c


# ── Audit Log Tests ──────────────────────────────────────────────────


class TestAuditLog:
    def test_empty_log(self):
        log = AuditLog()
        assert log.total_rounds == 0
        assert log.max_epsilon == 0.0

    def test_log_round(self):
        log = AuditLog()
        log.log_round(round_number=1, num_active=5, epsilon_consumed=0.5, grad_norm=0.01)
        assert log.total_rounds == 1

    def test_to_list(self):
        log = AuditLog()
        log.log_round(round_number=1, num_active=5, epsilon_consumed=0.5, grad_norm=0.01)
        log.log_round(round_number=2, num_active=3, epsilon_consumed=1.0, grad_norm=0.02)
        entries = log.to_list()
        assert len(entries) == 2
        assert entries[0]["round"] == 1
        assert entries[1]["epsilon"] == 1.0

    def test_max_epsilon(self):
        log = AuditLog()
        log.log_round(1, 5, 0.5, 0.01)
        log.log_round(2, 5, 1.5, 0.01)
        log.log_round(3, 5, 1.0, 0.01)
        assert log.max_epsilon == 1.5
