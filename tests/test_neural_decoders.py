# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for foundation-model neural population decoders

"""Tests for neural_decoders: POYO+, POSSM, NDT3, CEBRA.

Multi-angle, publication-property tests — not just happy path.
"""

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.analysis.spike_stats.neural_decoders import (
    CEBRAEncoder,
    NDT3Decoder,
    POSSMDecoder,
    POYODecoder,
    scaled_dot_product_attention,
    sinusoidal_position_encode,
    tokenise_spikes,
)


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------


class TestTokeniseSpikes:
    """Spike tokenisation used by POYO+ and POSSM."""

    def test_empty_input(self) -> None:
        uids, ts = tokenise_spikes([])
        assert len(uids) == 0
        assert len(ts) == 0

    def test_no_spikes(self) -> None:
        trains = [np.zeros(100), np.zeros(100)]
        uids, ts = tokenise_spikes(trains)
        assert len(uids) == 0

    def test_single_spike(self) -> None:
        train = np.zeros(50)
        train[10] = 1
        uids, ts = tokenise_spikes([train], dt=0.5)
        assert len(uids) == 1
        assert uids[0] == 0
        assert ts[0] == pytest.approx(5.0)

    def test_sorted_by_time(self) -> None:
        t0 = np.zeros(100)
        t0[50] = 1
        t1 = np.zeros(100)
        t1[10] = 1
        uids, ts = tokenise_spikes([t0, t1])
        assert ts[0] < ts[1]
        assert uids[0] == 1
        assert uids[1] == 0

    def test_multiple_spikes_per_unit(self) -> None:
        train = np.zeros(20)
        train[5] = 1
        train[15] = 1
        uids, ts = tokenise_spikes([train])
        assert len(uids) == 2
        assert np.all(uids == 0)

    def test_dt_scaling(self) -> None:
        train = np.zeros(10)
        train[4] = 1
        _, ts1 = tokenise_spikes([train], dt=1.0)
        _, ts2 = tokenise_spikes([train], dt=0.1)
        assert ts1[0] == pytest.approx(4.0)
        assert ts2[0] == pytest.approx(0.4)


class TestSinusoidalPositionEncode:
    def test_shape(self) -> None:
        timestamps = np.array([0.0, 1.0, 2.0])
        pe = sinusoidal_position_encode(timestamps, 16)
        assert pe.shape == (3, 16)

    def test_zero_timestamp(self) -> None:
        pe = sinusoidal_position_encode(np.array([0.0]), 8)
        # sin(0) = 0 for all even dims
        assert pe[0, 0] == pytest.approx(0.0)
        # cos(0) = 1 for all odd dims
        assert pe[0, 1] == pytest.approx(1.0)

    def test_different_timestamps_differ(self) -> None:
        pe = sinusoidal_position_encode(np.array([0.0, 100.0]), 32)
        assert not np.allclose(pe[0], pe[1])


class TestScaledDotProductAttention:
    def test_identity_keys(self) -> None:
        n, d = 4, 8
        q = np.eye(n, d)
        k = np.eye(n, d)
        v = np.random.default_rng(42).normal(0, 1, (n, d))
        out = scaled_dot_product_attention(q, k, v)
        assert out.shape == (n, d)

    def test_uniform_attention_on_equal_keys(self) -> None:
        q = np.ones((2, 4))
        k = np.ones((3, 4))
        v = np.arange(12, dtype=np.float64).reshape(3, 4)
        out = scaled_dot_product_attention(q, k, v)
        # All keys equal → uniform weights → output = mean of values
        expected = v.mean(axis=0)
        np.testing.assert_allclose(out[0], expected, atol=1e-10)


# ---------------------------------------------------------------------------
# POYO+ — Azabou et al. (2023)
# ---------------------------------------------------------------------------


class TestPOYODecoder:
    def test_defaults(self) -> None:
        dec = POYODecoder()
        assert dec.d_model == 64
        assert dec.n_latents == 32

    def test_encode_empty(self) -> None:
        dec = POYODecoder()
        latents = dec.encode([])
        assert latents.shape == (32, 64)
        assert np.allclose(latents, 0.0)

    def test_encode_shape(self) -> None:
        dec = POYODecoder(d_model=16, n_latents=8)
        trains = [np.zeros(100) for _ in range(5)]
        for i, t in enumerate(trains):
            t[i * 10 + 5] = 1
        latents = dec.encode(trains)
        assert latents.shape == (8, 16)

    def test_different_activity_different_latents(self) -> None:
        dec = POYODecoder(d_model=16, n_latents=4, seed=7)
        t1 = [np.zeros(100)]
        t1[0][10] = 1
        t2 = [np.zeros(100)]
        t2[0][90] = 1
        l1 = dec.encode(t1)
        l2 = dec.encode(t2)
        assert not np.allclose(l1, l2)

    def test_decode_shape(self) -> None:
        dec = POYODecoder(d_model=16, n_latents=8)
        latents = np.random.default_rng(1).normal(0, 1, (8, 16))
        queries = np.random.default_rng(2).normal(0, 1, (3, 16))
        out = dec.decode(latents, queries)
        assert out.shape == (3, 16)

    def test_reset_clears_embeddings(self) -> None:
        dec = POYODecoder(d_model=8, n_latents=4)
        train = np.zeros(10)
        train[5] = 1
        dec.encode([train])
        assert len(dec._unit_embeddings) > 0
        dec.reset()
        assert len(dec._unit_embeddings) == 0

    def test_deterministic(self) -> None:
        dec = POYODecoder(d_model=16, n_latents=4, seed=42)
        trains = [np.zeros(50)]
        trains[0][10] = 1
        trains[0][30] = 1
        l1 = dec.encode(trains).copy()
        dec.reset()
        l2 = dec.encode(trains)
        np.testing.assert_array_equal(l1, l2)


# ---------------------------------------------------------------------------
# POSSM — Ryoo et al. (2025)
# ---------------------------------------------------------------------------


class TestPOSSMDecoder:
    def test_defaults(self) -> None:
        dec = POSSMDecoder()
        assert dec.d_model == 64
        assert dec.d_state == 32

    def test_discretise_zoh(self) -> None:
        """A_bar = exp(dt * A) for diagonal SSM."""
        dec = POSSMDecoder(d_model=4, d_state=2, dt=0.1, seed=1)
        a_bar, b_bar = dec.discretise(0.1)
        expected_a = np.exp(0.1 * dec._A)
        np.testing.assert_allclose(a_bar, expected_a)

    def test_step_output_shape(self) -> None:
        dec = POSSMDecoder(d_model=8, d_state=4)
        x = np.ones(8)
        y = dec.step(x)
        assert y.shape == (8,)

    def test_step_state_changes(self) -> None:
        dec = POSSMDecoder(d_model=4, d_state=2, seed=3)
        h_before = dec._h.copy()
        dec.step(np.ones(4))
        assert not np.allclose(dec._h, h_before)

    def test_encode_causal_empty(self) -> None:
        dec = POSSMDecoder(d_model=8)
        out = dec.encode_causal([])
        assert out.shape == (0, 8)

    def test_encode_causal_shape(self) -> None:
        dec = POSSMDecoder(d_model=16, d_state=8)
        trains = [np.zeros(50), np.zeros(50)]
        trains[0][10] = 1
        trains[1][20] = 1
        out = dec.encode_causal(trains)
        assert out.shape == (50, 16)

    def test_causal_no_future_leakage(self) -> None:
        """Output at time t must not depend on spikes at t+k."""
        dec = POSSMDecoder(d_model=8, d_state=4, seed=5)
        t1 = np.zeros(30)
        t1[5] = 1
        out1 = dec.encode_causal([t1])
        # Add future spike at t=25
        t2 = t1.copy()
        t2[25] = 1
        dec.reset()
        out2 = dec.encode_causal([t2])
        # Outputs before t=25 must be identical (causal)
        np.testing.assert_allclose(out1[:25], out2[:25])

    def test_reset_zeros_state(self) -> None:
        dec = POSSMDecoder(d_model=4, d_state=2)
        dec.step(np.ones(4))
        dec.reset()
        assert np.allclose(dec._h, 0.0)

    def test_oscillatory_dynamics(self) -> None:
        """Complex diagonal A produces oscillatory hidden state."""
        dec = POSSMDecoder(d_model=4, d_state=4, dt=0.01, seed=10)
        x = np.array([1.0, 0.0, 0.0, 0.0])
        dec.step(x)
        h1 = dec._h.copy()
        for _ in range(100):
            dec.step(np.zeros(4))
        h2 = dec._h
        # Imaginary parts of A cause oscillation → h should not converge to 0
        # but should decay (real part is -0.5)
        assert np.linalg.norm(h2) < np.linalg.norm(h1)
        assert np.linalg.norm(h2) > 0


# ---------------------------------------------------------------------------
# NDT3 — Ye & Pandarinath (2025)
# ---------------------------------------------------------------------------


class TestNDT3Decoder:
    def test_defaults(self) -> None:
        dec = NDT3Decoder()
        assert dec.d_model == 64
        assert dec.bin_size_ms == pytest.approx(20.0)

    def test_bin_and_embed_empty(self) -> None:
        dec = NDT3Decoder(d_model=8)
        binned, embedded = dec.bin_and_embed([])
        assert binned.shape[0] == 0

    def test_bin_and_embed_shape(self) -> None:
        dec = NDT3Decoder(d_model=16, bin_size_ms=10.0)
        trains = [np.zeros(100), np.zeros(100)]  # 100 steps @ dt=1 → 10 bins
        trains[0][5] = 1
        binned, embedded = dec.bin_and_embed(trains, dt=1.0)
        assert binned.shape == (10, 2)
        assert embedded.shape == (10, 16)

    def test_binning_counts_spikes(self) -> None:
        dec = NDT3Decoder(bin_size_ms=10.0)
        train = np.zeros(100)
        train[3] = 1
        train[7] = 1
        train[15] = 1
        binned, _ = dec.bin_and_embed([train], dt=1.0)
        assert binned[0, 0] == pytest.approx(2.0)  # bin 0: indices 0-9
        assert binned[1, 0] == pytest.approx(1.0)  # bin 1: indices 10-19

    def test_causal_mask_in_predict(self) -> None:
        """First bin prediction depends only on itself (causal)."""
        dec = NDT3Decoder(d_model=8, bin_size_ms=5.0, seed=42)
        trains = [np.zeros(50)]
        trains[0][2] = 1
        trains[0][42] = 1
        _, emb1 = dec.bin_and_embed(trains, dt=1.0)
        out = dec.predict_next(emb1)
        # Modify later bins, first bin output should not change
        trains2 = [np.zeros(50)]
        trains2[0][2] = 1
        trains2[0][45] = 1  # different late spike
        _, emb2 = dec.bin_and_embed(trains2, dt=1.0)
        out2 = dec.predict_next(emb2)
        np.testing.assert_allclose(out[0], out2[0], atol=1e-10)

    def test_decode_pipeline(self) -> None:
        dec = NDT3Decoder(d_model=8, bin_size_ms=5.0)
        trains = [np.zeros(30) for _ in range(3)]
        for i, t in enumerate(trains):
            t[i * 5 + 2] = 1
        out = dec.decode(trains)
        assert out.shape[1] == 8
        assert out.shape[0] > 0

    def test_different_activity_different_output(self) -> None:
        dec = NDT3Decoder(d_model=8, bin_size_ms=10.0, seed=99)
        t1 = [np.zeros(50)]
        t1[0][5] = 1
        t2 = [np.zeros(50)]
        t2[0][45] = 1
        o1 = dec.decode(t1)
        o2 = dec.decode(t2)
        assert not np.allclose(o1, o2)

    def test_bin_and_embed_train_shorter_than_one_bin(self) -> None:
        """Trains too short to fill a single 20 ms bin yield no bins at all,
        keeping the neuron dimension on the (empty) binned matrix."""
        dec = NDT3Decoder(d_model=8)
        trains = [np.zeros(5), np.zeros(5)]
        binned, embedded = dec.bin_and_embed(trains, dt=1.0)
        assert binned.shape == (0, 2)
        assert embedded.shape == (0, 8)

    def test_predict_next_on_empty_embedding(self) -> None:
        """Predicting from an empty embedding short-circuits to an empty
        output rather than running attention over zero positions."""
        dec = NDT3Decoder(d_model=8)
        empty = np.zeros((0, 8))
        out = dec.predict_next(empty)
        assert out.shape == (0, 8)


# ---------------------------------------------------------------------------
# CEBRA — Schneider, Lee & Mathis (2023)
# ---------------------------------------------------------------------------


class TestCEBRAEncoder:
    def test_defaults(self) -> None:
        enc = CEBRAEncoder()
        assert enc.d_input == 64
        assert enc.d_output == 8
        assert enc.temperature == pytest.approx(1.0)

    def test_encode_shape(self) -> None:
        enc = CEBRAEncoder(d_input=10, d_output=3)
        x = np.random.default_rng(1).normal(0, 1, (5, 10))
        z = enc.encode(x)
        assert z.shape == (5, 3)

    def test_encode_single_vector(self) -> None:
        enc = CEBRAEncoder(d_input=10, d_output=3)
        x = np.random.default_rng(1).normal(0, 1, 10)
        z = enc.encode(x)
        assert z.shape == (3,)

    def test_unit_norm_embeddings(self) -> None:
        """CEBRA normalises embeddings to unit hypersphere."""
        enc = CEBRAEncoder(d_input=16, d_output=4)
        x = np.random.default_rng(7).normal(0, 5, (20, 16))
        z = enc.encode(x)
        norms = np.linalg.norm(z, axis=-1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-6)

    def test_cosine_similarity_self(self) -> None:
        a = np.array([[1.0, 0.0], [0.0, 1.0]])
        sim = CEBRAEncoder.cosine_similarity(a, a)
        np.testing.assert_allclose(np.diag(sim), 1.0, atol=1e-10)

    def test_cosine_similarity_orthogonal(self) -> None:
        a = np.array([[1.0, 0.0]])
        b = np.array([[0.0, 1.0]])
        sim = CEBRAEncoder.cosine_similarity(a, b)
        assert abs(sim[0, 0]) < 1e-10

    def test_infonce_loss_positive(self) -> None:
        enc = CEBRAEncoder(d_input=8, d_output=4, seed=42)
        data = np.random.default_rng(1).normal(0, 1, (10, 8))
        loss = enc.infonce_loss(data, data)
        assert loss >= 0.0

    def test_infonce_loss_perfect_alignment(self) -> None:
        """Identical pairs → loss approaches -log(1/N) = log(N)."""
        enc = CEBRAEncoder(d_input=4, d_output=2, temperature=0.1, seed=42)
        x = np.random.default_rng(5).normal(0, 1, (8, 4))
        loss = enc.infonce_loss(x, x)
        # With identical pairs and low temperature, positive similarity
        # dominates → loss should be low (near 0)
        assert loss < 2.0

    def test_fit_reduces_loss(self) -> None:
        """Training with time-contrastive learning should decrease loss."""
        rng = np.random.default_rng(42)
        # Generate temporally smooth data (consecutive points are similar)
        n = 50
        data = np.cumsum(rng.normal(0, 0.1, (n, 8)), axis=0)
        enc = CEBRAEncoder(d_input=8, d_output=3, temperature=0.5, learning_rate=0.01, seed=42)
        initial_loss = enc.infonce_loss(data[:-1], data[1:])
        final_loss = enc.fit(data, n_steps=50, time_offset=1)
        assert final_loss < initial_loss

    def test_transform_equals_encode(self) -> None:
        enc = CEBRAEncoder(d_input=8, d_output=4)
        x = np.random.default_rng(3).normal(0, 1, (5, 8))
        np.testing.assert_array_equal(enc.transform(x), enc.encode(x))

    def test_fit_insufficient_data(self) -> None:
        enc = CEBRAEncoder(d_input=4, d_output=2)
        data = np.ones((1, 4))
        loss = enc.fit(data, n_steps=10)
        assert loss == 0.0

    def test_temperature_scaling(self) -> None:
        """Lower temperature → sharper similarity distribution."""
        rng = np.random.default_rng(10)
        data = rng.normal(0, 1, (10, 8))
        enc_cold = CEBRAEncoder(d_input=8, d_output=4, temperature=0.1, seed=1)
        enc_hot = CEBRAEncoder(d_input=8, d_output=4, temperature=10.0, seed=1)
        loss_cold = enc_cold.infonce_loss(data, data)
        loss_hot = enc_hot.infonce_loss(data, data)
        # Cold temperature with identical pairs → stronger signal → lower loss
        assert loss_cold < loss_hot

    def test_analytical_gradient_correctness(self) -> None:
        """Verify analytical backprop against numerical finite differences."""
        enc = CEBRAEncoder(d_input=4, d_output=2, temperature=1.0, seed=7)
        rng = np.random.default_rng(11)
        anchors = rng.normal(0, 1, (5, 4))
        positives = rng.normal(0, 1, (5, 4))
        _, cache = enc._forward_and_loss(anchors, positives)
        grads = enc._backward(cache)
        # Numerical gradient for w2[0,0]
        eps = 1e-5
        enc._w2[0, 0] += eps
        loss_plus = enc.infonce_loss(anchors, positives)
        enc._w2[0, 0] -= 2 * eps
        loss_minus = enc.infonce_loss(anchors, positives)
        enc._w2[0, 0] += eps
        numerical_grad = (loss_plus - loss_minus) / (2 * eps)
        assert abs(grads["w2"][0, 0] - numerical_grad) < 1e-3
