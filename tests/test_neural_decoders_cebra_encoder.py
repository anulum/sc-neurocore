# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestCEBRAEncoder from former test_neural_decoders.py

"""Focused suite: TestCEBRAEncoder from former test_neural_decoders.py."""

from __future__ import annotations

from tests.neural_decoders_support import *  # noqa: F403

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
