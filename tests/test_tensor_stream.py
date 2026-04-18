# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for TensorStream domain conversions

"""Tests for TensorStream: prob↔bitstream↔quantum domain bridge."""

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.core.tensor_stream import TensorStream


class TestFromProb:
    def test_creates_prob_domain(self):
        ts = TensorStream.from_prob(np.array([0.5]))
        assert ts.domain == "prob"

    def test_preserves_data(self):
        data = np.array([0.1, 0.5, 0.9])
        ts = TensorStream.from_prob(data)
        np.testing.assert_array_equal(ts.data, data)


class TestProbToBitstream:
    def test_output_shape(self):
        ts = TensorStream.from_prob(np.array([0.5, 0.3]))
        bits = ts.to_bitstream(length=1024)
        assert bits.shape == (2, 1024)

    def test_output_binary(self):
        ts = TensorStream.from_prob(np.array([0.7]))
        bits = ts.to_bitstream(length=512)
        assert set(np.unique(bits)).issubset({0, 1})

    def test_probability_preserved(self):
        np.random.seed(42)
        p = 0.65
        ts = TensorStream.from_prob(np.array([p]))
        bits = ts.to_bitstream(length=10000)
        recovered = np.mean(bits)
        np.testing.assert_allclose(recovered, p, atol=0.02)

    @pytest.mark.parametrize("p", [0.0, 0.1, 0.5, 0.9, 1.0])
    def test_roundtrip_accuracy(self, p):
        np.random.seed(42)
        ts = TensorStream.from_prob(np.array([p]))
        bits = ts.to_bitstream(length=8192)
        ts_back = TensorStream(data=bits, domain="bitstream")
        recovered = ts_back.to_prob()[0]
        np.testing.assert_allclose(recovered, p, atol=0.03)


class TestBitstreamToProb:
    def test_all_ones(self):
        bits = np.ones((1, 100), dtype=np.uint8)
        ts = TensorStream(data=bits, domain="bitstream")
        np.testing.assert_allclose(ts.to_prob(), 1.0)

    def test_all_zeros(self):
        bits = np.zeros((1, 100), dtype=np.uint8)
        ts = TensorStream(data=bits, domain="bitstream")
        np.testing.assert_allclose(ts.to_prob(), 0.0)

    def test_half(self):
        bits = np.array([[1, 0, 1, 0, 1, 0, 1, 0, 1, 0]], dtype=np.uint8)
        ts = TensorStream(data=bits, domain="bitstream")
        np.testing.assert_allclose(ts.to_prob(), 0.5)


class TestProbToQuantum:
    def test_output_shape(self):
        ts = TensorStream.from_prob(np.array([0.3, 0.7]))
        q = ts.to_quantum()
        assert q.shape == (2, 2)  # 2 qubits × [alpha, beta]

    def test_normalisation(self):
        """Quantum states must be normalised: |α|² + |β|² = 1."""
        ts = TensorStream.from_prob(np.array([0.0, 0.25, 0.5, 0.75, 1.0]))
        q = ts.to_quantum()
        norms = np.abs(q[:, 0]) ** 2 + np.abs(q[:, 1]) ** 2
        np.testing.assert_allclose(norms, 1.0, atol=1e-10)

    def test_born_rule(self):
        """P(|1⟩) = |β|² must equal the original probability."""
        probs = np.array([0.0, 0.1, 0.5, 0.9, 1.0])
        ts = TensorStream.from_prob(probs)
        q = ts.to_quantum()
        born = np.abs(q[:, 1]) ** 2
        np.testing.assert_allclose(born, probs, atol=1e-10)

    def test_p_zero_gives_ground_state(self):
        """p=0 → |ψ⟩ = |0⟩ → α=1, β=0."""
        ts = TensorStream.from_prob(np.array([0.0]))
        q = ts.to_quantum()
        np.testing.assert_allclose(np.abs(q[0, 0]), 1.0, atol=1e-10)
        np.testing.assert_allclose(np.abs(q[0, 1]), 0.0, atol=1e-10)

    def test_p_one_gives_excited_state(self):
        """p=1 → |ψ⟩ = |1⟩ → α=0, β=1."""
        ts = TensorStream.from_prob(np.array([1.0]))
        q = ts.to_quantum()
        np.testing.assert_allclose(np.abs(q[0, 0]), 0.0, atol=1e-10)
        np.testing.assert_allclose(np.abs(q[0, 1]), 1.0, atol=1e-10)


class TestQuantumToProb:
    def test_roundtrip_exact(self):
        """prob → quantum → prob should be exact."""
        probs = np.array([0.0, 0.3, 0.5, 0.8, 1.0])
        ts = TensorStream.from_prob(probs)
        q = ts.to_quantum()
        ts_q = TensorStream(data=q, domain="quantum")
        recovered = ts_q.to_prob()
        np.testing.assert_allclose(recovered, probs, atol=1e-10)


class TestDomainPassthrough:
    def test_prob_to_prob(self):
        ts = TensorStream.from_prob(np.array([0.42]))
        np.testing.assert_allclose(ts.to_prob(), 0.42)

    def test_bitstream_to_bitstream(self):
        bits = np.array([[1, 0, 1]], dtype=np.uint8)
        ts = TensorStream(data=bits, domain="bitstream")
        np.testing.assert_array_equal(ts.to_bitstream(), bits)

    def test_quantum_to_quantum(self):
        q = np.array([[0.6 + 0j, 0.8 + 0j]])
        ts = TensorStream(data=q, domain="quantum")
        np.testing.assert_array_equal(ts.to_quantum(), q)


class TestInvalidConversions:
    def test_spike_to_bitstream_raises(self):
        ts = TensorStream(data=np.array([1]), domain="spike")
        with pytest.raises(ValueError):
            ts.to_bitstream()
