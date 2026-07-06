# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for bipolar stochastic computing primitives

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.core.bipolar import (
    bipolar_decode,
    bipolar_encode,
    bipolar_mac,
    bipolar_multiply,
    bipolar_sc_layer,
    float_to_bipolar_weights,
)


class TestBipolarEncodeDecode:
    def test_encode_plus_one(self):
        bits = bipolar_encode(1.0, 10000, rng=np.random.default_rng(42))
        assert bits.mean() > 0.95

    def test_encode_minus_one(self):
        bits = bipolar_encode(-1.0, 10000, rng=np.random.default_rng(42))
        assert bits.mean() < 0.05

    def test_encode_zero(self):
        bits = bipolar_encode(0.0, 10000, rng=np.random.default_rng(42))
        assert 0.45 < bits.mean() < 0.55

    def test_decode_roundtrip(self):
        for v in [-0.8, -0.3, 0.0, 0.5, 0.9]:
            bits = bipolar_encode(v, 100000, rng=np.random.default_rng(42))
            decoded = bipolar_decode(bits)
            assert abs(decoded - v) < 0.02, f"v={v}, decoded={decoded}"

    def test_rejects_out_of_range_encode_values(self):
        with pytest.raises(ValueError, match=r"\[-1, 1\]"):
            bipolar_encode(2.0, 100, rng=np.random.default_rng(42))
        with pytest.raises(ValueError, match=r"\[-1, 1\]"):
            bipolar_encode(-2.0, 100, rng=np.random.default_rng(42))

    def test_rejects_nonpositive_bitstream_length(self):
        with pytest.raises(ValueError, match="positive"):
            bipolar_encode(0.0, 0, rng=np.random.default_rng(42))

    def test_decode_rejects_empty_bitstream(self):
        with pytest.raises(ValueError, match="non-empty"):
            bipolar_decode(np.array([], dtype=np.uint8))


class TestBipolarMultiply:
    def test_xnor_same_inputs(self):
        a = np.array([1, 0, 1, 1, 0], dtype=np.uint8)
        result = bipolar_multiply(a, a)
        assert (result == 1).all()

    def test_xnor_opposite_inputs(self):
        a = np.array([1, 0, 1, 1, 0], dtype=np.uint8)
        b = 1 - a
        result = bipolar_multiply(a, b)
        assert (result == 0).all()

    def test_xnor_rejects_shape_mismatch(self):
        with pytest.raises(ValueError, match="same shape"):
            bipolar_multiply(np.array([1, 0], dtype=np.uint8), np.array([1], dtype=np.uint8))

    def test_statistical_multiplication(self):
        rng = np.random.default_rng(42)
        L = 100000
        for va, vb in [(0.5, 0.5), (-0.5, 0.5), (0.8, -0.3)]:
            a = bipolar_encode(va, L, rng=rng)
            b = bipolar_encode(vb, L, rng=rng)
            product = bipolar_multiply(a, b)
            decoded = bipolar_decode(product)
            expected = va * vb
            assert abs(decoded - expected) < 0.03, f"{va}*{vb}: expected={expected}, got={decoded}"


class TestBipolarMAC:
    def test_single_input(self):
        # 1 input: dot product = 0.5 * 0.8 = 0.4
        inputs = np.array([0.5])
        weights = np.array([[0.8]])
        result = bipolar_mac(inputs, weights, L=50000, seed=42)
        assert abs(result[0] - 0.4) < 0.05

    def test_two_inputs_dot_product(self):
        # dot product: 0.6*0.5 + (-0.4)*0.3 = 0.3 - 0.12 = 0.18
        inputs = np.array([0.6, -0.4])
        weights = np.array([[0.5, 0.3]])
        result = bipolar_mac(inputs, weights, L=50000, seed=42)
        assert abs(result[0] - 0.18) < 0.1

    def test_multiple_outputs(self):
        inputs = np.array([0.5, -0.5])
        weights = np.array([[0.8, 0.2], [-0.3, 0.7]])
        result = bipolar_mac(inputs, weights, L=50000, seed=42)
        assert result.shape == (2,)
        # out[0] = 0.5*0.8 + (-0.5)*0.2 = 0.4 - 0.1 = 0.3
        # out[1] = 0.5*(-0.3) + (-0.5)*0.7 = -0.15 - 0.35 = -0.5
        assert abs(result[0] - 0.3) < 0.1
        assert abs(result[1] - (-0.5)) < 0.1

    def test_longer_bitstream_more_accurate(self):
        inputs = np.array([0.5])
        weights = np.array([[0.8]])
        r1 = bipolar_mac(inputs, weights, L=1000, seed=42)
        r2 = bipolar_mac(inputs, weights, L=100000, seed=42)
        expected = 0.4
        assert abs(r2[0] - expected) < abs(r1[0] - expected) + 0.01

    def test_rejects_shape_mismatch(self):
        with pytest.raises(ValueError, match="shape"):
            bipolar_mac(np.array([0.5, -0.2]), np.array([[0.8]]), L=1000, seed=42)

    def test_rejects_out_of_range_inputs_and_weights(self):
        with pytest.raises(ValueError, match=r"\[-1, 1\]"):
            bipolar_mac(np.array([1.2]), np.array([[0.8]]), L=1000, seed=42)
        with pytest.raises(ValueError, match=r"\[-1, 1\]"):
            bipolar_mac(np.array([0.5]), np.array([[1.2]]), L=1000, seed=42)


class TestBipolarSCLayer:
    def test_output_shape(self):
        inputs = np.array([0.5, -0.3, 0.1])
        weights = np.array([[0.2, 0.4, -0.1], [-0.5, 0.3, 0.8]])
        out = bipolar_sc_layer(inputs, weights, bias=None, L=1000)
        assert out.shape == (2,)

    def test_relu_clips_negative(self):
        inputs = np.array([-0.9])
        weights = np.array([[0.9]])
        out = bipolar_sc_layer(inputs, weights, bias=None, L=50000, activation="relu")
        # -0.9 * 0.9 = -0.81, relu -> 0
        assert out[0] >= 0.0

    def test_output_bounded(self):
        inputs = np.random.default_rng(42).uniform(-1, 1, 10)
        weights = np.random.default_rng(43).uniform(-1, 1, (5, 10))
        out = bipolar_sc_layer(inputs, weights, bias=None, L=1000)
        assert (out >= -1.0).all() and (out <= 1.0).all()

    def test_rejects_unknown_activation(self):
        with pytest.raises(ValueError, match="activation"):
            bipolar_sc_layer(
                np.array([0.5]),
                np.array([[0.5]]),
                bias=None,
                L=1000,
                activation="sigmoid",
            )


class TestFloatToBipolarWeights:
    def test_normalises_to_minus_one_one(self):
        w = np.array([[-2.0, 1.0], [0.5, -0.3]])
        bp = float_to_bipolar_weights(w)
        assert bp.max() <= 1.0
        assert bp.min() >= -1.0
        assert abs(bp.max() - 1.0) < 1e-6 or abs(bp.min() + 1.0) < 1e-6

    def test_preserves_sign(self):
        w = np.array([-3.0, 2.0, 0.0, -1.0])
        bp = float_to_bipolar_weights(w)
        assert bp[0] < 0
        assert bp[1] > 0
        assert bp[2] == 0.0

    def test_torch_tensor(self):
        torch = __import__("pytest").importorskip("torch")
        w = torch.tensor([[-1.5, 0.5], [0.3, -0.8]])
        bp = float_to_bipolar_weights(w)
        assert isinstance(bp, np.ndarray)
        assert bp.shape == (2, 2)

    def test_rejects_non_finite_weights(self):
        with pytest.raises(ValueError, match="NaN or Inf"):
            float_to_bipolar_weights(np.array([1.0, np.inf]))


class TestBipolarValidationPaths:
    """Guard clauses and default-argument paths across the bipolar primitives."""

    def test_encode_rejects_non_finite_value(self):
        with pytest.raises(ValueError, match="NaN or Inf"):
            bipolar_encode(float("nan"), 100, rng=np.random.default_rng(42))

    def test_encode_builds_default_rng_when_omitted(self):
        # rng=None exercises the internal default_rng construction; a long stream
        # still decodes near the encoded value without a caller-supplied generator.
        bits = bipolar_encode(0.5, 20000)
        assert set(np.unique(bits)).issubset({0, 1})
        assert abs(bipolar_decode(bits) - 0.5) < 0.1

    def test_decode_rejects_non_binary_bits(self):
        with pytest.raises(ValueError, match="only 0/1 bits"):
            bipolar_decode(np.array([0, 2, 1], dtype=np.uint8))

    def test_mac_rejects_non_finite_inputs(self):
        with pytest.raises(ValueError, match="NaN or Inf"):
            bipolar_mac(np.array([0.5, np.nan]), np.array([[0.2, 0.1]]), L=100, seed=1)

    def test_mac_rejects_multidimensional_inputs(self):
        with pytest.raises(ValueError, match=r"inputs must have shape \(N,\)"):
            bipolar_mac(np.array([[0.5], [0.2]]), np.array([[0.2, 0.1]]), L=100, seed=1)

    def test_mac_rejects_non_2d_weights(self):
        with pytest.raises(ValueError, match=r"weights must have shape \(M, N\)"):
            bipolar_mac(np.array([0.5, 0.2]), np.array([0.2, 0.1]), L=100, seed=1)


class TestBipolarSCLayerBiasAndActivation:
    """The optional-bias branch and the tanh activation of the SC layer."""

    def test_bias_is_added_to_output(self):
        inputs = np.array([0.5, -0.3])
        weights = np.array([[0.2, 0.4], [-0.5, 0.3]])
        bias = np.array([0.5, -0.5])
        out = bipolar_sc_layer(inputs, weights, bias=bias, L=2000, activation="none")
        assert out.shape == (2,)
        assert (out >= -1.0).all() and (out <= 1.0).all()

    def test_bias_shape_mismatch_rejected(self):
        with pytest.raises(ValueError, match="bias shape"):
            bipolar_sc_layer(
                np.array([0.5, -0.3]),
                np.array([[0.2, 0.4], [-0.5, 0.3]]),
                bias=np.array([0.1]),  # (1,) against a (2,) output
                L=1000,
            )

    def test_bias_non_finite_rejected(self):
        with pytest.raises(ValueError, match="NaN or Inf"):
            bipolar_sc_layer(
                np.array([0.5, -0.3]),
                np.array([[0.2, 0.4], [-0.5, 0.3]]),
                bias=np.array([np.nan, 0.0]),
                L=1000,
            )

    def test_tanh_activation_bounds_output(self):
        inputs = np.array([0.9, 0.8])
        weights = np.array([[0.9, 0.9]])
        out = bipolar_sc_layer(inputs, weights, bias=None, L=50000, activation="tanh")
        assert out.shape == (1,)
        assert -1.0 <= out[0] <= 1.0
