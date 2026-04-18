# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for SC arithmetic primitives (MUX, XNOR, NOT)

"""Tests for SC fundamental gates: MUX (addition), XNOR (bipolar multiply), NOT (complement)."""

import numpy as np

from sc_neurocore.accel.vector_ops import (
    pack_bitstream,
    vec_and,
    vec_xnor,
    vec_not,
    vec_mux,
    vec_popcount,
)


def _prob(packed, length):
    """Estimate probability from packed bitstream."""
    return vec_popcount(packed) / length


def _bernoulli_packed(p, length, seed):
    """Generate a packed Bernoulli bitstream."""
    rng = np.random.RandomState(seed)
    bits = (rng.random(length) < p).astype(np.uint8)
    return pack_bitstream(bits), length


class TestVecNot:
    def test_complement_probability(self):
        p = 0.7
        packed, length = _bernoulli_packed(p, 10000, seed=42)
        result = vec_not(packed)
        estimated = _prob(result, length)
        np.testing.assert_allclose(estimated, 1.0 - p, atol=0.03)

    def test_double_not_identity(self):
        packed, length = _bernoulli_packed(0.6, 1000, seed=99)
        result = vec_not(vec_not(packed))
        assert np.array_equal(packed, result)


class TestVecXnor:
    def test_bipolar_multiply(self):
        # XNOR on unipolar streams gives P(A)*P(B) + (1-P(A))*(1-P(B))
        pa, pb = 0.8, 0.6
        a, length = _bernoulli_packed(pa, 10000, seed=10)
        b, _ = _bernoulli_packed(pb, 10000, seed=20)
        result = vec_xnor(a, b)
        expected = pa * pb + (1 - pa) * (1 - pb)
        np.testing.assert_allclose(_prob(result, length), expected, atol=0.03)

    def test_self_xnor_is_all_ones(self):
        packed, length = _bernoulli_packed(0.5, 1000, seed=42)
        result = vec_xnor(packed, packed)
        # XNOR(x, x) = NOT(XOR(x, x)) = NOT(0) = all 1s
        assert _prob(result, length) > 0.99


class TestVecMux:
    def test_half_addition(self):
        # MUX with sel=0.5 gives (A + B) / 2
        pa, pb = 0.8, 0.2
        a, length = _bernoulli_packed(pa, 20000, seed=1)
        b, _ = _bernoulli_packed(pb, 20000, seed=2)
        sel, _ = _bernoulli_packed(0.5, 20000, seed=3)
        result = vec_mux(sel, a, b)
        expected = 0.5 * pa + 0.5 * pb  # = 0.5
        np.testing.assert_allclose(_prob(result, length), expected, atol=0.03)

    def test_sel_one_passes_a(self):
        a, length = _bernoulli_packed(0.7, 5000, seed=10)
        b, _ = _bernoulli_packed(0.3, 5000, seed=20)
        sel = pack_bitstream(np.ones(5000, dtype=np.uint8))
        result = vec_mux(sel, a, b)
        assert np.array_equal(result, a)

    def test_sel_zero_passes_b(self):
        a, length = _bernoulli_packed(0.7, 5000, seed=10)
        b, _ = _bernoulli_packed(0.3, 5000, seed=20)
        sel = pack_bitstream(np.zeros(5000, dtype=np.uint8))
        result = vec_mux(sel, a, b)
        assert np.array_equal(result, b)

    def test_weighted_addition(self):
        # MUX with sel=0.3 gives 0.3*A + 0.7*B
        pa, pb = 0.9, 0.1
        a, length = _bernoulli_packed(pa, 20000, seed=5)
        b, _ = _bernoulli_packed(pb, 20000, seed=6)
        sel, _ = _bernoulli_packed(0.3, 20000, seed=7)
        result = vec_mux(sel, a, b)
        expected = 0.3 * pa + 0.7 * pb  # = 0.34
        np.testing.assert_allclose(_prob(result, length), expected, atol=0.03)


class TestBipolarEncoding:
    def test_bipolar_encode_positive(self):
        from sc_neurocore.utils.bitstreams import generate_bipolar_bitstream, bipolar_to_value

        bits = generate_bipolar_bitstream(0.6, 10000)
        val = bipolar_to_value(bits)
        np.testing.assert_allclose(val, 0.6, atol=0.05)

    def test_bipolar_encode_negative(self):
        from sc_neurocore.utils.bitstreams import generate_bipolar_bitstream, bipolar_to_value

        bits = generate_bipolar_bitstream(-0.4, 10000)
        val = bipolar_to_value(bits)
        np.testing.assert_allclose(val, -0.4, atol=0.05)

    def test_bipolar_multiply_via_xnor(self):
        from sc_neurocore.utils.bitstreams import generate_bipolar_bitstream

        a_bits = generate_bipolar_bitstream(0.6, 20000)
        b_bits = generate_bipolar_bitstream(-0.3, 20000)
        packed_a = pack_bitstream(a_bits)
        packed_b = pack_bitstream(b_bits)
        result_packed = vec_xnor(packed_a, packed_b)
        result_prob = _prob(result_packed, 20000)
        result_bipolar = 2.0 * result_prob - 1.0
        np.testing.assert_allclose(result_bipolar, 0.6 * -0.3, atol=0.06)

    def test_bipolar_encoder_mode(self):
        from sc_neurocore.utils.bitstreams import BitstreamEncoder

        enc = BitstreamEncoder(x_min=-5.0, x_max=5.0, length=10000, mode="bipolar")
        bits = enc.encode(2.5)
        decoded = enc.decode(bits)
        np.testing.assert_allclose(decoded, 2.5, atol=0.5)

    def test_bipolar_out_of_range_raises(self):
        from sc_neurocore.utils.bitstreams import generate_bipolar_bitstream
        from sc_neurocore.exceptions import SCEncodingError
        import pytest

        with pytest.raises(SCEncodingError):
            generate_bipolar_bitstream(1.5, 100)

    def test_bipolar_to_value_empty_raises(self):
        from sc_neurocore.utils.bitstreams import bipolar_to_value
        from sc_neurocore.exceptions import SCEncodingError
        import pytest

        with pytest.raises(SCEncodingError, match="empty"):
            bipolar_to_value(np.array([], dtype=np.uint8))

    def test_value_to_bipolar_prob(self):
        from sc_neurocore.utils.bitstreams import value_to_bipolar_prob

        assert value_to_bipolar_prob(0.0) == 0.5
        assert value_to_bipolar_prob(1.0) == 1.0
        assert value_to_bipolar_prob(-1.0) == 0.0

    def test_value_to_bipolar_prob_out_of_range(self):
        from sc_neurocore.utils.bitstreams import value_to_bipolar_prob
        from sc_neurocore.exceptions import SCEncodingError
        import pytest

        with pytest.raises(SCEncodingError):
            value_to_bipolar_prob(2.0)


class TestVecAnd:
    def test_multiply(self):
        pa, pb = 0.6, 0.4
        a, length = _bernoulli_packed(pa, 10000, seed=100)
        b, _ = _bernoulli_packed(pb, 10000, seed=200)
        result = vec_and(a, b)
        expected = pa * pb  # = 0.24
        np.testing.assert_allclose(_prob(result, length), expected, atol=0.03)
