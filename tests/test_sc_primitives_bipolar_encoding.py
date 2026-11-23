# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestBipolarEncoding from former test_sc_primitives.py

"""Focused suite: TestBipolarEncoding from former test_sc_primitives.py."""

from __future__ import annotations

import numpy as np
from sc_neurocore.accel.vector_ops import (
    pack_bitstream,
    vec_xnor,
)
from tests.sc_primitives_support import (
    _prob,
)


class TestBipolarEncoding:
    """Bipolar stochastic encoding of signed values."""

    def test_bipolar_encode_positive(self) -> None:
        """A positive value encodes and decodes within tolerance."""
        from sc_neurocore.utils.bitstreams import generate_bipolar_bitstream, bipolar_to_value

        bits = generate_bipolar_bitstream(0.6, 10000)
        val = bipolar_to_value(bits)
        np.testing.assert_allclose(val, 0.6, atol=0.05)

    def test_bipolar_encode_negative(self) -> None:
        """A negative value encodes and decodes within tolerance."""
        from sc_neurocore.utils.bitstreams import generate_bipolar_bitstream, bipolar_to_value

        bits = generate_bipolar_bitstream(-0.4, 10000)
        val = bipolar_to_value(bits)
        np.testing.assert_allclose(val, -0.4, atol=0.05)

    def test_bipolar_multiply_via_xnor(self) -> None:
        """XNOR multiplies two bipolar streams."""
        from sc_neurocore.utils.bitstreams import generate_bipolar_bitstream

        a_bits = generate_bipolar_bitstream(0.6, 20000)
        b_bits = generate_bipolar_bitstream(-0.3, 20000)
        packed_a = pack_bitstream(a_bits)
        packed_b = pack_bitstream(b_bits)
        result_packed = vec_xnor(packed_a, packed_b)
        result_prob = _prob(result_packed, 20000)
        result_bipolar = 2.0 * result_prob - 1.0
        np.testing.assert_allclose(result_bipolar, 0.6 * -0.3, atol=0.06)

    def test_bipolar_encoder_mode(self) -> None:
        """The encoder's bipolar mode decodes within tolerance."""
        from sc_neurocore.utils.bitstreams import BitstreamEncoder

        enc = BitstreamEncoder(x_min=-5.0, x_max=5.0, length=10000, mode="bipolar")
        bits = enc.encode(2.5)
        decoded = enc.decode(bits)
        np.testing.assert_allclose(decoded, 2.5, atol=0.5)

    def test_bipolar_out_of_range_raises(self) -> None:
        """A value outside [-1, 1] is rejected."""
        from sc_neurocore.utils.bitstreams import generate_bipolar_bitstream
        from sc_neurocore.exceptions import SCEncodingError
        import pytest

        with pytest.raises(SCEncodingError):
            generate_bipolar_bitstream(1.5, 100)

    def test_bipolar_to_value_empty_raises(self) -> None:
        """Decoding an empty bitstream is rejected."""
        from sc_neurocore.utils.bitstreams import bipolar_to_value
        from sc_neurocore.exceptions import SCEncodingError
        import pytest

        with pytest.raises(SCEncodingError, match="empty"):
            bipolar_to_value(np.array([], dtype=np.uint8))

    def test_value_to_bipolar_prob(self) -> None:
        """The values -1, 0 and 1 map to probabilities 0, 1/2 and 1."""
        from sc_neurocore.utils.bitstreams import value_to_bipolar_prob

        assert value_to_bipolar_prob(0.0) == 0.5
        assert value_to_bipolar_prob(1.0) == 1.0
        assert value_to_bipolar_prob(-1.0) == 0.0

    def test_value_to_bipolar_prob_out_of_range(self) -> None:
        """Mapping a value outside [-1, 1] is rejected."""
        from sc_neurocore.utils.bitstreams import value_to_bipolar_prob
        from sc_neurocore.exceptions import SCEncodingError
        import pytest

        with pytest.raises(SCEncodingError):
            value_to_bipolar_prob(2.0)
