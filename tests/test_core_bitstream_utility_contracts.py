# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for bitstream utility contracts

"""Contracts for stochastic bitstream generation, conversion, and averaging."""

from __future__ import annotations

import numpy as np
import pytest


def test_bitstream_generators_reject_invalid_probabilities() -> None:
    from sc_neurocore.utils.bitstreams import generate_bernoulli_bitstream
    from sc_neurocore.utils.bitstreams import generate_sobol_bitstream

    with pytest.raises(ValueError):
        generate_bernoulli_bitstream(-0.1, 100)
    with pytest.raises(ValueError):
        generate_sobol_bitstream(1.1, 64, seed=42)


def test_sobol_bitstream_is_binary_and_has_requested_length() -> None:
    from sc_neurocore.utils.bitstreams import generate_sobol_bitstream

    bits = generate_sobol_bitstream(0.5, 64, seed=42)

    assert bits.shape == (64,)
    assert set(np.unique(bits)).issubset({0, 1})


def test_bitstream_probability_and_unipolar_domain_boundaries() -> None:
    from sc_neurocore.utils.bitstreams import bitstream_to_probability
    from sc_neurocore.utils.bitstreams import unipolar_prob_to_value
    from sc_neurocore.utils.bitstreams import value_to_unipolar_prob

    from sc_neurocore.exceptions import SCEncodingError

    with pytest.raises(SCEncodingError, match="empty"):
        bitstream_to_probability(np.array([], dtype=np.uint8))

    assert value_to_unipolar_prob(-0.1, 0.0, 1.0) == 0.0
    assert value_to_unipolar_prob(1.1, 0.0, 1.0) == 1.0
    with pytest.raises(SCEncodingError, match="x_min"):
        value_to_unipolar_prob(0.5, 1.0, 1.0)
    with pytest.raises(SCEncodingError, match="Probability"):
        unipolar_prob_to_value(1.1, 0.0, 1.0)


def test_bitstream_encoder_rejects_unknown_mode() -> None:
    from sc_neurocore.exceptions import SCEncodingError
    from sc_neurocore.utils.bitstreams import BitstreamEncoder

    with pytest.raises(SCEncodingError, match="Unknown mode"):
        BitstreamEncoder(mode="not_a_mode")


def test_bitstream_encoder_sobol_mode_encodes_to_length() -> None:
    from sc_neurocore.utils.bitstreams import BitstreamEncoder

    stream = BitstreamEncoder(mode="sobol", length=16).encode(0.5)
    assert stream.shape == (16,)


def test_bitstream_averager_estimate_is_zero_before_any_push() -> None:
    from sc_neurocore.utils.bitstreams import BitstreamAverager

    assert BitstreamAverager(window=8).estimate() == 0.0
