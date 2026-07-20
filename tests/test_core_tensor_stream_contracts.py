# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for TensorStream contracts

"""Contracts for TensorStream domain conversions and stochastic primitives."""

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.core import bipolar_decode, bipolar_encode, bipolar_mac, bipolar_multiply
from sc_neurocore.core.tensor_stream import TensorStream


def test_core_exports_signed_stochastic_primitives() -> None:
    bits = bipolar_encode(1.0, 8, rng=np.random.default_rng(42))
    product = bipolar_multiply(bits, bits)
    decoded = bipolar_decode(product)
    result = bipolar_mac(np.array([1.0]), np.array([[1.0]]), L=8, seed=42)

    assert decoded == 1.0
    assert np.isclose(result[0], 1.0)


def test_tensor_stream_bitstream_conversion_tracks_probability() -> None:
    probabilities = np.array([0.25, 0.75])
    stream = TensorStream.from_prob(probabilities)

    bitstream = stream.to_bitstream(length=10_000)

    assert bitstream.shape == (2, 10_000)
    assert bitstream.dtype == np.uint8
    np.testing.assert_allclose(np.mean(bitstream, axis=-1), probabilities, atol=0.02)


def test_tensor_stream_quantum_conversion_is_normalised() -> None:
    probabilities = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    stream = TensorStream.from_prob(probabilities)

    quantum = stream.to_quantum()

    norms = np.abs(quantum[..., 0]) ** 2 + np.abs(quantum[..., 1]) ** 2
    np.testing.assert_allclose(norms, 1.0)


def test_tensor_stream_quantum_to_probability_uses_born_rule() -> None:
    quantum = np.array(
        [
            [1.0, 0.0],
            [np.sqrt(0.5), np.sqrt(0.5)],
            [0.0, 1.0],
        ],
        dtype=complex,
    )

    probabilities = TensorStream(data=quantum, domain="quantum").to_prob()

    np.testing.assert_allclose(probabilities, [0.0, 0.5, 1.0])


def test_tensor_stream_to_prob_falls_back_for_unknown_domain() -> None:
    """Return raw data when no specialised domain decoder applies."""
    stream = TensorStream(data=np.array([0.2, 0.8]), domain="spike")

    np.testing.assert_array_equal(stream.to_prob(), np.array([0.2, 0.8]))


def test_tensor_stream_rejects_unknown_domain_for_bitstream_conversion() -> None:
    stream = TensorStream(data=np.zeros(3), domain="unknown")

    with pytest.raises(ValueError):
        stream.to_bitstream()
