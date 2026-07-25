# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Photonic co-design bitstream contracts

"""Bitstream encoding contracts for stochastic photonic co-design."""

import numpy as np
import pytest

from sc_neurocore.bridges import encode_bitstream_bank


def test_encode_bitstream_bank_is_deterministic_and_density_bounded() -> None:
    first = encode_bitstream_bank(
        [0.25, 0.75],
        bitstream_length=512,
        seed=0x1234,
        names=["low", "high"],
    )
    second = encode_bitstream_bank(
        [0.25, 0.75],
        bitstream_length=512,
        seed=0x1234,
        names=["low", "high"],
    )

    assert first == second
    assert first[0].name == "low"
    assert first[1].measured_probability > first[0].measured_probability
    assert first[0].density_error < 0.08
    assert first[1].transitions > 0


@pytest.mark.parametrize(
    ("probabilities", "match"),
    [
        ([[0.5]], "one-dimensional"),
        ([-0.1], r"\[0, 1\]"),
        ([1.1], r"\[0, 1\]"),
    ],
)
def test_encode_bitstream_bank_rejects_invalid_probabilities(
    probabilities: list[float] | list[list[float]], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        encode_bitstream_bank(probabilities, bitstream_length=128, seed=1)


def test_transition_count_of_single_bit_is_zero() -> None:
    from sc_neurocore.bridges.photonic_codesign import _transition_count

    assert _transition_count(np.array([1], dtype=np.uint8)) == 0


def test_encode_bitstream_bank_rejects_nonpositive_length_and_name_mismatch() -> None:
    with pytest.raises(ValueError, match="bitstream_length must be positive"):
        encode_bitstream_bank([0.5], bitstream_length=0, seed=1)
    with pytest.raises(ValueError, match="names length must match"):
        encode_bitstream_bank([0.5, 0.5], bitstream_length=64, seed=1, names=["only_one"])
