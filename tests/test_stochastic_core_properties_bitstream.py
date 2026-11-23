# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (bitstream) from former test_stochastic_core_properties.py

"""Property-based checks of Bernoulli bitstream generation and decoding."""

from __future__ import annotations

import numpy as np
from hypothesis import (
    given,
    settings,
    strategies as st,
)
from sc_neurocore import (
    bitstream_to_probability,
    generate_bernoulli_bitstream,
)


@given(p=st.floats(min_value=0.0, max_value=1.0))
@settings(max_examples=50)
def test_bernoulli_bitstream_values_are_binary(p: float) -> None:
    """Any probability gives a binary bitstream."""
    bs = generate_bernoulli_bitstream(p, length=256)
    assert set(np.unique(bs)).issubset({0, 1})


@given(p=st.floats(min_value=0.0, max_value=1.0))
@settings(max_examples=50)
def test_bitstream_roundtrip_within_tolerance(p: float) -> None:
    """Any probability round-trips within tolerance."""
    bs = generate_bernoulli_bitstream(p, length=4096)
    recovered = bitstream_to_probability(bs)
    assert abs(recovered - p) < 0.1


@given(length=st.integers(min_value=1, max_value=2048))
@settings(max_examples=30)
def test_bernoulli_bitstream_correct_length(length: int) -> None:
    """Any length gives a bitstream of that length."""
    bs = generate_bernoulli_bitstream(0.5, length=length)
    assert len(bs) == length


@given(p=st.just(0.0))
def test_zero_probability_all_zeros(p: float) -> None:
    """Probability zero gives all zeros."""
    bs = generate_bernoulli_bitstream(p, length=1024)
    assert np.sum(bs) == 0


@given(p=st.just(1.0))
def test_one_probability_all_ones(p: float) -> None:
    """Probability one gives all ones."""
    bs = generate_bernoulli_bitstream(p, length=1024)
    assert np.sum(bs) == 1024
