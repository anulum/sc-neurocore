# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Stochastic-correlation validation tests

"""Fail-closed shape, binary-domain, length, and threshold contracts."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from sc_neurocore.core import correlation_diagnostic, estimate_scc, observed_and_bias


def test_length_mismatch_rejected() -> None:
    """Streams of different length are refused by the estimator."""
    with pytest.raises(ValueError):
        estimate_scc([1, 0, 1], [1, 0])


def test_empty_rejected() -> None:
    """An empty stream is refused."""
    with pytest.raises(ValueError):
        estimate_scc([], [])


def test_non_binary_rejected() -> None:
    """An integer other than 0 and 1 is refused."""
    with pytest.raises(ValueError):
        estimate_scc([0, 2, 1], [0, 1, 1])


def test_non_1d_rejected() -> None:
    """A two-dimensional array is refused."""
    with pytest.raises(ValueError):
        estimate_scc([[1, 0], [0, 1]], [[1, 0], [0, 1]])


def test_observed_bias_length_mismatch_rejected() -> None:
    """Streams of different length are refused by the bias measurement."""
    with pytest.raises(ValueError):
        observed_and_bias([1, 0, 1], [1, 0])


def test_diagnostic_length_mismatch_rejected() -> None:
    """Streams of different length are refused by the diagnostic."""
    with pytest.raises(ValueError):
        correlation_diagnostic([1, 0, 1], [1, 0])


def test_diagnostic_negative_threshold_rejected() -> None:
    """A negative bias threshold is refused."""
    with pytest.raises(ValueError):
        correlation_diagnostic([1, 0, 1], [0, 1, 1], bias_threshold=-0.1)


_NON_BINARY_INPUTS = {
    "fractional floats": np.array([0.9, 1.9]),
    "wrapping unsigned integers": np.array([256, 257], dtype=np.uint16),
    "negative integer": np.array([-1, 0], dtype=np.int64),
    "integer beyond one byte": np.array([0, 2**40], dtype=np.int64),
    "not a number": np.array([0.0, np.nan]),
    "infinity": np.array([1.0, np.inf]),
    "values near one": np.array([1.0 - 1e-12, 1.0]),
}

_INADMISSIBLE_TYPES = {
    "text": ["0", "1"],
    "bytes": [b"0", b"1"],
    "objects": np.array([0, 1, None], dtype=object),
    "complex": np.array([0j, 1 + 0j]),
    "integers too large for a machine word": [0, 2**70],
}

_BINARY_REPRESENTATIONS = {
    "python integers": ([1, 0, 1, 1, 0, 0, 1, 0], [1, 1, 0, 1, 0, 1, 0, 0]),
    "python booleans": (
        [True, False, True, True, False, False, True, False],
        [True, True, False, True, False, True, False, False],
    ),
    "boolean array": (
        np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.bool_),
        np.array([1, 1, 0, 1, 0, 1, 0, 0], dtype=np.bool_),
    ),
    "signed integers": (
        np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.int64),
        np.array([1, 1, 0, 1, 0, 1, 0, 0], dtype=np.int64),
    ),
    "exact floats": (
        np.array([1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0]),
        np.array([1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0]),
    ),
}

_PUBLIC_ENTRY_POINTS = (estimate_scc, observed_and_bias, correlation_diagnostic)


@pytest.mark.parametrize("entry_point", _PUBLIC_ENTRY_POINTS)
@pytest.mark.parametrize("case", sorted(_NON_BINARY_INPUTS))
def test_values_other_than_zero_and_one_are_refused_before_conversion(
    entry_point: Callable[..., object], case: str
) -> None:
    """A value that only becomes a bit through a cast to one byte is refused."""
    valid = np.array([0, 1], dtype=np.uint8)
    with pytest.raises(ValueError, match="only 0/1 bits"):
        entry_point(_NON_BINARY_INPUTS[case], valid)
    with pytest.raises(ValueError, match="only 0/1 bits"):
        entry_point(valid, _NON_BINARY_INPUTS[case])


@pytest.mark.parametrize("entry_point", _PUBLIC_ENTRY_POINTS)
@pytest.mark.parametrize("case", sorted(_INADMISSIBLE_TYPES))
def test_inadmissible_element_types_are_refused(
    entry_point: Callable[..., object], case: str
) -> None:
    """Text, bytes, objects and complex numbers are not bit containers."""
    valid = np.array([0, 1], dtype=np.uint8)
    with pytest.raises(ValueError, match="boolean, integer or real"):
        entry_point(_INADMISSIBLE_TYPES[case], valid)
    with pytest.raises(ValueError, match="boolean, integer or real"):
        entry_point(valid, _INADMISSIBLE_TYPES[case])


@pytest.mark.parametrize("entry_point", _PUBLIC_ENTRY_POINTS)
def test_nested_sequences_of_unequal_length_are_refused_by_name(
    entry_point: Callable[..., object],
) -> None:
    """A ragged nested sequence is refused with a message that names the argument."""
    valid = np.array([0, 1, 1, 0], dtype=np.uint8)
    with pytest.raises(ValueError, match="bits_a must be one-dimensional"):
        entry_point([[0, 1], [1]], valid)
    with pytest.raises(ValueError, match="bits_b must be one-dimensional"):
        entry_point(valid, [[0, 1], [1]])


def test_scalar_input_is_refused() -> None:
    """A scalar is not a stream and is refused."""
    with pytest.raises(ValueError, match="one-dimensional"):
        estimate_scc(np.uint8(1), np.uint8(1))


@pytest.mark.parametrize("case", sorted(_BINARY_REPRESENTATIONS))
def test_every_exact_binary_representation_gives_the_same_result(case: str) -> None:
    """The result depends on the bits, not on the container that holds them."""
    reference_a = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.uint8)
    reference_b = np.array([1, 1, 0, 1, 0, 1, 0, 0], dtype=np.uint8)
    bits_a, bits_b = _BINARY_REPRESENTATIONS[case]
    assert estimate_scc(bits_a, bits_b) == estimate_scc(reference_a, reference_b)
    assert observed_and_bias(bits_a, bits_b) == observed_and_bias(reference_a, reference_b)
    assert correlation_diagnostic(bits_a, bits_b) == correlation_diagnostic(
        reference_a, reference_b
    )


def test_refused_input_is_left_unchanged() -> None:
    """Validation reads its input and never writes to it."""
    fractional = np.array([0.9, 1.9])
    before = fractional.copy()
    with pytest.raises(ValueError):
        estimate_scc(fractional, np.array([0, 1], dtype=np.uint8))
    assert np.array_equal(fractional, before)
