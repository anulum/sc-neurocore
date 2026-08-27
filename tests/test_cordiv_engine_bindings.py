# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — CORDIV engine-binding contracts

"""Installed-extension contracts for CORDIV and stream-length bindings."""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from tests.engine_requirement import require_engine

require_engine()
import sc_neurocore_engine as engine

extension = importlib.import_module("sc_neurocore_engine.sc_neurocore_engine")


def test_exported_function_names_signatures_and_namespace_are_stable() -> None:
    assert extension.py_cordiv.__name__ == "py_cordiv"
    assert extension.py_cordiv.__text_signature__ == "(numerator, denominator)"
    assert extension.py_adaptive_length.__name__ == "py_adaptive_length"
    assert extension.py_adaptive_length.__text_signature__ == "(epsilon, confidence)"
    assert not hasattr(engine, "py_cordiv")
    assert not hasattr(engine, "py_adaptive_length")


def test_cordiv_recurrence_and_shortest_input_contract() -> None:
    numerator = np.array([1, 0, 0, 0, 1], dtype=np.uint8)
    denominator = np.array([1, 0, 1, 0], dtype=np.uint8)

    quotient = extension.py_cordiv(numerator, denominator)

    np.testing.assert_array_equal(quotient, np.array([1, 1, 0, 0], dtype=np.uint8))
    assert quotient.dtype == np.uint8
    assert not np.shares_memory(quotient, numerator)
    assert not np.shares_memory(quotient, denominator)


def test_empty_cordiv_inputs_return_a_distinct_empty_array() -> None:
    numerator = np.array([], dtype=np.uint8)
    denominator = np.array([], dtype=np.uint8)

    quotient = extension.py_cordiv(numerator, denominator)

    assert quotient.shape == (0,)
    assert quotient.dtype == np.uint8
    assert not np.shares_memory(quotient, numerator)
    assert not np.shares_memory(quotient, denominator)


@pytest.mark.parametrize("position", ("numerator", "denominator"))
def test_noncontiguous_inputs_preserve_type_error_contract(position: str) -> None:
    contiguous = np.ones(4, dtype=np.uint8)
    noncontiguous = np.arange(8, dtype=np.uint8)[::2]
    numerator = noncontiguous if position == "numerator" else contiguous
    denominator = noncontiguous if position == "denominator" else contiguous

    with pytest.raises(TypeError, match=r"^The given array is not contiguous or is misaligned\.$"):
        extension.py_cordiv(numerator, denominator)


@pytest.mark.parametrize(
    ("numerator", "denominator"),
    (
        (np.array([1, 0], dtype=np.int64), np.array([1, 1], dtype=np.uint8)),
        (np.array([[1, 0]], dtype=np.uint8), np.array([1, 1], dtype=np.uint8)),
    ),
)
def test_dtype_and_rank_mismatches_preserve_type_error_contract(
    numerator: NDArray[np.integer[Any]], denominator: NDArray[np.integer[Any]]
) -> None:
    with pytest.raises(TypeError, match=r"^'ndarray' object is not an instance of 'ndarray'"):
        extension.py_cordiv(numerator, denominator)


def test_adaptive_length_preserves_bounds_and_monotonicity() -> None:
    coarse = extension.py_adaptive_length(0.1, 0.95)
    fine = extension.py_adaptive_length(0.01, 0.95)

    assert coarse == 256
    assert fine == 32768
    assert fine > coarse
    assert coarse & (coarse - 1) == 0
    assert fine & (fine - 1) == 0


@pytest.mark.parametrize(
    ("epsilon", "confidence", "expected"),
    (
        (0.0, 0.95, 65536),
        (-1.0, 0.95, 65536),
        (0.1, 1.0, 65536),
        (float("nan"), 0.95, 64),
        (0.1, float("nan"), 64),
    ),
)
def test_adaptive_length_preserves_boundary_behavior(
    epsilon: float, confidence: float, expected: int
) -> None:
    assert extension.py_adaptive_length(epsilon, confidence) == expected
