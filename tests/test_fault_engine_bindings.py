# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Fault-injection engine-binding contracts

"""Installed-extension contracts for byte-level fault-injection bindings."""

from __future__ import annotations

from typing import Protocol

import numpy as np
import pytest
from numpy.typing import NDArray

from tests.engine_requirement import require_engine

require_engine()
import sc_neurocore_engine as engine


class Inject(Protocol):
    """Typed public contract shared by all fault-injection functions."""

    __name__: str
    __text_signature__: str

    def __call__(
        self, bitstream: NDArray[np.uint8], ber: float, seed: int
    ) -> tuple[NDArray[np.uint8], int]: ...


FUNCTIONS: tuple[Inject, ...] = (
    engine.py_inject_bitflip_u8,
    engine.py_inject_stuck_at_0_u8,
    engine.py_inject_stuck_at_1_u8,
    engine.py_inject_dropout_u8,
    engine.py_inject_gaussian_u8,
)


def test_exported_function_names_and_signatures_are_stable() -> None:
    expected = (
        "py_inject_bitflip_u8",
        "py_inject_stuck_at_0_u8",
        "py_inject_stuck_at_1_u8",
        "py_inject_dropout_u8",
        "py_inject_gaussian_u8",
    )
    for function, name in zip(FUNCTIONS, expected, strict=True):
        assert function.__name__ == name
        assert function.__text_signature__ == "(bitstream, ber, seed)"


@pytest.mark.parametrize("inject", FUNCTIONS)
def test_zero_rate_preserves_input_without_aliasing(inject: Inject) -> None:
    bits = np.array([0, 1, 0, 1, 1, 0, 0, 1], dtype=np.uint8)
    faulted, affected = inject(bits, 0.0, 17)

    np.testing.assert_array_equal(faulted, bits)
    assert affected == 0
    assert not np.shares_memory(faulted, bits)


def test_full_rate_fault_contracts_are_exact() -> None:
    bits = np.array([0, 1, 0, 1, 1, 0, 0, 1], dtype=np.uint8)

    flipped, flip_count = engine.py_inject_bitflip_u8(bits, 1.0, 17)
    stuck_zero, zero_count = engine.py_inject_stuck_at_0_u8(bits, 1.0, 17)
    stuck_one, one_count = engine.py_inject_stuck_at_1_u8(bits, 1.0, 17)
    dropout, dropout_count = engine.py_inject_dropout_u8(bits, 1.0, 17)

    np.testing.assert_array_equal(flipped, 1 - bits)
    np.testing.assert_array_equal(stuck_zero, np.zeros_like(bits))
    np.testing.assert_array_equal(stuck_one, np.ones_like(bits))
    np.testing.assert_array_equal(dropout, stuck_zero)
    assert (flip_count, zero_count, one_count, dropout_count) == (8, 4, 4, 4)


@pytest.mark.parametrize("inject", FUNCTIONS)
def test_seeded_faults_are_reproducible(inject: Inject) -> None:
    bits = np.tile(np.array([0, 1], dtype=np.uint8), 128)

    first, first_count = inject(bits, 0.5, 0xACE1)
    second, second_count = inject(bits, 0.5, 0xACE1)

    np.testing.assert_array_equal(first, second)
    assert first_count == second_count


def test_dropout_remains_seedwise_equivalent_to_stuck_at_zero() -> None:
    bits = np.tile(np.array([0, 1, 1, 0], dtype=np.uint8), 64)

    dropout = engine.py_inject_dropout_u8(bits, 0.25, 41)
    stuck_zero = engine.py_inject_stuck_at_0_u8(bits, 0.25, 41)

    np.testing.assert_array_equal(dropout[0], stuck_zero[0])
    assert dropout[1] == stuck_zero[1]


@pytest.mark.parametrize("inject", FUNCTIONS)
def test_noncontiguous_input_preserves_type_error_contract(inject: Inject) -> None:
    bits = np.array([0, 1, 0, 1], dtype=np.uint8)

    with pytest.raises(TypeError, match=r"^The given array is not contiguous"):
        inject(bits[::-1], 0.5, 17)
