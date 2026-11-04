# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Exact word polynomial regressions

"""Check exact polynomial arithmetic, signed slices and bounded expansion."""

from __future__ import annotations

import itertools

import pytest

from tools.function_synthesis_polynomial import ExactWordPolynomials, Polynomial


def _evaluate(ring: ExactWordPolynomials, polynomial: Polynomial, bits: dict[str, int]) -> int:
    """Evaluate retained expressions independently with Python integer multiplication and floor."""

    def atom_value(identity: int) -> int:
        definition = ring.atom_definitions[identity]
        if definition[0] == "bit":
            return bits[definition[1]]
        return _evaluate(ring, dict(definition[2]), bits) // (1 << definition[1])

    result = 0
    for monomial, coefficient in polynomial.items():
        term = coefficient
        for identity in monomial:
            term *= atom_value(identity)
        result += term
    return result


def test_signed_slices_wrap_and_floor_agree_on_the_entire_small_domain() -> None:
    """Compare every six-bit assignment against independent signed integer arithmetic."""
    ring = ExactWordPolynomials()
    names = [f"x[{index}]" for index in range(6)]
    word = ring.add(
        *(ring.scale(ring.boolean(name), 1 << index) for index, name in enumerate(names))
    )
    signed = ring.signed_wrap(word, 6)
    shifted = ring.floor(signed, 2)
    low = ring.slice(word, 6, 1, 3, False)
    signed_low = ring.slice(word, 6, 1, 3, True)
    high = ring.slice(word, 6, 2, 5, True)
    for assignment in itertools.product((0, 1), repeat=6):
        bits = dict(zip(names, assignment, strict=True))
        unsigned = sum(value << index for index, value in enumerate(assignment))
        expected = unsigned if unsigned < 32 else unsigned - 64
        fragment = (unsigned >> 1) & 7
        assert _evaluate(ring, signed, bits) == expected
        assert _evaluate(ring, shifted, bits) == expected // 4
        assert _evaluate(ring, low, bits) == fragment
        assert _evaluate(ring, signed_low, bits) == (fragment if fragment < 4 else fragment - 8)
        assert _evaluate(ring, high, bits) == expected // 4


def test_exact_distributivity_boolean_idempotence_and_signed_coefficients() -> None:
    """Prove a universal product identity and preserve a real counterexample to a changed bit."""
    ring = ExactWordPolynomials()
    a, b = ring.boolean("a"), ring.boolean("b")
    square = ring.multiply(ring.add(a, b), ring.add(a, b))
    expected = ring.add(a, b, ring.scale(ring.multiply(a, b), 2))
    difference = ring.add(square, ring.scale(expected, -1))
    assert difference == {} and ring.boolean_witness(difference) is None
    assert ring.scale(a, 0) == ring.multiply(a, {}) == {}
    assert ring.bounds(ring.add(ring.constant(-3), ring.scale(a, -4))) == (-7, -3)
    changed = ring.add(a, ring.scale(b, -1))
    witness = ring.boolean_witness(changed)
    assert witness is not None and _evaluate(ring, changed, witness) != 0
    assert ring.describe(expected)


def test_nested_floor_and_negative_rounding_match_integer_arithmetic() -> None:
    """Check nested floor normalization without rounding negative inputs toward zero."""
    ring = ExactWordPolynomials()
    word = ring.add(*(ring.scale(ring.boolean(f"x{i}"), 1 << i) for i in range(4)))
    shifted = ring.floor(word, 1)
    assert ring.floor(shifted, 1) == ring.floor(word, 2)
    assert ring.floor(word, 0) == word
    assert ring.floor(ring.boolean("one-independent-bit"), 1) == {}
    assert ring.floor(ring.constant(-5), 1) == ring.constant(-3)
    assert ring.floor(ring.constant(8), 2) == ring.constant(2)
    assert ring.signed_wrap(ring.constant(2), 4) == ring.constant(2)
    assert ring.residue(ring.constant(-1), 8) == ring.constant(255)
    non_boolean_floor = ring.floor(ring.add(ring.boolean("a"), ring.boolean("b")), 1)
    assert ring.boolean_witness(non_boolean_floor) is None
    assert ring.floor(non_boolean_floor, 1) == {}


@pytest.mark.parametrize("budget", [True, 0, -1, 100001])
def test_invalid_exact_algebra_budget_is_refused(budget: int) -> None:
    """Keep expression growth bounded before an actual proof starts."""
    with pytest.raises(ValueError, match="budget"):
        ExactWordPolynomials(budget)


def test_finite_algebra_budgets_refuse_actual_expansion() -> None:
    """Exercise finite atom, sum and product growth limits through public algebra operations."""
    atoms = ExactWordPolynomials(1)
    atoms.boolean("x")
    with pytest.raises(ValueError, match="atom limit"):
        atoms.boolean("y")
    sums = ExactWordPolynomials(2)
    a, b = sums.boolean("a"), sums.boolean("b")
    with pytest.raises(ValueError, match="term limit"):
        sums.add(sums.constant(1), a, b)
    products = ExactWordPolynomials(3)
    a, b, c = (products.boolean(name) for name in ["a", "b", "c"])
    with pytest.raises(ValueError, match="multiplication limit"):
        products.multiply(products.add(a, b), products.add(products.constant(1), c))


def test_invalid_floor_slice_and_signed_width_are_refused() -> None:
    """Reject impossible word geometry while leaving the algebra usable afterward."""
    ring = ExactWordPolynomials()
    with pytest.raises(ValueError, match="nonnegative"):
        ring.floor(ring.constant(1), -1)
    with pytest.raises(ValueError, match="positive"):
        ring.signed_wrap(ring.constant(1), 0)
    with pytest.raises(ValueError, match="slice"):
        ring.slice(ring.constant(1), 8, 2, 8, False)
    assert ring.constant(0) == {}
