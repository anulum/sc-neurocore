# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Exact bounded integer word identities

"""Check universal integer word identities with explicit floor and wrap semantics."""

from __future__ import annotations

from collections import Counter
from typing import Literal, TypeAlias

Monomial = tuple[int, ...]
Polynomial = dict[Monomial, int]
PolynomialKey: TypeAlias = tuple[tuple[Monomial, int], ...]
AtomKey: TypeAlias = tuple[Literal["bit"], str] | tuple[Literal["floor"], int, PolynomialKey]


class ExactWordPolynomials:
    """Normalize universal integer identities; refuse unsupported complexity."""

    def __init__(self, maximum_terms: int = 100_000) -> None:
        """Retain a finite algebra budget and exact canonical atom definitions."""
        if isinstance(maximum_terms, bool) or not 1 <= maximum_terms <= 100_000:
            raise ValueError("The exact word term budget must lie between one and 100000")
        self.maximum_terms = maximum_terms
        self.atom_keys: dict[AtomKey, int] = {}
        self.atom_definitions: dict[int, AtomKey] = {}
        self.atom_bounds: dict[int, tuple[int, int]] = {}
        self.boolean_atoms: set[int] = set()

    def constant(self, value: int) -> Polynomial:
        """Represent an exact integer constant, including canonical zero."""
        return {(): value} if value else {}

    def _atom(self, key: AtomKey, bounds: tuple[int, int]) -> Polynomial:
        """Intern one complete Boolean or floor expression with proved interval bounds."""
        identity = self.atom_keys.get(key)
        if identity is None:
            if len(self.atom_keys) >= self.maximum_terms:
                raise ValueError("Exact atom limit exceeded")
            identity = len(self.atom_keys) + 1
            self.atom_keys[key] = identity
            self.atom_definitions[identity] = key
            self.atom_bounds[identity] = bounds
        return {(identity,): 1}

    def boolean(self, name: str) -> Polynomial:
        """Represent one universally independent bit constrained only to zero or one."""
        result = self._atom(("bit", name), (0, 1))
        self.boolean_atoms.add(next(iter(result))[0])
        return result

    def key(self, polynomial: Polynomial) -> PolynomialKey:
        """Return an exact ordered polynomial identity without digest-only equality."""
        return tuple(sorted(polynomial.items()))

    def add(self, *polynomials: Polynomial) -> Polynomial:
        """Combine integer coefficients exactly and refuse excess expression size."""
        result: Counter[Monomial] = Counter()
        for polynomial in polynomials:
            for term, coefficient in polynomial.items():
                result[term] += coefficient
        output = {term: coefficient for term, coefficient in result.items() if coefficient}
        if len(output) > self.maximum_terms:
            raise ValueError("Exact polynomial term limit exceeded")
        return output

    def scale(self, polynomial: Polynomial, coefficient: int) -> Polynomial:
        """Multiply every coefficient by an exact signed integer."""
        return {
            term: value * coefficient for term, value in polynomial.items() if value * coefficient
        }

    def multiply(self, left: Polynomial, right: Polynomial) -> Polynomial:
        """Expand exact integer products with Boolean idempotence only for real bits."""
        result: Counter[Monomial] = Counter()
        for a, x in left.items():
            for b, y in right.items():
                factors = sorted(a + b)
                term = tuple(
                    value
                    for index, value in enumerate(factors)
                    if value not in self.boolean_atoms or index == 0 or factors[index - 1] != value
                )
                result[term] += x * y
                if len(result) > self.maximum_terms:
                    raise ValueError("Exact polynomial multiplication limit exceeded")
        return {term: coefficient for term, coefficient in result.items() if coefficient}

    def bounds(self, polynomial: Polynomial) -> tuple[int, int]:
        """Bound every actual integer atom by conservative interval arithmetic."""
        low = high = 0
        for term, coefficient in polynomial.items():
            a = b = 1
            for atom in term:
                c, d = self.atom_bounds[atom]
                products = (a * c, a * d, b * c, b * d)
                a, b = min(products), max(products)
            a, b = sorted((a * coefficient, b * coefficient))
            low += a
            high += b
        return low, high

    def floor(self, polynomial: Polynomial, shift: int) -> Polynomial:
        """Divide by a power of two with mathematical floor for negative values too."""
        if shift < 0:
            raise ValueError("A floor shift must be nonnegative")
        if shift == 0:
            return dict(polynomial)
        if len(polynomial) == 1:
            term, coefficient = next(iter(polynomial.items()))
            if len(term) == 1 and coefficient == 1:
                definition = self.atom_definitions[term[0]]
                if definition[0] == "floor":
                    nested_shift = int(definition[1])
                    nested = dict(definition[2])
                    return self.floor(nested, shift + nested_shift)
        divisor = 1 << shift
        quotient: Polynomial = {}
        remainder: Polynomial = {}
        for term, coefficient in polynomial.items():
            q, r = divmod(coefficient, divisor)
            if q:
                quotient[term] = q
            if r:
                remainder[term] = r
        if not remainder:
            return quotient
        low, high = self.bounds(remainder)
        a, b = low // divisor, high // divisor
        if a == b:
            return self.add(quotient, self.constant(a))
        atom = self._atom(("floor", shift, self.key(remainder)), (a, b))
        return self.add(quotient, atom)

    def signed_wrap(self, polynomial: Polynomial, width: int) -> Polynomial:
        """Interpret an exact integer modulo the declared signed word width."""
        if width <= 0:
            raise ValueError("Signed word width must be positive")
        low, high = self.bounds(polynomial)
        half = 1 << (width - 1)
        if -half <= low and high < half:
            return dict(polynomial)
        carry = self.floor(self.add(polynomial, self.constant(half)), width)
        return self.add(polynomial, self.scale(carry, -(1 << width)))

    def slice(
        self, polynomial: Polynomial, width: int, low: int, high: int, signed: bool
    ) -> Polynomial:
        """Read the actual finite bit slice, including signed extension semantics."""
        if not 0 <= low <= high < width:
            raise ValueError("Invalid exact word slice")
        if signed and high == width - 1:
            return self.floor(self.signed_wrap(polynomial, width), low)
        shifted = self.floor(polynomial, low)
        count = high - low + 1
        if signed:
            return self.signed_wrap(shifted, count)
        carry = self.floor(polynomial, high + 1)
        return self.add(shifted, self.scale(carry, -(1 << count)))

    def residue(self, polynomial: Polynomial, width: int) -> Polynomial:
        """Reduce coefficients modulo a complete unsigned result word."""
        modulus = 1 << width
        return {
            term: coefficient % modulus
            for term, coefficient in polynomial.items()
            if coefficient % modulus
        }

    def boolean_witness(self, difference: Polynomial) -> dict[str, int] | None:
        """Return a separating bit assignment only when every atom is an actual bit."""
        if not difference:
            return None
        if any(atom not in self.boolean_atoms for term in difference for atom in term):
            return None
        chosen = min(difference, key=lambda term: (len(term), term))
        return {
            str(definition[1]): int(atom in chosen)
            for atom, definition in self.atom_definitions.items()
            if atom in self.boolean_atoms
        }

    def describe(self, polynomial: Polynomial) -> list[dict[str, object]]:
        """Serialize exact coefficients and full atom definitions for retained evidence."""
        return [
            {"coefficient": coefficient, "factors": [self.atom_definitions[atom] for atom in term]}
            for term, coefficient in sorted(polynomial.items())
        ]
