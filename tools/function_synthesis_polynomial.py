# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Exact bounded integer word identities

"""Check universal integer word identities with explicit floor and wrap semantics."""

from __future__ import annotations

import math
import time
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Literal, TypeAlias

Monomial = tuple[int, ...]
Polynomial = dict[Monomial, int]
PolynomialKey: TypeAlias = tuple[tuple[Monomial, int], ...]
AtomKey: TypeAlias = tuple[Literal["bit"], str] | tuple[Literal["floor"], int, PolynomialKey]


class ExactWordPolynomials:
    """Normalize universal integer identities; refuse unsupported complexity."""

    def __init__(
        self,
        maximum_terms: int = 100_000,
        *,
        observer: Callable[[str, int], None] | None = None,
    ) -> None:
        """Retain the expression-size guard and optional materialised-work observer.

        Parameters
        ----------
        maximum_terms : int, optional
            Maximum terms in one expression, between one and 100000.
        observer : callable or None, optional
            Receive each completed operation's name and result cardinality.
            It may refuse an exhausted enclosing deadline; it never receives
            mutable expressions or changes their mathematical interpretation.
        """
        if isinstance(maximum_terms, bool) or not 1 <= maximum_terms <= 100_000:
            raise ValueError("The exact word term budget must lie between one and 100000")
        self.maximum_terms = maximum_terms
        self.observer = observer
        self.atom_keys: dict[AtomKey, int] = {}
        self.atom_definitions: dict[int, AtomKey] = {}
        self.atom_bounds: dict[int, tuple[int, int]] = {}
        self.boolean_atoms: set[int] = set()

    def constant(self, value: int) -> Polynomial:
        """Represent an exact integer constant, including canonical zero."""
        observed_result: Polynomial = {(): value} if value else {}
        if self.observer is not None:
            self.observer("constant", len(observed_result))
        return observed_result

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
        observed_result = result
        if self.observer is not None:
            self.observer("boolean", len(observed_result))
        return observed_result

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
        observed_result = output
        if self.observer is not None:
            self.observer("add", len(observed_result))
        return observed_result

    def scale(self, polynomial: Polynomial, coefficient: int) -> Polynomial:
        """Multiply every coefficient by an exact signed integer."""
        observed_result = {
            term: value * coefficient for term, value in polynomial.items() if value * coefficient
        }
        if self.observer is not None:
            self.observer("scale", len(observed_result))
        return observed_result

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
        observed_result = {term: coefficient for term, coefficient in result.items() if coefficient}
        if self.observer is not None:
            self.observer("multiply", len(observed_result))
        return observed_result

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
            observed_result = dict(polynomial)
            if self.observer is not None:
                self.observer("floor", len(observed_result))
            return observed_result
        if len(polynomial) == 1:
            term, coefficient = next(iter(polynomial.items()))
            if len(term) == 1 and coefficient == 1:
                definition = self.atom_definitions[term[0]]
                if definition[0] == "floor":
                    nested_shift = int(definition[1])
                    nested = dict(definition[2])
                    observed_result = self.floor(nested, shift + nested_shift)
                    if self.observer is not None:
                        self.observer("floor", len(observed_result))
                    return observed_result
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
            observed_result = quotient
            if self.observer is not None:
                self.observer("floor", len(observed_result))
            return observed_result
        low, high = self.bounds(remainder)
        a, b = low // divisor, high // divisor
        if a == b:
            observed_result = self.add(quotient, self.constant(a))
            if self.observer is not None:
                self.observer("floor", len(observed_result))
            return observed_result
        atom = self._atom(("floor", shift, self.key(remainder)), (a, b))
        observed_result = self.add(quotient, atom)
        if self.observer is not None:
            self.observer("floor", len(observed_result))
        return observed_result

    def signed_wrap(self, polynomial: Polynomial, width: int) -> Polynomial:
        """Interpret an exact integer modulo the declared signed word width."""
        if width <= 0:
            raise ValueError("Signed word width must be positive")
        low, high = self.bounds(polynomial)
        half = 1 << (width - 1)
        if -half <= low and high < half:
            observed_result = dict(polynomial)
            if self.observer is not None:
                self.observer("signed_wrap", len(observed_result))
            return observed_result
        carry = self.floor(self.add(polynomial, self.constant(half)), width)
        observed_result = self.add(polynomial, self.scale(carry, -(1 << width)))
        if self.observer is not None:
            self.observer("signed_wrap", len(observed_result))
        return observed_result

    def slice(
        self, polynomial: Polynomial, width: int, low: int, high: int, signed: bool
    ) -> Polynomial:
        """Read the actual finite bit slice, including signed extension semantics."""
        if not 0 <= low <= high < width:
            raise ValueError("Invalid exact word slice")
        if signed and high == width - 1:
            observed_result = self.floor(self.signed_wrap(polynomial, width), low)
            if self.observer is not None:
                self.observer("slice", len(observed_result))
            return observed_result
        shifted = self.floor(polynomial, low)
        count = high - low + 1
        if signed:
            observed_result = self.signed_wrap(shifted, count)
            if self.observer is not None:
                self.observer("slice", len(observed_result))
            return observed_result
        carry = self.floor(polynomial, high + 1)
        observed_result = self.add(shifted, self.scale(carry, -(1 << count)))
        if self.observer is not None:
            self.observer("slice", len(observed_result))
        return observed_result

    def residue(self, polynomial: Polynomial, width: int) -> Polynomial:
        """Reduce coefficients modulo a complete unsigned result word."""
        modulus = 1 << width
        observed_result = {
            term: coefficient % modulus
            for term, coefficient in polynomial.items()
            if coefficient % modulus
        }
        if self.observer is not None:
            self.observer("residue", len(observed_result))
        return observed_result

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


@dataclass(slots=True)
class WordProofWork:
    """Measure materialised algebra work under one inherited wall deadline.

    Parameters
    ----------
    deadline : float
        Absolute monotonic enclosing case deadline, never extended by a reader.
    """

    deadline: float
    began: float = field(default_factory=time.monotonic, init=False)
    calls: Counter[str] = field(default_factory=Counter, init=False)
    terms: Counter[str] = field(default_factory=Counter, init=False)
    peak: int = field(default=0, init=False)

    def check(self) -> None:
        """Refuse CPU-only work after the same enclosing case allowance expires."""
        if time.monotonic() > self.deadline:
            raise ValueError("Grouped word proof exhausted the shared case budget")

    def observe(self, operation: str, terms: int) -> None:
        """Count each materialised result; cumulative occurrences are evidence only.

        Parameters
        ----------
        operation : str
            Exact arithmetic operation which completed.
        terms : int
            Actual cardinality of its materialised expression.

        Raises
        ------
        ValueError
            If this computation exhausted its inherited wall-clock allowance.
        """
        self.calls[operation] += 1
        self.terms[operation] += terms
        self.peak = max(self.peak, terms)
        self.check()

    def to_dict(self) -> dict[str, object]:
        """Report actual work separately from the deterministic arithmetic certificate."""
        return {
            "operation_calls": dict(sorted(self.calls.items())),
            "operation_term_occurrences": dict(sorted(self.terms.items())),
            "peak_expression_terms": self.peak,
            "expression_term_limit": 100_000,
            "elapsed_seconds": time.monotonic() - self.began,
        }

    def check_record(self, retained: Mapping[str, object]) -> float:
        """Validate retained work against this actual computation without restamping it.

        Parameters
        ----------
        retained : mapping of str to object
            Original producer or reader operation counts and measured elapsed time.

        Returns
        -------
        float
            Original bounded elapsed time, for the enclosing shared-budget check.

        Raises
        ------
        ValueError
            For invalid or changed counts, or explicit retained/current exhaustion.
        """
        self.check()
        elapsed = retained.get("elapsed_seconds")
        if (
            isinstance(elapsed, bool)
            or not isinstance(elapsed, (float, int))
            or not math.isfinite(elapsed)
            or elapsed < 0
        ):
            raise ValueError("Retained grouped arithmetic work has invalid elapsed time")
        if elapsed > 3300:
            raise ValueError("Retained grouped arithmetic work exhausted the shared case budget")
        actual = self.to_dict()
        if retained.keys() != actual.keys() or any(
            retained[name] != value for name, value in actual.items() if name != "elapsed_seconds"
        ):
            raise ValueError(
                "Retained grouped arithmetic work differs from actual bounded computation"
            )
        return float(elapsed)
