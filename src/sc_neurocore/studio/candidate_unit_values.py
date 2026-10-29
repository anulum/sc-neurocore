# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Static candidate quantity dimensions and scales

"""Describe units without evaluating or converting a candidate's model values.

Pint supplies unit definitions only. Arithmetic combines their base dimensions
and scale factors; addition requires the same scale because the numeric DSL
does not convert magnitudes. Literal zero can join a unitful sum or reset.
An initially zero state or a zero-valued physical parameter retains its unit.
"""

from __future__ import annotations

import math
import tokenize
from functools import lru_cache
from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING, TypeAlias

if TYPE_CHECKING:
    import pint

from sc_neurocore.refusals import AuthoredRefusal


class CandidateUnitRefused(AuthoredRefusal):
    """An authored dimensional-admission failure, safe to place beside a field."""


@dataclass(frozen=True)
class Unit:
    """Base dimensions, numeric scale and an optional static dimensionless value.

    Parameters
    ----------
    dimensions : tuple of pairs
        Sorted base-dimension names and rational powers, omitting zero powers.
    scale : float
        Positive multiplier from the declared unit to its base units.
    constant : fractions.Fraction, optional
        Known dimensionless literal or immutable parameter value. State and
        input samples never receive a constant, even when initially zero.
    literal : bool
        Whether the constant comes solely from numeric literals. Only a
        literal zero, rather than a named zero parameter, is an additive
        identity that may take the other operand's physical unit.
    """

    dimensions: tuple[tuple[str, Fraction], ...] = ()
    scale: float = 1.0
    constant: Fraction | None = None
    literal: bool = False

    @property
    def zero(self) -> bool:
        """Tell whether this is a statically known, unscaled dimensionless zero."""
        return self.literal and not self.dimensions and self.scale == 1.0 and self.constant == 0.0

    def power(self, exponent: Fraction) -> Unit:
        """Raise dimensions and scale to a bounded static exponent.

        Parameters
        ----------
        exponent : fractions.Fraction
            Power established without evaluating state or input expressions.

        Returns
        -------
        Unit
            The resulting dimensions and scale, without a model magnitude.

        Raises
        ------
        CandidateUnitRefused
            The scale cannot be represented as a finite positive number.
        """
        try:
            scale = self.scale ** float(exponent)
        except (OverflowError, ZeroDivisionError):
            raise CandidateUnitRefused("the expression's unit scale is not finite") from None
        if not math.isfinite(scale) or scale <= 0:
            raise CandidateUnitRefused("the expression's unit scale is not finite")
        return Unit(
            tuple((name, power * exponent) for name, power in self.dimensions if power * exponent),
            scale,
        )


@dataclass(frozen=True)
class UnitSequence:
    """An authored sequence or a unit-wise array-function result.

    Parameters
    ----------
    items : tuple
        Unit descriptions of all elements, including unselected elements.
    array : bool
        Whether an array function produced this value. Plain Python sequences
        retain the runtime's refusal of numeric sequence arithmetic.
    """

    items: tuple[UnitValue, ...]
    array: bool = False


UnitValue: TypeAlias = Unit | UnitSequence | None
"""A scalar, sequence or the literal no-output argument of an array function."""


@lru_cache(maxsize=1)
def _registry() -> pint.UnitRegistry:
    """Load unit definitions lazily without importing the optional Studio extra at boot."""
    import pint

    return pint.UnitRegistry()


@lru_cache(maxsize=256)
def parse_unit(text: str) -> Unit:
    """Read a finite multiplicative unit's dimensions and scale with Pint.

    Parameters
    ----------
    text : str
        Unit expression already admitted by candidate field validation.

    Returns
    -------
    Unit
        Base dimensions and scale only. No candidate magnitude is converted.

    Raises
    ------
    CandidateUnitRefused
        An affine, logarithmic or non-finite scale cannot describe arithmetic
        performed by the numeric DSL. Invalid definitions are refused safely.
    """
    import pint

    registry = _registry()
    try:
        units = registry.parse_units(text)
        zero = registry.Quantity(0, units).to_base_units()
        one = registry.Quantity(1, units).to_base_units()
        two = registry.Quantity(2, units).to_base_units()
        scale = float(one.magnitude)
        offset = float(zero.magnitude)
        doubled = float(two.magnitude)
        dimensions = tuple(
            sorted((str(name), Fraction(str(power))) for name, power in one.dimensionality.items())
        )
    except (pint.errors.PintError, tokenize.TokenError, ValueError, TypeError, ArithmeticError):
        raise CandidateUnitRefused(
            "the declared unit cannot establish finite multiplicative dimensions"
        ) from None
    if (
        offset != 0.0
        or not math.isfinite(scale)
        or scale <= 0
        or not math.isclose(doubled, 2 * scale, rel_tol=1e-12, abs_tol=0)
    ):
        raise CandidateUnitRefused(
            "a finite multiplicative unit is required; offset and logarithmic units are not supported"
        )
    return Unit(dimensions, scale)


def scalar(value: UnitValue) -> Unit:
    """Require a scalar unit description without reading any model magnitude.

    Parameters
    ----------
    value : UnitValue
        Inferred expression result.

    Returns
    -------
    Unit
        The scalar description.

    Raises
    ------
    CandidateUnitRefused
        A sequence or None cannot supply a scalar equation value.
    """
    if not isinstance(value, Unit):
        raise CandidateUnitRefused(
            "a scalar quantity is required to establish this expression's units"
        )
    return value


def coherent(first: Unit, second: Unit) -> Unit:
    """Require equal dimensions and scales, allowing a literal additive zero.

    Parameters
    ----------
    first, second : Unit
        Quantities joined by addition, comparison, a branch or an assignment.

    Returns
    -------
    Unit
        Their common unit, with no inferred constant value.

    Raises
    ------
    CandidateUnitRefused
        Dimensions differ, or equal dimensions would require conversion.
    """
    if first.zero:
        return second
    if second.zero:
        return first
    if first.dimensions != second.dimensions:
        raise CandidateUnitRefused("incompatible dimensions; use quantities with matching units")
    # Unit definitions such as nS*mV and pA differ by binary64 rounding only.
    if not math.isclose(first.scale, second.scale, rel_tol=1e-12, abs_tol=0.0):
        raise CandidateUnitRefused(
            "incompatible unit scales; express quantities in the same scale explicitly"
        )
    return Unit(first.dimensions, first.scale)


def dimensionless(value: Unit) -> None:
    """Require an unscaled dimensionless argument or predicate.

    Parameters
    ----------
    value : Unit
        Inferred scalar argument.

    Returns
    -------
    None
        No unit or numeric value is changed.

    Raises
    ------
    CandidateUnitRefused
        The argument is physical or uses a scale such as percent or degrees.
    """
    coherent(value, Unit())


def product(first: Unit, second: Unit, *, divide: bool = False) -> Unit:
    """Combine physical dimensions and scales for multiplication or division.

    Parameters
    ----------
    first, second : Unit
        Inferred scalar operands.
    divide : bool
        Subtract the second operand's powers and divide its scale when true.

    Returns
    -------
    Unit
        The product or quotient, without computing model values.

    Raises
    ------
    CandidateUnitRefused
        Scale arithmetic leaves the finite representable range.
    """
    direction = -1 if divide else 1
    powers = dict(first.dimensions)
    for name, power in second.dimensions:
        powers[name] = powers.get(name, Fraction(0)) + direction * power
    scale = first.scale / second.scale if divide else first.scale * second.scale
    if not math.isfinite(scale) or scale <= 0:
        raise CandidateUnitRefused("the expression's unit scale is not finite")
    return Unit(tuple(sorted((name, power) for name, power in powers.items() if power)), scale)
