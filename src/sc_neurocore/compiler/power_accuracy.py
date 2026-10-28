# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Power domains, independent measurements and arithmetic error bounds
"""Measure literal and dynamic powers without confusing a sample with a global bound.

On the normalized logarithm, ``t < 1/3``. Coefficient quantization, the geometric
Horner recurrence, quotient truncation and the omitted atanh tail total less
than 16 Q56 units. The exp2 Taylor tail, 21 coefficient errors and 20 product
truncations total less than 32 units. The clipped exponential is Lipschitz with
constant ``word_limit*ln(2)``. These bounds include independently quantized
literal exponents and one final half-LSB rounding; invalid-domain sentinels
are excluded from accuracy measurements.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal, localcontext
from fractions import Fraction
from functools import lru_cache
from itertools import product

from .lut_accuracy import LutAccuracy
from .power_units import INTERNAL_FRACTION, PowerExponent, exponent_format, power_code


@dataclass(frozen=True, slots=True)
class PowerAccuracy(LutAccuracy):
    """A power measurement retaining its literal specialization and true domain.

    Parameters
    ----------
    exponent : PowerExponent or None, optional
        Literal's independent format. None denotes the second dynamic operand.
        Other fields retain the base ``LutAccuracy`` meanings.
    """

    exponent: PowerExponent | None = None

    def to_public_dict(self) -> dict[str, object]:
        """Return domain, specialization, measured error and a separate global bound."""
        result = LutAccuracy.to_public_dict(self)
        literal = self.exponent
        high, count = (1 << (self.data_width - 1)) - 1, 1 << self.data_width
        if literal is None:
            domain = "base > 0; dynamic exponent is any signed Q word"
            cardinality = high * count
            specialization: dict[str, object] = {"kind": "dynamic", "fraction": self.fraction}
        else:
            if literal.integral:
                domain = "any real base" if literal.value >= 0 else "base != 0"
                cardinality = count if literal.value >= 0 else count - 1
            else:
                domain = "base >= 0" if literal.value > 0 else "base > 0"
                cardinality = high + int(literal.value > 0)
            specialization = {
                "kind": "literal",
                "value": literal.value,
                "code": literal.code,
                "fraction": literal.fraction,
                "integral": literal.integral,
                "odd": literal.odd,
            }
        result.update(
            {
                "reference": "96-digit Decimal real power, clipped to the signed word; errors reported as float",
                "domain": domain,
                "domain_cardinality": cardinality,
                "exponent": specialization,
                "outside_domain": "zero sentinel, outside the mathematical approximation claim",
                "internal_fraction": INTERNAL_FRACTION,
                "literal_zero_power": "saturated one, including at zero base",
            }
        )
        return result


def _axis(low: int, high: int, fraction: int, exhaustive: bool) -> list[int]:
    """Select deterministic signed endpoints, scale neighbors and leading-bit boundaries."""
    if exhaustive:
        return list(range(low, high + 1))
    picked = {low + (high - low) * index // 64 for index in range(65)}
    for bit in range(max(abs(low), abs(high)).bit_length() + 1):
        for sign in (-1, 1):
            picked.update(sign * (1 << bit) + offset for offset in (-1, 0, 1))
    picked.update((1 << fraction) + offset for offset in range(-8, 9))
    picked.update(range(-8, 9))
    return sorted(value for value in picked if low <= value <= high)


def _bound(data_width: int, fraction: int, literal: PowerExponent | None) -> float:
    """Propagate Q56 log/product/exp errors through the clipped real exponential."""
    scale = 1 << fraction
    limit = (1 << (data_width - 1)) / scale
    if literal is None:
        magnitude, exponent_error = limit, 0.0
    else:
        coded = Fraction(literal.code, 1 << literal.fraction)
        clipped = min(abs(Fraction(literal.value)), Fraction(1 << 36))
        magnitude = float(abs(coded))
        exponent_error = float(abs(clipped - abs(coded)))
    logarithm_limit = max(fraction, data_width - 1 - fraction)
    epsilon = (16 * magnitude + 1) / (1 << INTERNAL_FRACTION) + logarithm_limit * exponent_error
    return 0.5 / scale + limit * (math.log(2) * epsilon + 32 / (1 << INTERNAL_FRACTION))


@lru_cache(maxsize=128)
def _measure(data_width: int, fraction: int, literal: PowerExponent | None) -> PowerAccuracy:
    """Measure real inputs at complete small domains or declared deterministic samples."""
    scale = 1 << fraction
    low, high = -(1 << (data_width - 1)), (1 << (data_width - 1)) - 1
    first = 1 if literal is None or not literal.integral and literal.value < 0 else 0
    if literal is not None and literal.integral:
        first = low
    exhaustive = data_width <= (4 if literal is None else 8)
    bases = _axis(first, high, fraction, exhaustive)
    if literal is not None and literal.value < 0:
        bases = [base for base in bases if base != 0]
    exponents = _axis(low, high, fraction, exhaustive) if literal is None else [0]
    worst, count = -1.0, 0
    worst_inputs: tuple[float, ...] = ()
    with localcontext() as context:
        context.prec = 96
        logarithms = {base: (Decimal(abs(base)) / scale).ln() for base in bases if base}
        for base, exponent in product(bases, exponents):
            value = (
                Decimal(exponent) / scale
                if literal is None
                else Decimal(literal.value)
                if isinstance(literal.value, int)
                else Decimal.from_float(literal.value)
            )
            negative = base < 0 and literal is not None and literal.odd
            ceiling = Decimal(-low if negative else high)
            if value == 0:
                reference = min(ceiling, Decimal(scale))
            elif base == 0:
                reference = Decimal(0)
            else:
                logarithm = logarithms[base] * value
                if logarithm >= (ceiling / scale).ln():
                    reference = ceiling
                elif logarithm < -1024:
                    # Even after Q scaling this is below the least positive float.
                    reference = Decimal(0)
                else:
                    reference = logarithm.exp() * scale
            if negative:
                reference = -reference
            actual = power_code(
                base,
                data_width,
                fraction,
                exponent=None if literal is None else literal.value,
                exponent_code=exponent if literal is None else None,
            )
            error = float(abs(Decimal(actual) - reference)) / scale
            if error > worst:
                worst, worst_inputs = (
                    error,
                    (base / scale, exponent / scale) if literal is None else (base / scale,),
                )
            count += 1
    bound = _bound(data_width, fraction, literal)
    if worst > bound:
        raise ValueError("power measurement exceeds its declared arithmetic error bound")
    ranges: tuple[tuple[float, float], ...] = ((first / scale, high / scale),)
    if literal is None:
        ranges += ((low / scale, high / scale),)
    return PowerAccuracy(
        function="power",
        data_width=data_width,
        fraction=fraction,
        lowering="v2",
        exhaustive=exhaustive,
        codes_evaluated=count,
        input_min=first / scale,
        input_max=high / scale,
        table_min=first / scale,
        table_max=high / scale,
        max_error=worst,
        max_error_at=worst_inputs[0],
        max_error_in_table=worst,
        max_error_in_table_at=worst_inputs[0],
        resolution=1 / scale,
        operand_ranges=ranges,
        worst_inputs=worst_inputs,
        declared_error_bound=bound,
        exponent=literal,
    )


def measure_power(
    data_width: int, fraction: int, exponent: int | float | None = None
) -> PowerAccuracy:
    """Measure one complete small power domain or deterministic wider-format sample.

    Parameters
    ----------
    data_width, fraction : int
        The admitted v2 Q format.
    exponent : int or float or None, optional
        Original finite literal, or None for a dynamic second operand.

    Returns
    -------
    PowerAccuracy
        Independent Decimal measurement, original exponent, operand ranges,
        domain cardinality and a separate conservative global arithmetic bound.
        Literal domains through eight bits and dynamic domains through four
        bits are exhaustive. Larger domains are explicitly sampled.

    Raises
    ------
    ValueError
        For invalid arguments or a measured error beyond the declared bound.
    """
    power_code(
        0, data_width, fraction, exponent=exponent, exponent_code=0 if exponent is None else None
    )
    return _measure(data_width, fraction, None if exponent is None else exponent_format(exponent))
