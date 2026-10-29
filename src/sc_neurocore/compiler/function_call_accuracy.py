# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Function accuracy over conservative call-argument domains
"""Measure actual units over a call's interval hull, independently of reachability.

The Cartesian product includes holes and operand combinations not proved reachable.
Small products are exhausted; larger products use explicit deterministic samples.
Invalid real-function inputs are counted and excluded, never called zero error.
Declared arithmetic bounds retain their separate global meaning.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal, localcontext
from functools import lru_cache
from itertools import product

import numpy as np

from . import function_units
from .fixed_expression_ranges import CodeRange
from .lut_accuracy import _MODEL_DOMAIN, _OPEN_LOWER, _reference, lut_read_int, measure
from .power_accuracy import measure_power
from .power_units import exponent_format, power_code

Interval = tuple[int, int]
Axis = tuple[Interval, ...]


@dataclass(frozen=True, slots=True)
class CallAccuracy:
    """Measurement of admitted codes in an inferred argument box.

    Parameters
    ----------
    valid_code_intervals : tuple of interval tuples
        Closed mathematical-domain intervals for each operand, in call order.
    domain_cardinality, excluded_domain_codes : int
        Valid and invalid tuple counts in the supplied Cartesian interval hull.
    codes_evaluated : int
        Number of valid tuples actually measured.
    exhaustive : bool
        Whether every valid tuple in the conservative hull was measured.
    max_error : float or None
        Measured maximum in real units; None means the valid domain is empty.
    worst_inputs : tuple of float
        Complete measured operand tuple at the largest error.
    declared_error_bound : float or None
        Existing global arithmetic bound, separate from the measured maximum.
    reference : str
        Independent real-function reference used for the comparison.
    """

    valid_code_intervals: tuple[Axis, ...]
    domain_cardinality: int
    excluded_domain_codes: int
    codes_evaluated: int
    exhaustive: bool
    max_error: float | None
    worst_inputs: tuple[float, ...]
    declared_error_bound: float | None
    reference: str

    def to_public_dict(self, fraction: int) -> dict[str, object]:
        """Return the measured scope, valid domain and separate arithmetic bound."""
        return {
            "valid_code_intervals": [
                [list(interval) for interval in axis] for axis in self.valid_code_intervals
            ],
            "domain_cardinality": self.domain_cardinality,
            "excluded_domain_codes": self.excluded_domain_codes,
            "codes_evaluated": self.codes_evaluated,
            "exhaustive": self.exhaustive,
            "max_error": self.max_error,
            "max_error_lsb": None if self.max_error is None else self.max_error * (1 << fraction),
            "max_error_at_inputs": list(self.worst_inputs),
            "declared_error_bound": self.declared_error_bound,
            "reference": self.reference,
            "scope": "Cartesian interval hull; holes and operand correlations are not inferred",
        }


def measure_call(
    function: str,
    data_width: int,
    fraction: int,
    operands: tuple[CodeRange, ...],
    *,
    lowering: str = "v2",
    exponent: int | float | None = None,
) -> CallAccuracy:
    """Measure a function over the actual encoded argument intervals of one call.

    Parameters
    ----------
    function : str
        Canonical unary/binary unit name, or ``power``.
    data_width, fraction : int
        Input and output word format.
    operands : tuple of CodeRange
        Inclusive input intervals. State reachability and correlations are not
        assumed. A literal power has only its base interval.
    lowering : str, optional
        ``v2`` or ``lut-v1``; general powers require v2.
    exponent : int or float or None, optional
        Independent literal power exponent, or None for a dynamic second input.

    Returns
    -------
    CallAccuracy
        All valid tuples when their count is at most 65536 (4096 for powers),
        otherwise a deterministic sample with endpoints and critical codes.

    Raises
    ------
    ValueError
        For an invalid function, format, lowering, arity or code interval, or a
        measured violation of an existing declared arithmetic bound.
    """
    if lowering not in {"v2", "lut-v1"}:
        raise ValueError(f"unknown function lowering {lowering!r}")
    if function == "power":
        if lowering != "v2":
            raise ValueError("general power measurements require v2")
        function_units.check_format(data_width, fraction)
        arity = 2 if exponent is None else 1
        if exponent is not None:
            exponent_format(exponent)
    else:
        # The canonical measurement also validates the format and vocabulary.
        measure(function, data_width, fraction, lowering)
        arity = function_units.FUNCTION_ARITY[function]
        if exponent is not None:
            raise ValueError("only a power call has a literal exponent")
    if len(operands) != arity:
        raise ValueError(f"{function} needs {arity} argument interval(s)")
    low, high = -(1 << (data_width - 1)), (1 << (data_width - 1)) - 1
    ranges = tuple((arg.minimum, arg.maximum) for arg in operands)
    if any(not low <= first <= last <= high for first, last in ranges):
        raise ValueError("argument intervals must be ordered signed-word codes")
    return _measure(function, data_width, fraction, ranges, lowering, exponent)


def _valid_axes(
    function: str, ranges: tuple[Interval, ...], fraction: int, exponent: int | float | None
) -> tuple[Axis, ...]:
    axes: list[Axis] = []
    for index, (low, high) in enumerate(ranges):
        if index == 0 and function == "power":
            literal = None if exponent is None else exponent_format(exponent)
            if literal is None or not literal.integral:
                low = max(low, 1 if literal is None or literal.value < 0 else 0)
            if literal is not None and literal.value < 0 and low <= 0 <= high:
                axes.append(tuple((a, b) for a, b in [(low, -1), (1, high)] if a <= b))
                continue
        elif index == 0 and function in _MODEL_DOMAIN:
            first, last = _MODEL_DOMAIN[function]
            low = max(low, math.ceil(first * (1 << fraction)) + int(function in _OPEN_LOWER))
            if math.isfinite(last):
                high = min(high, math.floor(last * (1 << fraction)))
        axes.append(((low, high),) if low <= high else ())
    return tuple(axes)


def _count(axis: Axis) -> int:
    return sum(high - low + 1 for low, high in axis)


def _sample(axis: Axis, fraction: int, exhaustive: bool, count: int) -> list[int]:
    if exhaustive:
        return [code for low, high in axis for code in range(low, high + 1)]
    result: set[int] = set()
    for low, high in axis:
        result.update(low + (high - low) * index // (count - 1) for index in range(count))
        result.update(
            code
            for code in [-1, 0, 1, (1 << fraction) - 1, 1 << fraction, (1 << fraction) + 1]
            if low <= code <= high
        )
        for bit in range(max(abs(low), abs(high)).bit_length() + 1):
            result.update(
                code
                for sign in (-1, 1)
                for offset in (-1, 0, 1)
                if low <= (code := sign * (1 << bit) + offset) <= high
            )
    return sorted(result)


def _power_reference(
    base: int, exponent: int | float | Decimal, fraction: int, low: int, high: int
) -> float:
    scale = 1 << fraction
    value = (
        exponent
        if isinstance(exponent, Decimal)
        else Decimal(exponent)
        if isinstance(exponent, int)
        else Decimal.from_float(exponent)
    )
    if value == 0:
        return min(high, scale) / scale
    if base == 0:
        return 0.0
    negative = base < 0 and int(value) % 2 != 0
    ceiling = Decimal(-low if negative else high)
    logarithm = (Decimal(abs(base)) / scale).ln() * value
    reference = (
        ceiling
        if logarithm >= (ceiling / scale).ln()
        else Decimal(0)
        if logarithm < -1024
        else logarithm.exp() * scale
    )
    return float(-reference if negative else reference) / scale


@lru_cache(maxsize=128)
def _measure(
    function: str,
    width: int,
    fraction: int,
    ranges: tuple[Interval, ...],
    lowering: str,
    exponent: int | float | None,
) -> CallAccuracy:
    axes = _valid_axes(function, ranges, fraction, exponent)
    total = math.prod(high - low + 1 for low, high in ranges)
    cardinality = math.prod(_count(axis) for axis in axes)
    reference_name = (
        "96-digit Decimal real power, clipped to the signed word"
        if function == "power"
        else "Python model function, clipped to the signed word"
    )
    global_accuracy = (
        measure_power(width, fraction, exponent)
        if function == "power"
        else measure(function, width, fraction, lowering)
    )
    bound = global_accuracy.declared_error_bound
    if cardinality == 0:
        return CallAccuracy(axes, 0, total, 0, True, None, (), bound, reference_name)
    exhaustive = cardinality <= (4096 if function == "power" else 65536)
    count = 65 if function == "power" else 257 if len(axes) > 1 else 65536
    samples = [_sample(axis, fraction, exhaustive, count) for axis in axes]
    tuples = list(product(*samples))
    scale = 1 << fraction
    low, high = -(1 << (width - 1)), (1 << (width - 1)) - 1
    if function == "power":
        errors: list[float] = []
        with localcontext() as context:
            context.prec = 96
            for inputs in tuples:
                real_exponent = Decimal(inputs[1]) / scale if exponent is None else exponent
                reference = _power_reference(inputs[0], real_exponent, fraction, low, high)
                actual = power_code(
                    inputs[0],
                    width,
                    fraction,
                    exponent=exponent,
                    exponent_code=inputs[1] if exponent is None else None,
                )
                errors.append(abs(actual / scale - reference))
    else:
        actuals = (
            np.asarray(
                [
                    function_units.evaluate(
                        function, args[0], width, fraction, additional_inputs=args[1:]
                    )
                    if lowering == "v2"
                    else lut_read_int(function, args[0], width, fraction)
                    for args in tuples
                ],
                dtype=np.float64,
            )
            / scale
        )
        operands = [
            np.asarray([args[index] for args in tuples], dtype=np.float64) / scale
            for index in range(len(axes))
        ]
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            if len(axes) == 1:
                references = _reference(function)(operands[0])
            else:
                from sc_neurocore.neurons.equation_namespace import build_eval_namespace

                references = np.asarray(
                    build_eval_namespace()[function](*operands), dtype=np.float64
                )
            errors = list(np.abs(actuals - np.clip(references, low / scale, high / scale)))
    if any(not math.isfinite(value) for value in errors):
        raise ValueError("call reference is not finite on its declared mathematical domain")
    worst = max(range(len(errors)), key=errors.__getitem__)
    error = float(errors[worst])
    if bound is not None and error > bound:
        raise ValueError(f"{function} call measurement exceeds its declared arithmetic error bound")
    return CallAccuracy(
        axes,
        cardinality,
        total - cardinality,
        len(tuples),
        exhaustive,
        error,
        tuple(code / scale for code in tuples[worst]),
        bound,
        reference_name,
    )
