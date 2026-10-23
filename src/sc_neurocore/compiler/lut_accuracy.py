# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — measured approximation error of the transcendental look-up tables

"""Measure what each transcendental table returns against the Python model's function.

The Verilog emitter and the bit-true C/Rust kernel read the same tables
(:mod:`sc_neurocore.compiler.expr_lut_tables`), so their co-simulation shows that
the RTL equals the kernel, not that either equals the function. This module states
the second relation for the current lowering, ``lut-v1``: for one function and one
fixed-point format it reads the table exactly as the RTL indexes it
(:func:`lut_read`) and compares every result with the function the Python model
evaluates (:mod:`sc_neurocore.neurons.equation_namespace`).

The reference is the model's value clipped to the word's range, because no RTL
of that width can hold more; where the function leaves the word, the table's
saturated entry is therefore not counted as error. Formats up to
:data:`EXHAUSTIVE_MAX_WIDTH` bits are evaluated at every input code. Wider formats
are sampled: every table cell's first and last code, evenly spaced codes inside
each cell (every code inside the table when the format's codes are coarser than
its cells, as at fraction 0), and evenly spaced codes outside the table's
domain; the result says which.

``lowering="v2"`` measures the function units of
:mod:`sc_neurocore.compiler.function_units` the same way, reading each result
from that integer reference (which the generated RTL and kernel equal; see
``tests/test_function_lowering_v2_cosim.py``): every code of formats up to
:data:`V2_EXHAUSTIVE_MAX_WIDTH` bits, otherwise 65 536 codes spread over the word
plus every code within 4096 of zero. A unit has no table domain; it is stated as
the whole input range.

What is not stated: the error after the value enters further arithmetic, and the
range a model's argument reaches during a run.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import numpy.typing as npt

from sc_neurocore.neurons.equation_namespace import build_eval_namespace

from . import expr_lut_tables
from . import function_units as fu

LOWERING = "lut-v1"
LOWERINGS = ("lut-v1", "v2")
EXHAUSTIVE_MAX_WIDTH = 18
V2_EXHAUSTIVE_MAX_WIDTH = 16
_INTERIOR_SAMPLES_PER_CELL = 64
_OUTSIDE_SAMPLES = 4096

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class TableGeometry:
    """Where a table samples its function.

    Parameters
    ----------
    lo : float
        First sample point.
    step : float
        Sample spacing, a power of two.
    size : int
        Number of entries.

    """

    lo: float
    step: float
    size: int


_SYMMETRIC = TableGeometry(-16.0, 0.125, 256)
GEOMETRY: dict[str, TableGeometry] = {
    "exp": _SYMMETRIC,
    "tanh": _SYMMETRIC,
    "cosh": _SYMMETRIC,
    "exprel": _SYMMETRIC,
    "sigmoid": _SYMMETRIC,
    "sin": _SYMMETRIC,
    "cos": _SYMMETRIC,
    "cbrt": _SYMMETRIC,
    "log": TableGeometry(
        expr_lut_tables.LOG_LUT_MIN, expr_lut_tables.LOG_LUT_STEP, expr_lut_tables.LOG_LUT_SIZE
    ),
    "sqrt": TableGeometry(
        expr_lut_tables.SQRT_LUT_MIN, expr_lut_tables.SQRT_LUT_STEP, expr_lut_tables.SQRT_LUT_SIZE
    ),
}

_ENTRIES: dict[str, Callable[[int, int], list[int]]] = {
    "exp": expr_lut_tables.exp_lut_entries,
    "tanh": lambda _dw, fr: expr_lut_tables.tanh_lut_entries(fr),
    "cosh": expr_lut_tables.cosh_lut_entries,
    "exprel": expr_lut_tables.exprel_lut_entries,
    "sigmoid": lambda _dw, fr: expr_lut_tables.sigmoid_lut_entries(fr),
    "sin": lambda _dw, fr: expr_lut_tables.sin_lut_entries(fr),
    "cos": lambda _dw, fr: expr_lut_tables.cos_lut_entries(fr),
    "cbrt": lambda _dw, fr: expr_lut_tables.cbrt_lut_entries(fr),
    "log": lambda _dw, fr: expr_lut_tables.log_lut_entries(fr),
    "sqrt": lambda _dw, fr: expr_lut_tables.sqrt_lut_entries(fr),
}

# Inputs the Python model evaluates to a real number: the logarithms need x > 0
# (log1p: x > -1); ``sqrt`` refuses x < 0; ``x ** (1/3)`` of a negative float is
# NaN in NumPy, as are arcsin and arccos outside [-1, 1].
_MODEL_DOMAIN: dict[str, tuple[float, float]] = {
    "log": (0.0, math.inf),
    "log2": (0.0, math.inf),
    "log10": (0.0, math.inf),
    "log1p": (-1.0, math.inf),
    "sqrt": (0.0, math.inf),
    "cbrt": (0.0, math.inf),
    "arcsin": (-1.0, 1.0),
    "arccos": (-1.0, 1.0),
}
_OPEN_LOWER = frozenset({"log", "log2", "log10", "log1p"})


def _reference(function: str) -> Callable[[FloatArray], FloatArray]:
    """Return the Python model's function for ``function``, vectorised."""
    namespace = build_eval_namespace()
    if function == "cbrt":
        return lambda x: np.power(x, 1.0 / 3.0)
    if function == "sqrt":
        return lambda x: np.sqrt(x)
    model = namespace[function]
    return lambda x: np.asarray(model(x), dtype=np.float64)


def table_functions(lowering: str = LOWERING) -> tuple[str, ...]:
    """Return the functions a lowering computes through a table or unit, sorted.

    Parameters
    ----------
    lowering : str
        ``"lut-v1"`` (default) or ``"v2"``.

    Returns
    -------
    tuple[str, ...]
        ``cbrt`` and ``sqrt`` stand for ``x ** (1/3)`` and ``sqrt``/``x ** 0.5``;
        ``sigmoid`` also serves ``expit``. ``v2`` adds ``sinh``.

    Raises
    ------
    ValueError
        For an unknown lowering.

    """
    if lowering not in LOWERINGS:
        raise ValueError(f"unknown lowering {lowering!r}")
    return tuple(sorted(GEOMETRY if lowering == LOWERING else fu.FUNCTIONS))


def lut_read(function: str, codes: IntArray, data_width: int, fraction: int) -> IntArray:
    """Return the table entries the RTL selects for signed input codes.

    Mirrors ``_VerilogExprEmitter._emit_lut_call``: index
    ``(code - round(lo * 2**fraction)) >> (fraction - log2(1/step))``, a left shift
    when that is negative, clamped to ``[0, size - 1]``, computed in a width that holds
    every shifted sum (``expr_lut_tables.lut_offset_width``); entries saturated to the
    word (``expr_lut_tables.saturate_entries``). Only formats for which
    :func:`fits_int64` holds; the others use :func:`lut_read_int`.

    Parameters
    ----------
    function : str
        A member of :func:`table_functions`.
    codes : numpy.ndarray of int64
        Signed input codes of ``data_width`` bits.
    data_width, fraction : int
        The fixed-point format.

    Returns
    -------
    numpy.ndarray of int64
        The selected entries, as signed codes.

    Raises
    ------
    ValueError
        When the format's shifted index does not fit a signed 64-bit integer.

    """
    if not fits_int64(function, data_width, fraction):
        raise ValueError(
            f"{function} at data_width={data_width}, fraction={fraction} needs more than 63 "
            "bits for the shifted index; use lut_read_int"
        )
    geometry = GEOMETRY[function]
    stored = expr_lut_tables.saturate_entries(_ENTRIES[function](data_width, fraction), data_width)
    entries = np.asarray(stored, dtype=np.int64)
    min_q = round(geometry.lo * (1 << fraction))
    shift = fraction - round(-math.log2(geometry.step))
    offset = codes.astype(np.int64) - min_q
    index = offset >> shift if shift >= 0 else offset << -shift
    selected: IntArray = entries[np.clip(index, 0, geometry.size - 1)]
    return selected


def fits_int64(function: str, data_width: int, fraction: int) -> bool:
    """Return whether :func:`lut_read` computes this table's index exactly in int64.

    The index is ``(code - min_q)`` shifted by ``fraction - log2(1/step)``; the sum
    needs one bit more than the wider of the code and ``min_q``, and a left shift adds
    its amount. Every intermediate must stay within a signed 64-bit integer.

    Parameters
    ----------
    function : str
        A member of :func:`table_functions`.
    data_width, fraction : int
        The fixed-point format.

    Returns
    -------
    bool
        ``True`` when every intermediate fits in 63 magnitude bits.

    """
    geometry = GEOMETRY[function]
    min_q = round(geometry.lo * (1 << fraction))
    shift = fraction - round(-math.log2(geometry.step))
    offset_bits = max(data_width, abs(min_q).bit_length() + 1) + 1
    return offset_bits + max(0, -shift) <= 63


def lut_read_int(function: str, code: int, data_width: int, fraction: int) -> int:
    """Return the entry the RTL selects for one input code, in exact integer arithmetic.

    Parameters
    ----------
    function : str
        A member of :func:`table_functions`.
    code : int
        Signed input code of ``data_width`` bits (any width up to 64).
    data_width, fraction : int
        The fixed-point format.

    Returns
    -------
    int
        The selected entry, saturated to the word, as :func:`lut_read` computes it.

    """
    geometry = GEOMETRY[function]
    entries = expr_lut_tables.saturate_entries(_ENTRIES[function](data_width, fraction), data_width)
    shift = fraction - round(-math.log2(geometry.step))
    offset = code - round(geometry.lo * (1 << fraction))
    index = offset >> shift if shift >= 0 else offset << -shift
    return entries[max(0, min(geometry.size - 1, index))]


@dataclass(frozen=True, slots=True)
class LutAccuracy:
    """Measured error of one table at one format, against the Python model.

    Parameters
    ----------
    function : str
        The tabulated function.
    data_width, fraction : int
        The format.
    lowering : str
        The lowering measured (``"lut-v1"`` or ``"v2"``).
    exhaustive : bool
        Whether every input code was evaluated (otherwise sampled, see module).
    codes_evaluated : int
        Number of input codes evaluated.
    input_min, input_max : float
        The evaluated inputs: the format's range within the model's domain.
    table_min, table_max : float
        The table's domain ``[lo, lo + size*step)``; outside it the endpoint is read.
    max_error : float
        Largest ``|table - clip(f)|`` over the evaluated inputs, in real units.
    max_error_at : float
        An input where ``max_error`` occurs (first coordinate for binary calls;
        ``worst_inputs`` holds the complete tuple).
    max_error_in_table : float
        The same, over inputs inside the table's domain only.
    max_error_in_table_at : float
        An input where ``max_error_in_table`` occurs.
    resolution : float
        One least significant bit, ``2**-fraction``.
    operand_ranges : tuple of pairs, optional
        Inclusive ranges of all operands when an explicit function domain is reported.
    worst_inputs : tuple of float, optional
        Complete operand tuple at the measured maximum, in call order.
    declared_error_bound : float or None, optional
        Arithmetic error bound in real units, separate from the sampled maximum.

    """

    function: str
    data_width: int
    fraction: int
    lowering: str
    exhaustive: bool
    codes_evaluated: int
    input_min: float
    input_max: float
    table_min: float
    table_max: float
    max_error: float
    max_error_at: float
    max_error_in_table: float
    max_error_in_table_at: float
    resolution: float
    operand_ranges: tuple[tuple[float, float], ...] = ()
    worst_inputs: tuple[float, ...] = ()
    declared_error_bound: float | None = None

    @property
    def q_format(self) -> str:
        """Return the format label ``Q<integer bits>.<fraction bits>``."""
        return f"Q{self.data_width - self.fraction}.{self.fraction}"

    def to_public_dict(self) -> dict[str, object]:
        """Return the JSON projection, errors also in least significant bits."""
        result: dict[str, object] = {
            "function": self.function,
            "q_format": self.q_format,
            "lowering": self.lowering,
            "exhaustive": self.exhaustive,
            "codes_evaluated": self.codes_evaluated,
            "inputs": [self.input_min, self.input_max],
            "table_domain": [self.table_min, self.table_max],
            "max_error": self.max_error,
            "max_error_lsb": self.max_error / self.resolution,
            "max_error_at": self.max_error_at,
            "max_error_in_table": self.max_error_in_table,
            "max_error_in_table_lsb": self.max_error_in_table / self.resolution,
            "max_error_in_table_at": self.max_error_in_table_at,
            "reference": "Python model function, clipped to the word's range",
        }
        if self.operand_ranges:
            for key in (
                "table_domain",
                "max_error_in_table",
                "max_error_in_table_lsb",
                "max_error_in_table_at",
            ):
                result.pop(key)
            result.update(
                {
                    "input_arity": len(self.operand_ranges),
                    "operand_ranges": [list(pair) for pair in self.operand_ranges],
                    "max_error_at_inputs": list(self.worst_inputs),
                    "declared_error_bound": self.declared_error_bound,
                    "domain": "signed word codes in call order; fixed-point zero is positive zero",
                    "domain_cardinality": 1 << (self.data_width * len(self.operand_ranges)),
                }
            )
        return result


def _codes(function: str, data_width: int, fraction: int) -> tuple[list[int], bool]:
    """Return the input codes to evaluate and whether they are every code (exact integers)."""
    low_code = -(1 << (data_width - 1))
    high_code = (1 << (data_width - 1)) - 1
    domain = _MODEL_DOMAIN.get(function)
    if domain is not None:
        first = math.ceil(domain[0] * (1 << fraction))
        low_code = max(low_code, first + 1 if function in _OPEN_LOWER else first)
    if data_width <= EXHAUSTIVE_MAX_WIDTH:
        return list(range(low_code, high_code + 1)), True
    geometry = GEOMETRY[function]
    # The cells as lut_read indexes them: index = (code - min_q) >> shift.
    min_q = round(geometry.lo * (1 << fraction))
    shift = fraction - round(-math.log2(geometry.step))
    picked = {low_code, high_code}
    if shift >= 0:
        starts = [min_q + (k << shift) for k in range(geometry.size + 1)]
    else:
        # Coarser codes than cells (fraction 0 against a 1/8 step): every code
        # inside the table lands on its own cell, so all of them are taken.
        in_table = -(-geometry.size >> -shift)  # ceil(size / 2**-shift) codes
        starts = [min_q, min_q + in_table]
        picked.update(range(min_q, min_q + in_table))
    for first, nxt in zip(starts[:-1], starts[1:]):
        picked.update(_spread(first, nxt - 1, _INTERIOR_SAMPLES_PER_CELL))
        picked.update((first - 1, first))
    picked.update(_spread(low_code, starts[0], _OUTSIDE_SAMPLES))
    picked.update(_spread(starts[-1], high_code, _OUTSIDE_SAMPLES))
    return sorted(c for c in picked if low_code <= c <= high_code), False


def _spread(first: int, last: int, count: int) -> list[int]:
    """Return up to ``count`` integers evenly spread over ``[first, last]``, exactly."""
    if last <= first:
        return [first]
    return [first + (last - first) * k // (count - 1) for k in range(count)]


def _v2_codes(function: str, data_width: int, fraction: int) -> tuple[IntArray, bool]:
    """Return the input codes a ``v2`` measurement evaluates and whether they are every code."""
    low_code = -(1 << (data_width - 1))
    high_code = (1 << (data_width - 1)) - 1
    domain = _MODEL_DOMAIN.get(function)
    if domain is not None:
        first = math.ceil(domain[0] * (1 << fraction))
        low_code = max(low_code, first + 1 if function in _OPEN_LOWER else first)
        if math.isfinite(domain[1]):
            high_code = min(high_code, math.floor(domain[1] * (1 << fraction)))
    if data_width <= V2_EXHAUSTIVE_MAX_WIDTH:
        return np.arange(low_code, high_code + 1, dtype=np.int64), True
    spread = np.linspace(low_code, high_code, 1 << 16).astype(np.int64)
    near_zero = np.arange(-4096, 4097, dtype=np.int64)
    codes = np.unique(np.concatenate([spread, near_zero, np.array([low_code, high_code])]))
    return codes[(codes >= low_code) & (codes <= high_code)], False


def _measure_v2(function: str, data_width: int, fraction: int) -> LutAccuracy:
    """Measure one ``v2`` unit at one format against the Python model's function."""
    if function not in fu.FUNCTIONS:
        raise ValueError(f"no v2 unit for {function!r}")
    fu.check_format(data_width, fraction)
    if function == "atan2":
        return _measure_atan2(data_width, fraction)
    if function == "heaviside":
        return _measure_heaviside(data_width, fraction)
    codes, exhaustive = _v2_codes(function, data_width, fraction)
    scale = float(1 << fraction)
    x = codes.astype(np.float64) / scale
    values = np.array(
        [fu.evaluate(function, int(c), data_width, fraction) for c in codes], dtype=np.float64
    )
    word_min = -(1 << (data_width - 1)) / scale
    word_max = ((1 << (data_width - 1)) - 1) / scale
    with np.errstate(over="ignore", invalid="ignore"):
        reference = np.clip(_reference(function)(x), word_min, word_max)
    error = np.abs(values / scale - reference)
    worst = int(np.argmax(error))
    bound = 0.5 / scale + 16.0 / (1 << fu.P) if function == "softplus" else None
    if function == "round":
        bound = 0.0
    if bound is not None and float(error[worst]) > bound:
        raise ValueError(f"{function} measurement exceeds its declared arithmetic error bound")
    return LutAccuracy(
        function=function,
        data_width=data_width,
        fraction=fraction,
        lowering="v2",
        exhaustive=exhaustive,
        codes_evaluated=len(codes),
        input_min=float(x[0]),
        input_max=float(x[-1]),
        table_min=float(x[0]),
        table_max=float(x[-1]) + 1.0 / scale,
        max_error=float(error[worst]),
        max_error_at=float(x[worst]),
        max_error_in_table=float(error[worst]),
        max_error_in_table_at=float(x[worst]),
        resolution=1.0 / scale,
        operand_ranges=((word_min, word_max),) if bound is not None else (),
        worst_inputs=(float(x[worst]),) if bound is not None else (),
        declared_error_bound=bound,
    )


def _measure_atan2(data_width: int, fraction: int) -> LutAccuracy:
    low, high = -(1 << (data_width - 1)), (1 << (data_width - 1)) - 1
    exhaustive = data_width <= 8
    if exhaustive:
        axis = np.arange(low, high + 1, dtype=np.int64)
    else:
        edges = {0, low, high, *range(-8, 9)}
        for bit in range(data_width):
            edges.update(sign * (1 << bit) + delta for sign in (-1, 1) for delta in (-1, 0, 1))
        spread = np.linspace(low, high, 257).astype(np.int64)
        axis = np.unique(np.concatenate([spread, np.array(sorted(edges), dtype=np.int64)]))
        axis = axis[(axis >= low) & (axis <= high)]
    y = np.repeat(axis, len(axis))
    x = np.tile(axis, len(axis))
    scale = float(1 << fraction)
    actual = (
        np.array(
            [
                fu.evaluate("atan2", int(a), data_width, fraction, additional_inputs=(int(b),))
                for a, b in zip(y, x)
            ],
            dtype=np.float64,
        )
        / scale
    )
    reference = np.clip(np.arctan2(y / scale, x / scale), low / scale, high / scale)
    error = np.abs(actual - reference)
    worst = int(np.argmax(error))
    bound = 0.5 / scale + 8.0 / (1 << fu.P)
    if float(error[worst]) > bound:
        raise ValueError("atan2 measurement exceeds its declared arithmetic error bound")
    return LutAccuracy(
        function="atan2",
        data_width=data_width,
        fraction=fraction,
        lowering="v2",
        exhaustive=exhaustive,
        codes_evaluated=len(y),
        input_min=low / scale,
        input_max=high / scale,
        table_min=low / scale,
        table_max=(high + 1) / scale,
        max_error=float(error[worst]),
        max_error_at=float(y[worst] / scale),
        max_error_in_table=float(error[worst]),
        max_error_in_table_at=float(y[worst] / scale),
        resolution=1.0 / scale,
        operand_ranges=((low / scale, high / scale),) * 2,
        worst_inputs=(float(y[worst] / scale), float(x[worst] / scale)),
        declared_error_bound=bound,
    )


def _measure_heaviside(data_width: int, fraction: int) -> LutAccuracy:
    """Measure both operands while declaring the exact selection/code-identity bound."""
    low, high = -(1 << (data_width - 1)), (1 << (data_width - 1)) - 1
    exhaustive = data_width <= 8
    if exhaustive:
        first_axis = second_axis = np.arange(low, high + 1, dtype=np.int64)
    else:
        first_axis = np.unique(np.array([low, -1, 0, 1, high], dtype=np.int64))
        second_axis = np.unique(
            np.concatenate([np.linspace(low, high, 257).astype(np.int64), np.array([-1, 0, 1])])
        )
    first = np.repeat(first_axis, len(second_axis))
    second = np.tile(second_axis, len(first_axis))
    scale = float(1 << fraction)
    actual = (
        np.array(
            [
                fu.evaluate(
                    "heaviside", int(x), data_width, fraction, additional_inputs=(int(at_zero),)
                )
                for x, at_zero in zip(first, second, strict=True)
            ],
            dtype=np.float64,
        )
        / scale
    )
    reference = np.clip(np.heaviside(first / scale, second / scale), low / scale, high / scale)
    error = np.abs(actual - reference)
    worst = int(np.argmax(error))
    if float(error[worst]) != 0.0:
        raise ValueError("heaviside measurement exceeds its exact arithmetic error bound")
    return LutAccuracy(
        function="heaviside",
        data_width=data_width,
        fraction=fraction,
        lowering="v2",
        exhaustive=exhaustive,
        codes_evaluated=len(first),
        input_min=low / scale,
        input_max=high / scale,
        table_min=low / scale,
        table_max=(high + 1) / scale,
        max_error=float(error[worst]),
        max_error_at=float(first[worst] / scale),
        max_error_in_table=float(error[worst]),
        max_error_in_table_at=float(first[worst] / scale),
        resolution=1.0 / scale,
        operand_ranges=((low / scale, high / scale),) * 2,
        worst_inputs=(float(first[worst] / scale), float(second[worst] / scale)),
        declared_error_bound=0.0,
    )


@lru_cache(maxsize=512)
def measure(function: str, data_width: int, fraction: int, lowering: str = LOWERING) -> LutAccuracy:
    """Measure one table or unit at one format against the Python model's function.

    Parameters
    ----------
    function : str
        A member of :func:`table_functions` for the lowering.
    data_width, fraction : int
        The fixed-point format; ``0 <= fraction < data_width <= 64`` (``v2``: see
        :func:`sc_neurocore.compiler.function_units.check_format`).
    lowering : str
        ``"lut-v1"`` (default) or ``"v2"``.

    Returns
    -------
    LutAccuracy
        The measured errors and what was evaluated.

    Raises
    ------
    ValueError
        For an unknown function, lowering or an invalid format.

    """
    if lowering not in LOWERINGS:
        raise ValueError(f"unknown lowering {lowering!r}")
    if lowering == "v2":
        return _measure_v2(function, data_width, fraction)
    if function not in GEOMETRY:
        raise ValueError(f"no lut-v1 table for {function!r}")
    if not 0 <= fraction < data_width <= 64:
        raise ValueError(f"invalid format data_width={data_width}, fraction={fraction}")
    codes, exhaustive = _codes(function, data_width, fraction)
    scale = float(1 << fraction)
    x = np.array([code / scale for code in codes], dtype=np.float64)
    if fits_int64(function, data_width, fraction):
        selected = lut_read(function, np.array(codes, dtype=np.int64), data_width, fraction)
        table = selected.astype(np.float64) / scale
    else:
        table = np.array([lut_read_int(function, c, data_width, fraction) / scale for c in codes])
    word_min = -(1 << (data_width - 1)) / scale
    word_max = ((1 << (data_width - 1)) - 1) / scale
    with np.errstate(over="ignore", invalid="ignore"):
        reference = np.clip(_reference(function)(x), word_min, word_max)
    error = np.abs(table - reference)
    geometry = GEOMETRY[function]
    table_max = geometry.lo + geometry.size * geometry.step
    inside = (x >= geometry.lo) & (x < table_max)
    worst = int(np.argmax(error))
    if inside.any():
        inside_index = np.flatnonzero(inside)
        worst_inside = int(inside_index[np.argmax(error[inside])])
        max_inside, at_inside = float(error[worst_inside]), float(x[worst_inside])
    else:
        max_inside, at_inside = 0.0, math.nan
    return LutAccuracy(
        function=function,
        data_width=data_width,
        fraction=fraction,
        lowering=LOWERING,
        exhaustive=exhaustive,
        codes_evaluated=len(codes),
        input_min=float(x[0]),
        input_max=float(x[-1]),
        table_min=geometry.lo,
        table_max=table_max,
        max_error=float(error[worst]),
        max_error_at=float(x[worst]),
        max_error_in_table=max_inside,
        max_error_in_table_at=at_inside,
        resolution=1.0 / scale,
    )


__all__ = [
    "EXHAUSTIVE_MAX_WIDTH",
    "GEOMETRY",
    "LOWERING",
    "LOWERINGS",
    "LutAccuracy",
    "V2_EXHAUSTIVE_MAX_WIDTH",
    "TableGeometry",
    "fits_int64",
    "lut_read",
    "lut_read_int",
    "measure",
    "table_functions",
]
