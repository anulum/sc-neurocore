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
each cell, and evenly spaced codes outside the table's domain; the result says
which.

What is not stated: the error after the value enters further arithmetic, the
range a model's argument reaches during a run, and anything about a lowering
other than ``lut-v1``.
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

LOWERING = "lut-v1"
EXHAUSTIVE_MAX_WIDTH = 18
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

# Inputs the Python model evaluates to a real number: ``log`` needs x > 0; ``sqrt``
# refuses x < 0; ``x ** (1/3)`` of a negative float is NaN in NumPy.
_MODEL_DOMAIN: dict[str, tuple[float, float]] = {
    "log": (0.0, math.inf),
    "sqrt": (0.0, math.inf),
    "cbrt": (0.0, math.inf),
}
_OPEN_LOWER = frozenset({"log"})


def _reference(function: str) -> Callable[[FloatArray], FloatArray]:
    """Return the Python model's function for ``function``, vectorised."""
    namespace = build_eval_namespace()
    if function == "cbrt":
        return lambda x: np.power(x, 1.0 / 3.0)
    if function == "sqrt":
        return lambda x: np.sqrt(x)
    model = namespace[function]
    return lambda x: np.asarray(model(x), dtype=np.float64)


def table_functions() -> tuple[str, ...]:
    """Return the functions ``lut-v1`` lowers through a table, sorted.

    Returns
    -------
    tuple[str, ...]
        ``cbrt`` and ``sqrt`` stand for ``x ** (1/3)`` and ``sqrt``/``x ** 0.5``;
        ``sigmoid`` also serves ``expit``.
    """
    return tuple(sorted(GEOMETRY))


def lut_read(function: str, codes: IntArray, data_width: int, fraction: int) -> IntArray:
    """Return the table entries the RTL selects for signed input codes.

    Mirrors ``_VerilogExprEmitter._emit_lut_call``: index
    ``(code - round(lo * 2**fraction)) >> (fraction - log2(1/step))``, a left shift
    when that is negative, clamped to ``[0, size - 1]``.

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
    """
    geometry = GEOMETRY[function]
    entries = np.asarray(_ENTRIES[function](data_width, fraction), dtype=np.int64)
    min_q = round(geometry.lo * (1 << fraction))
    shift = fraction - round(-math.log2(geometry.step))
    offset = codes.astype(np.int64) - min_q
    index = offset >> shift if shift >= 0 else offset << -shift
    selected: IntArray = entries[np.clip(index, 0, geometry.size - 1)]
    return selected


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
        The lowering measured (``"lut-v1"``).
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
        An input where ``max_error`` occurs.
    max_error_in_table : float
        The same, over inputs inside the table's domain only.
    max_error_in_table_at : float
        An input where ``max_error_in_table`` occurs.
    resolution : float
        One least significant bit, ``2**-fraction``.
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

    @property
    def q_format(self) -> str:
        """Return the format label ``Q<integer bits>.<fraction bits>``."""
        return f"Q{self.data_width - self.fraction}.{self.fraction}"

    def to_public_dict(self) -> dict[str, object]:
        """Return the JSON projection, errors also in least significant bits."""
        return {
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


def _codes(function: str, data_width: int, fraction: int) -> tuple[IntArray, bool]:
    """Return the input codes to evaluate and whether they are every code."""
    low_code = -(1 << (data_width - 1))
    high_code = (1 << (data_width - 1)) - 1
    domain = _MODEL_DOMAIN.get(function)
    if domain is not None:
        first = math.ceil(domain[0] * (1 << fraction))
        low_code = max(low_code, first + 1 if function in _OPEN_LOWER else first)
    if data_width <= EXHAUSTIVE_MAX_WIDTH:
        return np.arange(low_code, high_code + 1, dtype=np.int64), True
    geometry = GEOMETRY[function]
    scale = 1 << fraction
    starts = np.round((geometry.lo + np.arange(geometry.size + 1) * geometry.step) * scale)
    starts = starts.astype(np.int64)
    pieces: list[IntArray] = [starts, starts - 1]
    for first, nxt in zip(starts[:-1], starts[1:]):
        pieces.append(np.linspace(first, nxt - 1, _INTERIOR_SAMPLES_PER_CELL).astype(np.int64))
    pieces.append(np.linspace(low_code, starts[0], _OUTSIDE_SAMPLES).astype(np.int64))
    pieces.append(np.linspace(starts[-1], high_code, _OUTSIDE_SAMPLES).astype(np.int64))
    pieces.append(np.array([low_code, high_code], dtype=np.int64))
    codes = np.unique(np.concatenate(pieces))
    return codes[(codes >= low_code) & (codes <= high_code)], False


@lru_cache(maxsize=256)
def measure(function: str, data_width: int, fraction: int) -> LutAccuracy:
    """Measure one table at one format against the Python model's function.

    Parameters
    ----------
    function : str
        A member of :func:`table_functions`.
    data_width, fraction : int
        The fixed-point format; ``0 <= fraction < data_width <= 64``.

    Returns
    -------
    LutAccuracy
        The measured errors and what was evaluated.

    Raises
    ------
    ValueError
        For an unknown function or an invalid format.
    """
    if function not in GEOMETRY:
        raise ValueError(f"no lut-v1 table for {function!r}")
    if not 0 <= fraction < data_width <= 64:
        raise ValueError(f"invalid format data_width={data_width}, fraction={fraction}")
    codes, exhaustive = _codes(function, data_width, fraction)
    scale = float(1 << fraction)
    x = codes.astype(np.float64) / scale
    table = lut_read(function, codes, data_width, fraction).astype(np.float64) / scale
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
        codes_evaluated=int(codes.size),
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
    "LutAccuracy",
    "TableGeometry",
    "lut_read",
    "measure",
    "table_functions",
]
