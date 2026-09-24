# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — What one neuron becomes at one fixed-point format

"""State what a neuron's generated RTL holds at one fixed-point format.

The equation compiler encodes every parameter, constant, initial state, the
time step and every numeric literal as ``round(value * 2**fraction)`` wrapped
to the word width. A value outside the format's range therefore changes sign
or magnitude without an error, and a non-zero value below its resolution
becomes zero. :func:`hardware_numeric_contract` reads the neuron the compiler
would lower and reports, for one format, the value each of those quantities
actually takes in the RTL, the look-up tables the datapath uses, and whether
the generated bit-true C kernel mirrors the RTL.

A format is *representable* for a neuron when no quantity leaves the format's
range and no parameter, constant, initial state, time step, literal divisor or
modulo period rounds to zero. A plain literal below the resolution, such as a
floating-point round-off guard, is reported but does not make the format
unrepresentable: the fixed-point comparison it guards is exact.

What is not stated: the range the state reaches during a run (only encoded
values and the initial state are checked, so a run can still saturate or
wrap), the approximation error of the look-up tables, timing closure, place
and route, and board behaviour.
"""

from __future__ import annotations

import ast
import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Literal

from ..neurons.equation_builder import EquationNeuron
from . import expr_lut_tables
from .intelligence.bit_true_kernel import (
    generate_bittrue_kernel_from_neuron,
    kernel_arithmetic_contract,
)
from .q_format import QFormat

HARDWARE_NUMERIC_CONTRACT_SCHEMA_VERSION = "sc-neurocore.hardware-numeric-contract.v1"

QuantityKind = Literal[
    "parameter",
    "constant",
    "initial_state",
    "time_step",
    "literal",
    "literal_divisor",
    "modulo_period",
]
EncodingStatus = Literal["exact", "rounded", "underflows_to_zero", "out_of_range"]

NOT_STATED: tuple[str, ...] = (
    "the range the state reaches during a run: only encoded values and the initial "
    "state are checked, so a run can still saturate or wrap",
    "the approximation error of the look-up tables",
    "timing closure, place and route, and board behaviour",
)
"""What a hardware numeric contract does not establish."""

MIRROR_EVIDENCE = (
    "the generated C kernel is compared with the RTL by finite Icarus Verilog "
    "co-simulation on the tested stimuli; this is a check, not a proof"
)
"""What the bit-true mirror of the RTL rests on."""

_LUT_FUNCTIONS: dict[str, str] = {
    "exp": "exp",
    "log": "log",
    "sqrt": "sqrt",
    "tanh": "tanh",
    "cosh": "cosh",
    "exprel": "exprel",
    "sigmoid": "sigmoid",
    "expit": "sigmoid",
    "sin": "sin",
    "cos": "cos",
}


@dataclass(frozen=True, slots=True)
class EncodedQuantity:
    """One value the RTL encodes, and the value it holds there.

    Parameters
    ----------
    kind:
        Where the value comes from. ``literal_divisor`` is the reciprocal the
        compiler multiplies by when an expression divides by a literal.
    name:
        The parameter, constant or state name, or where a literal appears.
    value:
        The value the neuron declares.
    rtl_value:
        The value the encoded word represents in the RTL.
    status:
        ``exact``, ``rounded``, ``underflows_to_zero`` or ``out_of_range``.
    blocking:
        ``True`` when the status makes the format unrepresentable.
    """

    kind: QuantityKind
    name: str
    value: float
    rtl_value: float
    status: EncodingStatus
    blocking: bool

    @property
    def relative_error(self) -> float | None:
        """Return ``|rtl_value - value| / |value|``, or ``None`` for a zero value."""
        if self.value == 0.0:
            return None
        return abs(self.rtl_value - self.value) / abs(self.value)

    def describe(self) -> str:
        """Return a one-line statement of what the RTL holds."""
        return (
            f"{self.kind.replace('_', ' ')} {self.name}={self.value!r} becomes {self.rtl_value!r}"
        )

    def to_public_dict(self) -> dict[str, object]:
        """Return the JSON projection."""
        return {
            "kind": self.kind,
            "name": self.name,
            "value": self.value,
            "rtl_value": self.rtl_value,
            "status": self.status,
            "relative_error": self.relative_error,
            "blocking": self.blocking,
        }


@dataclass(frozen=True, slots=True)
class LookupTable:
    """One transcendental look-up table the datapath uses.

    Arguments outside ``[domain_min, domain_max)`` are clamped to the first or
    last entry.
    """

    function: str
    domain_min: float
    domain_max: float
    step: float
    entries: int

    def to_public_dict(self) -> dict[str, object]:
        """Return the JSON projection."""
        return {
            "function": self.function,
            "domain_min": self.domain_min,
            "domain_max": self.domain_max,
            "step": self.step,
            "entries": self.entries,
            "outside_domain": "clamped to the first or last entry",
        }


@dataclass(frozen=True, slots=True)
class HardwareNumericContract:
    """What one neuron's generated RTL holds at one fixed-point format.

    Parameters
    ----------
    q_format:
        Studio label, ``Q<integer bits>.<fraction bits>``.
    data_width, fraction:
        Word width and fraction bits.
    method:
        The integration method the RTL realises.
    overflow, rounding:
        Accumulate commit policy and multiply product policy.
    arithmetic:
        The bit-true arithmetic statement when the generated C kernel mirrors
        this neuron, else ``None``.
    mirror_refusal:
        Why no bit-true C kernel mirrors this neuron, or ``""``.
    quantities:
        Every encoded value, in declaration order, literals once each.
    lookup_tables:
        The look-up tables the datapath uses.
    """

    q_format: str
    data_width: int
    fraction: int
    method: str
    overflow: str
    rounding: str
    arithmetic: Mapping[str, object] | None
    mirror_refusal: str
    quantities: tuple[EncodedQuantity, ...]
    lookup_tables: tuple[LookupTable, ...]

    @property
    def resolution(self) -> float:
        """Return the value of one least-significant bit."""
        return 1.0 / (1 << self.fraction)

    @property
    def min_value(self) -> float:
        """Return the most negative representable value."""
        return -(1 << (self.data_width - 1)) / (1 << self.fraction)

    @property
    def max_value(self) -> float:
        """Return the most positive representable value."""
        return ((1 << (self.data_width - 1)) - 1) / (1 << self.fraction)

    @property
    def blocking(self) -> tuple[EncodedQuantity, ...]:
        """Return the quantities that make the format unrepresentable."""
        return tuple(quantity for quantity in self.quantities if quantity.blocking)

    @property
    def representable(self) -> bool:
        """Return whether every blocking check passes."""
        return not self.blocking

    def refusal(self) -> str:
        """Return why the format is unrepresentable, or ``""``."""
        blocking = self.blocking
        if not blocking:
            return ""
        return (
            f"{self.q_format} cannot hold this neuron (range [{self.min_value!r}, "
            f"{self.max_value!r}], resolution {self.resolution!r}): "
            + "; ".join(quantity.describe() for quantity in blocking)
        )

    def to_public_dict(self) -> dict[str, object]:
        """Return the path-free JSON projection."""
        return {
            "schema_version": HARDWARE_NUMERIC_CONTRACT_SCHEMA_VERSION,
            "q_format": self.q_format,
            "data_width": self.data_width,
            "fraction": self.fraction,
            "resolution": self.resolution,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "method": self.method,
            "overflow": self.overflow,
            "rounding": self.rounding,
            "representable": self.representable,
            "refusal": self.refusal(),
            "quantities": [quantity.to_public_dict() for quantity in self.quantities],
            "lookup_tables": [table.to_public_dict() for table in self.lookup_tables],
            "timing": {
                "cycles_per_step": 1,
                "pipeline_latency_cycles": 0,
                "basis": (
                    "compiled without pipeline stages: every state register and the "
                    "spike output take their next value on each rising clock edge"
                ),
            },
            "bit_true_mirror": {
                "available": self.arithmetic is not None,
                "refusal": self.mirror_refusal,
                "evidence": MIRROR_EVIDENCE if self.arithmetic is not None else "",
                "arithmetic": None if self.arithmetic is None else dict(self.arithmetic),
            },
            "not_stated": list(NOT_STATED),
        }


def hardware_numeric_contract(
    neuron: EquationNeuron,
    q_format: QFormat,
    *,
    overflow: str = "saturate",
    rounding: str = "truncate",
) -> HardwareNumericContract:
    """Return what ``neuron``'s generated RTL holds at ``q_format``.

    Parameters
    ----------
    neuron:
        The neuron the equation compiler lowers.
    q_format:
        The signed fixed-point format.
    overflow, rounding:
        The accumulate and multiply policies the RTL is compiled with.

    Returns
    -------
    HardwareNumericContract
        The encoded quantities, look-up tables and bit-true mirror status.
    """
    data_width, fraction = q_format.total_bits, q_format.fraction_bits
    quantities = _deduplicated(_quantities(neuron, data_width=data_width, fraction=fraction))
    try:
        generate_bittrue_kernel_from_neuron(
            neuron,
            data_width=data_width,
            fraction=fraction,
            overflow=overflow,
            rounding=rounding,
        )
    except (ValueError, NotImplementedError) as exc:
        arithmetic: dict[str, object] | None = None
        mirror_refusal = str(exc)
    else:
        arithmetic = kernel_arithmetic_contract(
            data_width=data_width,
            fraction=fraction,
            overflow=overflow,
            rounding=rounding,
            method=neuron.method,
        )
        mirror_refusal = ""
    return HardwareNumericContract(
        q_format=q_format.q_label,
        data_width=data_width,
        fraction=fraction,
        method=neuron.method,
        overflow=overflow,
        rounding=rounding,
        arithmetic=arithmetic,
        mirror_refusal=mirror_refusal,
        quantities=quantities,
        lookup_tables=_lookup_tables(neuron),
    )


def _expressions(neuron: EquationNeuron) -> Iterator[tuple[str, str]]:
    """Yield every expression the compiler lowers, with where it appears."""
    for variable, expression in neuron.equations.items():
        yield f"equation {variable}", expression
    if neuron.method == "exp_euler":
        # The exponential-Euler datapath also lowers each diagonal Jacobian term.
        for variable, expression in neuron.jacobian_expressions.items():
            yield f"jacobian {variable}", expression
    if neuron.threshold_expr:
        yield "threshold", neuron.threshold_expr
    for variable, expression in neuron.reset_rules.items():
        yield f"reset {variable}", expression
    if neuron.rate_expression:
        yield "escape rate", neuron.rate_expression
    if neuron.probability_expression:
        yield "spike probability", neuron.probability_expression


def _literals(node: ast.AST) -> Iterator[tuple[QuantityKind, float]]:
    """Yield each numeric literal the Verilog emitter encodes, as it encodes it.

    A literal divisor becomes its encoded reciprocal, a modulo period is
    encoded as it stands, and exponents and floor divisors are not encoded.
    """
    value = _numeric_literal(node)
    if value is not None:
        yield "literal", value
        return
    if isinstance(node, ast.BinOp):
        yield from _literals(node.left)
        if isinstance(node.op, (ast.Pow, ast.FloorDiv)):
            return
        right = _numeric_literal(node.right)
        if isinstance(node.op, ast.Div) and right is not None:
            if right == 0.0:
                raise ValueError("an expression divides by the literal 0")
            yield "literal_divisor", 1.0 / right
            return
        if isinstance(node.op, ast.Mod) and right is not None:
            yield "modulo_period", right
            return
        yield from _literals(node.right)
        return
    for child in ast.iter_child_nodes(node):
        yield from _literals(child)


def _numeric_literal(node: ast.AST) -> float | None:
    """Return the value of a numeric (non-boolean) literal node, else ``None``."""
    if isinstance(node, ast.Constant):
        value = node.value
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return None


def _quantities(
    neuron: EquationNeuron, *, data_width: int, fraction: int
) -> Iterator[EncodedQuantity]:
    """Yield every encoded value of ``neuron`` in declaration order."""
    for name, value in neuron.parameters.items():
        yield _encoded("parameter", name, float(value), data_width=data_width, fraction=fraction)
    for name, value in neuron.constants.items():
        yield _encoded("constant", name, float(value), data_width=data_width, fraction=fraction)
    for name, value in neuron.initial_state.items():
        yield _encoded(
            "initial_state", name, float(value), data_width=data_width, fraction=fraction
        )
    if neuron.dt != 0.0:
        # RK4 also scales its stages by dt/2 and its weighted sum by dt/6.
        steps = {"dt": 1.0, "dt/2": 2.0, "dt/6": 6.0} if neuron.method == "rk4" else {"dt": 1.0}
        for name, divisor in steps.items():
            yield _encoded(
                "time_step",
                name,
                float(neuron.dt) / divisor,
                data_width=data_width,
                fraction=fraction,
            )
    for where, expression in _expressions(neuron):
        for kind, value in _literals(ast.parse(expression, mode="eval")):
            yield _encoded(kind, where, value, data_width=data_width, fraction=fraction)


def _encoded(
    kind: QuantityKind, name: str, value: float, *, data_width: int, fraction: int
) -> EncodedQuantity:
    """Encode ``value`` as the compiler does and classify the result."""
    if not math.isfinite(value):
        raise ValueError(f"{kind.replace('_', ' ')} {name} is not finite: {value!r}")
    scale = 1 << fraction
    raw = int(round(value * scale))
    word = raw & ((1 << data_width) - 1)
    if word >= 1 << (data_width - 1):
        word -= 1 << data_width
    rtl_value = word / scale
    status: EncodingStatus
    if word != raw:
        status = "out_of_range"
    elif raw == 0 and value != 0.0:
        status = "underflows_to_zero"
    elif rtl_value == value:
        status = "exact"
    else:
        status = "rounded"
    blocking = status == "out_of_range" or (status == "underflows_to_zero" and kind != "literal")
    return EncodedQuantity(
        kind=kind,
        name=name,
        value=value,
        rtl_value=rtl_value,
        status=status,
        blocking=blocking,
    )


def _deduplicated(quantities: Iterator[EncodedQuantity]) -> tuple[EncodedQuantity, ...]:
    """Keep each literal once, at its first appearance; keep every named value."""
    seen: set[tuple[str, float]] = set()
    kept: list[EncodedQuantity] = []
    for quantity in quantities:
        if quantity.kind in {"literal", "literal_divisor", "modulo_period"}:
            key = (quantity.kind, quantity.value)
            if key in seen:
                continue
            seen.add(key)
        kept.append(quantity)
    return tuple(kept)


def _lookup_tables(neuron: EquationNeuron) -> tuple[LookupTable, ...]:
    """Return the look-up tables the Verilog emitter builds for ``neuron``."""
    # The exponential-Euler increment is f * dt * exprel(A * dt).
    functions: set[str] = {"exprel"} if neuron.method == "exp_euler" else set()
    for _where, expression in _expressions(neuron):
        for node in ast.walk(ast.parse(expression, mode="eval")):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                function = _LUT_FUNCTIONS.get(node.func.id)
                if function is not None:
                    functions.add(function)
            elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
                exponent = expr_lut_tables.const_float(node.right)
                if exponent is not None and abs(exponent - 1.0 / 3.0) < 1e-6:
                    functions.add("cbrt")
                elif exponent is not None and abs(exponent - 0.5) < 1e-6:
                    functions.add("sqrt")
    return tuple(_table(function) for function in sorted(functions))


def _table(function: str) -> LookupTable:
    """Return the sample grid of one look-up table."""
    if function == "log":
        points = expr_lut_tables.log_sample_points()
    elif function == "sqrt":
        points = expr_lut_tables.sqrt_sample_points()
    else:
        points = expr_lut_tables.symmetric_sample_points()
    step = points[1] - points[0]
    return LookupTable(
        function=function,
        domain_min=points[0],
        domain_max=points[-1] + step,
        step=step,
        entries=len(points),
    )


__all__ = [
    "HARDWARE_NUMERIC_CONTRACT_SCHEMA_VERSION",
    "MIRROR_EVIDENCE",
    "NOT_STATED",
    "EncodedQuantity",
    "HardwareNumericContract",
    "LookupTable",
    "hardware_numeric_contract",
]
