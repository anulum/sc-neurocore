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

Each look-up table carries its measured error at the format against the Python
model's function (:mod:`sc_neurocore.compiler.lut_accuracy`), over the inputs the
format can hold. What is not stated: the range the state, and so a table's
argument, reaches during a run (only encoded values and the initial state are
checked, so a run can still saturate or wrap), how a table's error propagates
through later arithmetic, timing closure, place and route, and board behaviour.
"""

from __future__ import annotations

import ast
import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from fractions import Fraction
from typing import Literal

from ..neurons.equation_builder import EquationNeuron
from . import expr_lut_tables
from .equation_lowering import equation_expressions as _expressions, prepare_neuron
from .lut_accuracy import LutAccuracy, measure
from .intelligence.bit_true_kernel import (
    generate_bittrue_kernel_from_neuron,
    kernel_arithmetic_contract,
)
from .q_format import QFormat
from .power_accuracy import measure_power
from .power_units import power_code

HARDWARE_NUMERIC_CONTRACT_SCHEMA_VERSION = "sc-neurocore.hardware-numeric-contract.v1"

QuantityKind = Literal[
    "parameter",
    "constant",
    "initial_state",
    "time_step",
    "literal",
    "literal_divisor",
    "modulo_period",
    "divisor",
]
EncodingStatus = Literal["exact", "rounded", "underflows_to_zero", "out_of_range"]

NOT_STATED: tuple[str, ...] = (
    "the range the state reaches during a run: only encoded values and the initial "
    "state are checked, so a run can still saturate or wrap",
    "the range a table's or unit's argument reaches during a run: each stated "
    "accuracy covers every input the format holds (or a stated sample of them "
    "for wide formats), not the inputs this model produces",
    "how a table's or unit's error propagates through the model to its state and spikes",
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
    last entry. ``accuracy`` is the table's measured error at the contract's
    format against the Python model's function.
    """

    function: str
    domain_min: float
    domain_max: float
    step: float
    entries: int
    accuracy: LutAccuracy

    def to_public_dict(self) -> dict[str, object]:
        """Return the JSON projection."""
        return {
            "function": self.function,
            "domain_min": self.domain_min,
            "domain_max": self.domain_max,
            "step": self.step,
            "entries": self.entries,
            "outside_domain": "clamped to the first or last entry",
            "accuracy": self.accuracy.to_public_dict(),
        }


@dataclass(frozen=True, slots=True)
class FunctionUnit:
    """One ``v2`` function unit the datapath instantiates, with its measured error.

    Parameters
    ----------
    function : str
        A unary/binary function unit or one ``power`` operator specialization.
    accuracy : LutAccuracy
        Its error at the contract's format against the Python model's function.
        Binary measurements retain both operand ranges, the worst tuple, and
        distinguish a sampled maximum from the declared arithmetic bound.

    """

    function: str
    accuracy: LutAccuracy

    def to_public_dict(self) -> dict[str, object]:
        """Return the JSON projection."""
        return {"function": self.function, "accuracy": self.accuracy.to_public_dict()}


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
        The look-up tables the datapath uses (``lut-v1``).
    function_lowering:
        ``"lut-v1"`` or ``"v2"``.
    function_units:
        The function units the datapath instantiates (``v2``).
    random_precision:
        ``"standard"`` or ``"exact"``: how ``randn()`` is lowered.

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
    function_lowering: str = "lut-v1"
    function_units: tuple[FunctionUnit, ...] = ()
    random_precision: str = "standard"

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
            "function_lowering": self.function_lowering,
            "function_units": [unit.to_public_dict() for unit in self.function_units],
            "random_precision": self.random_precision,
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
    function_lowering: str = "lut-v1",
    random_precision: str = "standard",
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
    function_lowering:
        ``"lut-v1"`` (default; tables reported in ``lookup_tables``) or ``"v2"``
        (units reported in ``function_units``).
    random_precision:
        ``"standard"`` (default) or ``"exact"`` lowering of ``randn()``; stated in
        ``random_precision`` and the arithmetic's ``randomness``.

    Returns
    -------
    HardwareNumericContract
        The encoded quantities, look-up tables or function units with their
        measured error, and bit-true mirror status.

    Raises
    ------
    ValueError
        For an unknown lowering, or a format the ``v2`` units cannot mirror.

    """
    if function_lowering not in ("lut-v1", "v2"):
        raise ValueError(f"function_lowering must be 'lut-v1' or 'v2', got {function_lowering!r}")
    data_width, fraction = q_format.total_bits, q_format.fraction_bits
    # Counter-based draws: xi and a Philox threshold become ordinary rand()/randn().
    authored_neuron = neuron
    neuron = prepare_neuron(neuron)
    quantities = _deduplicated(
        _quantities(
            neuron,
            data_width=data_width,
            fraction=fraction,
            rounding=rounding,
            function_lowering=function_lowering,
        )
    )
    try:
        generate_bittrue_kernel_from_neuron(
            authored_neuron,
            data_width=data_width,
            fraction=fraction,
            overflow=overflow,
            rounding=rounding,
            function_lowering=function_lowering,
            random_precision=random_precision,
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
            function_lowering=function_lowering,
            random_precision=random_precision,
            random_draws=(
                (neuron.random_seed, neuron.random_stream) if neuron.uses_random_draws else None
            ),
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
        lookup_tables=(
            _lookup_tables(neuron, data_width=data_width, fraction=fraction)
            if function_lowering == "lut-v1"
            else ()
        ),
        function_lowering=function_lowering,
        function_units=(
            _function_units(neuron, data_width=data_width, fraction=fraction)
            if function_lowering == "v2"
            else ()
        ),
        random_precision=random_precision,
    )


def _literals(
    node: ast.AST, bound: frozenset[str] = frozenset()
) -> Iterator[tuple[QuantityKind, float]]:
    """Yield each numeric literal the Verilog emitter encodes, as it encodes it.

    A literal divisor becomes its encoded reciprocal, a modulo period is
    encoded as it stands, and floor divisors are not encoded.
    A dynamic power's exponent subtree is encoded normally; a constant exponent
    retains the independently declared format in its power-unit contract. A
    named constant such as ``pi`` is encoded as a literal unless a state
    variable, parameter or constant in ``bound`` takes its name.
    """
    value = _numeric_literal(node)
    if value is not None:
        yield "literal", value
        return
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in expr_lut_tables.RANDOM_FUNCTIONS
    ):
        # The site is a number, not an encoded value; a constant poisson() mean builds
        # thresholds. Only poisson_grid() encodes its mean expression as a word.
        if node.func.id == "poisson_grid" and node.args:
            yield from _literals(node.args[0], bound)
        return
    if isinstance(node, ast.Name):
        if node.id in expr_lut_tables.NAMED_CONSTANTS and node.id not in bound:
            yield "literal", expr_lut_tables.NAMED_CONSTANTS[node.id]
        return
    if isinstance(node, ast.BinOp):
        yield from _literals(node.left, bound)
        if isinstance(node.op, ast.Pow):
            _, exponent = expr_lut_tables.classify_power(node.right, "v2", bound)
            if exponent is None:
                yield from _literals(node.right, bound)
            return
        if isinstance(node.op, ast.FloorDiv):
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
        yield from _literals(node.right, bound)
        return
    for child in ast.iter_child_nodes(node):
        yield from _literals(child, bound)


def _numeric_literal(node: ast.AST) -> float | None:
    """Return the value of a numeric (non-boolean) literal node, else ``None``."""
    if isinstance(node, ast.Constant):
        value = node.value
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return None


def _quantities(
    neuron: EquationNeuron,
    *,
    data_width: int,
    fraction: int,
    rounding: str,
    function_lowering: str,
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
    bound = frozenset(
        {*neuron.parameters, *neuron.constants, *neuron.initial_state, *neuron.equations}
    )
    for where, expression in _expressions(neuron):
        for kind, value in _literals(ast.parse(expression, mode="eval"), bound):
            yield _encoded(kind, where, value, data_width=data_width, fraction=fraction)
    values = {**neuron.parameters, **neuron.constants}
    fixed = _FixedValue(
        values,
        data_width=data_width,
        fraction=fraction,
        rounding=rounding,
        function_lowering=function_lowering,
    )
    for where, expression in _expressions(neuron):
        for node in ast.walk(ast.parse(expression, mode="eval")):
            if not isinstance(node, ast.BinOp) or not isinstance(node.op, ast.Div):
                continue
            denominator = node.right
            # A bare parameter or literal divisor is already checked above.
            if isinstance(denominator, (ast.Name, ast.Constant)):
                continue
            evaluated = fixed.evaluate(denominator)
            if evaluated is None or evaluated[0] != 0:
                continue
            try:
                real = float(evaluated[1])
            except OverflowError:
                # Only an out-of-range operand folds past the float range, and that
                # operand is already reported (and blocking) on its own.
                continue
            yield EncodedQuantity(
                kind="divisor",
                name=f"{where}: {ast.unparse(denominator)}",
                value=real,
                rtl_value=0.0,
                status="underflows_to_zero",
                blocking=True,
            )


class _FixedValue:
    """Evaluate a parameter-only expression as the RTL datapath computes it.

    Sums are exact; products and quotients are narrowed to the word as the
    bit-true kernel narrows them, under the selected rounding (truncate,
    nearest, or bankers' half-to-even exactly as the RTL's guard bit does it).
    A rational real value accompanies each word; nonintegral general powers
    use the model's finite floating real result. An expression reading state,
    inputs or unsupported calls is not folded. In v2, parameter exponents
    use their encoded words while literal powers retain independent precision.
    """

    def __init__(
        self,
        values: dict[str, float],
        *,
        data_width: int,
        fraction: int,
        rounding: str,
        function_lowering: str = "lut-v1",
    ) -> None:
        self._values = values
        self._dw = data_width
        self._frac = fraction
        self._rounding = rounding if fraction > 0 else "truncate"
        self._function_lowering = function_lowering

    def _wrap(self, value: int, bits: int) -> int:
        value &= (1 << bits) - 1
        return value - (1 << bits) if value >= 1 << (bits - 1) else value

    def _encode(self, value: float | Fraction) -> int:
        return self._wrap(round(Fraction(value) * (1 << self._frac)), self._dw)

    def _mul(self, left: int, right: int) -> int:
        product = self._wrap(left * right, 2 * self._dw)
        if self._frac == 0:
            return self._wrap(product, self._dw)
        half = 1 << (self._frac - 1)
        if self._rounding == "nearest":
            product = self._wrap(product + (half - 1 if product < 0 else half), 2 * self._dw)
        elif self._rounding == "bankers":
            # The RTL adds half, shifts, and clears the LSB on an exact tie.
            tie = product & ((1 << self._frac) - 1) == half
            word = self._wrap(self._wrap(product + half, 2 * self._dw) >> self._frac, self._dw)
            return word & ~1 if tie else word
        return self._wrap(product >> self._frac, self._dw)

    def evaluate(self, node: ast.AST) -> tuple[int, Fraction] | None:
        """Return RTL word and rational/model-float value, or None outside the foldable subset."""
        literal = _numeric_literal(node)
        if literal is not None:
            return self._encode(literal), Fraction(literal)
        if isinstance(node, ast.Name):
            value = self._values.get(node.id)
            return None if value is None else (self._encode(float(value)), Fraction(float(value)))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            operand = self.evaluate(node.operand)
            if operand is None or isinstance(node.op, ast.UAdd):
                return operand
            return -operand[0], -operand[1]
        if not isinstance(node, ast.BinOp):
            return None
        left = self.evaluate(node.left)
        if left is None:
            return None
        if isinstance(node.op, ast.Pow):
            if self._function_lowering == "v2":
                kind, folded_exponent = expr_lut_tables.classify_power(
                    node.right, "v2", self._values
                )
                if kind == "power":
                    return self._power(left, node.right, folded_exponent)
                if kind == "multiply":
                    assert folded_exponent is not None
                    word = left[0]
                    for _ in range(int(folded_exponent) - 1):
                        word = self._mul(word, left[0])
                    return word, left[1] ** int(folded_exponent)
            exponent = node.right
            if not (
                isinstance(exponent, ast.Constant)
                and type(exponent.value) is int
                and 2 <= exponent.value <= 8
            ):
                return None
            word = left[0]
            for _ in range(exponent.value - 1):
                word = self._mul(word, left[0])
            return word, left[1] ** exponent.value
        right = self.evaluate(node.right)
        if right is None:
            return None
        if isinstance(node.op, ast.Add):
            return left[0] + right[0], left[1] + right[1]
        if isinstance(node.op, ast.Sub):
            return left[0] - right[0], left[1] - right[1]
        if isinstance(node.op, ast.Mult):
            return self._mul(left[0], right[0]), left[1] * right[1]
        if not isinstance(node.op, ast.Div) or right[1] == 0:
            return None
        if _numeric_literal(node.right) is not None:
            # The compiler multiplies by the float reciprocal of the literal.
            word = self._mul(left[0], self._encode(1.0 / float(right[1])))
        elif right[0] == 0:
            return None
        else:
            shifted = self._wrap(left[0] * (1 << self._frac), 2 * self._dw)
            quotient = abs(shifted) // abs(right[0])
            word = self._wrap(quotient if (shifted < 0) == (right[0] < 0) else -quotient, self._dw)
        return word, left[1] / right[1]

    def _power(
        self, base: tuple[int, Fraction], node: ast.AST, literal: float | None
    ) -> tuple[int, Fraction] | None:
        """Detect a constant power divisor lost to final word rounding."""
        if literal is None:
            exponent = self.evaluate(node)
            if exponent is None:
                return None
            real_exponent = exponent[1]
            exponent_code = self._wrap(exponent[0], self._dw)
        else:
            real_exponent, exponent_code = Fraction(literal), None
        code = power_code(
            self._wrap(base[0], self._dw),
            self._dw,
            self._frac,
            exponent=literal,
            exponent_code=exponent_code,
        )
        try:
            if real_exponent.denominator == 1 and abs(real_exponent) <= 64:
                real = base[1] ** int(real_exponent)
            else:
                real = Fraction(math.pow(float(base[1]), float(real_exponent)))
        except (ValueError, OverflowError, ZeroDivisionError):
            return None
        return code, real


def _encoded(
    kind: QuantityKind, name: str, value: float, *, data_width: int, fraction: int
) -> EncodedQuantity:
    """Encode ``value`` as the compiler does and classify the result."""
    if not math.isfinite(value):
        raise ValueError(f"{kind.replace('_', ' ')} {name} is not finite: {value!r}")
    scale = 1 << fraction
    # Exact: value * scale is a float product that overflows to inf for a large
    # finite value (1e308 at fraction 8); the rational product rounds the same way.
    raw = round(Fraction(value) * scale)
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


def _lookup_tables(
    neuron: EquationNeuron, *, data_width: int, fraction: int
) -> tuple[LookupTable, ...]:
    """Return the look-up tables the Verilog emitter builds for ``neuron``, measured at the format."""
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
    return tuple(
        _table(function, data_width=data_width, fraction=fraction) for function in sorted(functions)
    )


def _function_units(
    neuron: EquationNeuron, *, data_width: int, fraction: int
) -> tuple[FunctionUnit, ...]:
    """Return the ``v2`` units the Verilog emitter instantiates for ``neuron``, measured at the format."""
    functions: set[str] = {"exprel"} if neuron.method == "exp_euler" else set()
    powers: set[float | None] = set()
    bound = frozenset(
        {*neuron.parameters, *neuron.constants, *neuron.initial_state, *neuron.equations}
    )
    for _where, expression in _expressions(neuron):
        for node in ast.walk(ast.parse(expression, mode="eval")):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                unit = expr_lut_tables.V2_UNIT_FOR.get(node.func.id)
                if unit is not None:
                    functions.add(unit)
            elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
                kind, exponent = expr_lut_tables.classify_power(node.right, "v2", bound)
                if kind in {"sqrt", "cbrt"}:
                    functions.add(kind)
                elif kind == "power":
                    powers.add(exponent)
    units = tuple(
        FunctionUnit(function, measure(function, data_width, fraction, "v2"))
        for function in sorted(functions)
    )
    specialized = tuple(
        FunctionUnit("power", measure_power(data_width, fraction, exponent))
        for exponent in sorted(powers, key=repr)
    )
    return tuple(sorted((*units, *specialized), key=lambda unit: unit.function))


def _table(function: str, *, data_width: int, fraction: int) -> LookupTable:
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
        accuracy=measure(function, data_width, fraction),
    )


__all__ = [
    "HARDWARE_NUMERIC_CONTRACT_SCHEMA_VERSION",
    "MIRROR_EVIDENCE",
    "NOT_STATED",
    "EncodedQuantity",
    "FunctionUnit",
    "HardwareNumericContract",
    "LookupTable",
    "hardware_numeric_contract",
]
