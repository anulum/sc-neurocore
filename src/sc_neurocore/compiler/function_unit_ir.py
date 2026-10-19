# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — function lowering v2: one 64-bit integer program per unit, emitted to Verilog, C and Rust

"""A straight-line 64-bit integer program, written once and emitted for every target.

Each ``v2`` function unit (:mod:`sc_neurocore.compiler.function_unit_programs`) is
built as a list of :class:`Step` values over signed 64-bit integers. The same
list is

* evaluated in Python (:meth:`Program.evaluate`), so the program can be checked
  against the independent reference :mod:`sc_neurocore.compiler.function_units`;
* emitted as Verilog ``wire signed [63:0]`` declarations and ``case`` tables;
* emitted as C ``int64_t`` or Rust ``i64`` locals, with the tables as arrays.

Every value is kept within ``(-2**63, 2**63)`` by construction of the units, so
the Verilog 64-bit context, C ``int64_t`` and Rust ``i64`` all hold it exactly.
Left shifts are emitted as multiplications in C and Rust (a left shift of a
negative value is undefined in C); right shifts are arithmetic and floor;
division truncates towards zero; variable shift amounts are clamped to
``[0, 63]`` before use, and both sides of a select are always computed, so no
shift amount is ever negative.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

_OPS = frozenset(
    {
        "const", "add", "sub", "mul", "div", "neg", "shl", "shr", "shl_var", "shr_var",
        "mask", "lt", "le", "eq", "select", "table", "input", "xor", "extern",
    }
)  # fmt: skip


@dataclass(frozen=True, slots=True)
class Step:
    """One assignment ``name = op(args)`` of a program.

    Parameters
    ----------
    name : str
        Local name, unique within the program.
    op : str
        Operation, one of ``const, add, sub, mul, div, neg, shl, shr, shl_var,
        shr_var, mask, xor, lt, le, eq, select, table, input, extern``.
    args : tuple[str, ...]
        Operand names, in operation order.
    value : int
        Constant value (``const``), constant shift (``shl``, ``shr``), mask bits
        (``mask``) or coefficient position 0–2 (``table``).
    table : str
        Table name for ``table``; input name for ``extern``.

    """

    name: str
    op: str
    args: tuple[str, ...] = ()
    value: int = 0
    table: str = ""


@dataclass(slots=True)
class Program:
    """A straight-line program with one input and one output.

    Parameters
    ----------
    tables : dict[str, tuple[tuple[int, int, int], ...]]
        Coefficient tables referenced by ``table`` steps.

    """

    tables: dict[str, tuple[tuple[int, int, int], ...]] = field(default_factory=dict)
    steps: list[Step] = field(default_factory=list)
    output: str = ""
    _count: int = 0

    def _add(self, op: str, args: Sequence[str] = (), value: int = 0, table: str = "") -> str:
        if op not in _OPS:
            raise ValueError(f"unknown operation {op!r}")
        name = f"t{self._count}"
        self._count += 1
        self.steps.append(Step(name, op, tuple(args), value, table))
        return name

    # -- construction -------------------------------------------------------------------------
    def input(self) -> str:
        """Return the program's input (the signed input code)."""
        return self._add("input")

    def const(self, value: int) -> str:
        """Return a constant; ``|value| < 2**63``."""
        if not -(1 << 63) < value < 1 << 63:
            raise ValueError(f"constant {value} does not fit 64 bits")
        return self._add("const", value=value)

    def add(self, a: str, b: str) -> str:
        """Return the name of the 64-bit sum ``a + b``."""
        return self._add("add", (a, b))

    def sub(self, a: str, b: str) -> str:
        """Return the name of the 64-bit difference ``a - b``."""
        return self._add("sub", (a, b))

    def mul(self, a: str, b: str) -> str:
        """Return the name of the 64-bit product ``a · b``."""
        return self._add("mul", (a, b))

    def div(self, a: str, b: str) -> str:
        """Return ``a / b`` truncated towards zero; ``b`` must not be zero where selected."""
        return self._add("div", (a, b))

    def neg(self, a: str) -> str:
        """Return the name of the 64-bit negation ``-a``."""
        return self._add("neg", (a,))

    def shl(self, a: str, amount: int) -> str:
        """Return ``a · 2**amount`` for a constant ``amount >= 0``."""
        return self._add("shl", (a,), amount)

    def shr(self, a: str, amount: int) -> str:
        """Return ``floor(a / 2**amount)`` for a constant ``amount >= 0``."""
        return self._add("shr", (a,), amount)

    def shl_var(self, a: str, amount: str) -> str:
        """Return ``a · 2**clamp(amount, 0, 63)``."""
        return self._add("shl_var", (a, amount))

    def shr_var(self, a: str, amount: str) -> str:
        """Return ``floor(a / 2**clamp(amount, 0, 63))``."""
        return self._add("shr_var", (a, amount))

    def xor(self, a: str, b: str) -> str:
        """Return the bitwise exclusive or of ``a`` and ``b``."""
        return self._add("xor", (a, b))

    def extern(self, name: str) -> str:
        """Return a named external input (a step counter, a seed), bound at emission or evaluation."""
        return self._add("extern", table=name)

    def mask(self, a: str, bits: int) -> str:
        """Return the low ``bits`` bits of ``a`` (non-negative)."""
        return self._add("mask", (a,), bits)

    def lt(self, a: str, b: str) -> str:
        """Return 1 if ``a < b`` else 0."""
        return self._add("lt", (a, b))

    def le(self, a: str, b: str) -> str:
        """Return 1 if ``a <= b`` else 0."""
        return self._add("le", (a, b))

    def eq(self, a: str, b: str) -> str:
        """Return 1 if ``a == b`` else 0."""
        return self._add("eq", (a, b))

    def select(self, condition: str, when_true: str, when_false: str) -> str:
        """Return ``when_true`` if ``condition`` is non-zero, else ``when_false``."""
        return self._add("select", (condition, when_true, when_false))

    def lookup(self, table: str, index: str, coefficient: int) -> str:
        """Return coefficient ``0``, ``1`` or ``2`` of ``tables[table][index]``."""
        if table not in self.tables:
            raise ValueError(f"unknown table {table!r}")
        return self._add("table", (index,), coefficient, table)

    # -- evaluation ---------------------------------------------------------------------------
    def evaluate(self, code: int, externs: dict[str, int] | None = None) -> int:
        """Evaluate the program on one input code, checking every value fits 64 bits.

        Parameters
        ----------
        code : int
            Signed input code.
        externs : dict[str, int], optional
            Values of the program's named external inputs.

        Returns
        -------
        int
            The output value.

        Raises
        ------
        OverflowError
            If any intermediate leaves the signed 64-bit range.

        """
        values: dict[str, int] = {}
        bound = externs or {}
        for step in self.steps:
            if step.op == "extern":
                values[step.name] = _check(step, bound[step.table])
                continue
            values[step.name] = _check(step, _apply(step, values, code, self.tables))
        return values[self.output]


def _clamp_shift(amount: int) -> int:
    return max(0, min(63, amount))


def _apply(
    step: Step,
    values: dict[str, int],
    code: int,
    tables: dict[str, tuple[tuple[int, int, int], ...]],
) -> int:
    a = [values[name] for name in step.args]
    op = step.op
    if op == "input":
        return code
    if op == "const":
        return step.value
    if op in ("add", "sub", "mul"):
        return a[0] + a[1] if op == "add" else a[0] - a[1] if op == "sub" else a[0] * a[1]
    if op == "div":
        if a[1] == 0:
            return 0  # an unselected branch: both sides of a select are computed
        quotient = abs(a[0]) // abs(a[1])
        return -quotient if (a[0] < 0) != (a[1] < 0) else quotient
    if op == "neg":
        return -a[0]
    if op == "shl":
        return a[0] << step.value
    if op == "shr":
        return a[0] >> step.value
    if op == "shl_var":
        return a[0] << _clamp_shift(a[1])
    if op == "shr_var":
        return a[0] >> _clamp_shift(a[1])
    if op == "mask":
        return a[0] & ((1 << step.value) - 1)
    if op == "xor":
        return a[0] ^ a[1]
    if op in ("lt", "le", "eq"):
        return int(a[0] < a[1] if op == "lt" else a[0] <= a[1] if op == "le" else a[0] == a[1])
    if op == "select":
        return a[1] if a[0] else a[2]
    rows = tables[step.table]
    return rows[max(0, min(len(rows) - 1, a[0]))][step.value]


def _check(step: Step, value: int) -> int:
    if not -(1 << 63) < value < 1 << 63:
        raise OverflowError(f"step {step.name} ({step.op}) leaves 64 bits: {value}")
    return value


# -- emission ---------------------------------------------------------------------------------
# C/Rust coefficient arrays carry this prefix; kernels declare them as 64-bit.
TABLE_PREFIX = "_fu_"
_BINARY = {"add": "+", "sub": "-", "mul": "*", "xor": "^"}
_COMPARE = {"lt": "<", "le": "<=", "eq": "=="}


def _v_const(value: int) -> str:
    return f"-64'sd{-value}" if value < 0 else f"64'sd{value}"


def _v_clamp(amount: str) -> str:
    return f"(({amount} < 64'sd0) ? 64'sd0 : (({amount} > 64'sd63) ? 64'sd63 : {amount}))"


def _verilog_expression(step: Step, a: list[str], argument: str, externs: dict[str, str]) -> str:
    op = step.op
    if op == "input":
        return f"$signed({argument})"
    if op == "extern":
        return f"$signed({externs[step.table]})"
    if op == "const":
        return _v_const(step.value)
    if op in _BINARY:
        return f"{a[0]} {_BINARY[op]} {a[1]}"
    if op == "div":
        return f"({a[1]} == 64'sd0) ? 64'sd0 : ({a[0]} / {a[1]})"
    if op == "neg":
        return f"-{a[0]}"
    if op == "shl":
        return f"{a[0]} <<< {step.value}"
    if op == "shr":
        return f"{a[0]} >>> {step.value}"
    if op == "shl_var":
        return f"{a[0]} <<< {_v_clamp(a[1])}"
    if op == "shr_var":
        return f"{a[0]} >>> {_v_clamp(a[1])}"
    if op == "mask":
        return f"{a[0]} & {_v_const((1 << step.value) - 1)}"
    if op in _COMPARE:
        return f"({a[0]} {_COMPARE[op]} {a[1]}) ? 64'sd1 : 64'sd0"
    return f"({a[0]} != 64'sd0) ? {a[1]} : {a[2]}"  # select


def emit_verilog(
    program: Program, prefix: str, argument: str, externs: dict[str, str] | None = None
) -> tuple[list[str], str]:
    """Emit the program as Verilog module items.

    Parameters
    ----------
    program : Program
        The unit.
    prefix : str
        Unique wire prefix for this call site.
    argument : str
        Verilog expression of the signed input code.
    externs : dict[str, str], optional
        Verilog expression of each named external input (64 bits).

    Returns
    -------
    tuple[list[str], str]
        Module items (wire declarations and ``always @(*)`` table cases) and the
        name of the 64-bit output wire.

    """
    lines: list[str] = []
    name = {step.name: f"{prefix}_{step.name}" for step in program.steps}
    for step in program.steps:
        target = name[step.name]
        a = [name[arg] for arg in step.args]
        if step.op == "table":
            rows = program.tables[step.table]
            lines.append(f"reg signed [63:0] {target};")
            lines.append(f"always @(*) case ({a[0]})")
            for index, row in enumerate(rows):
                lines.append(f"    {index}: {target} = {_v_const(row[step.value])};")
            lines.append(f"    default: {target} = {_v_const(rows[-1][step.value])};")
            lines.append("endcase")
        else:
            expression = _verilog_expression(step, a, argument, externs or {})
            lines.append(f"wire signed [63:0] {target} = {expression};")
    return lines, name[program.output]


def _c_expression(
    step: Step, a: list[str], argument: str, rust: bool, externs: dict[str, str]
) -> str:
    op = step.op
    one = "1i64" if rust else "(int64_t)1"

    def clamp(x: str) -> str:
        if rust:
            return f"(if {x} < 0 {{ 0 }} else if {x} > 63 {{ 63 }} else {{ {x} }})"
        return f"(({x}) < 0 ? 0 : (({x}) > 63 ? 63 : ({x})))"

    if op == "input":
        return f"({argument}) as i64" if rust else f"(int64_t)({argument})"
    if op == "extern":
        return f"({externs[step.table]}) as i64" if rust else f"(int64_t)({externs[step.table]})"
    if op == "const":
        if rust:
            return f"{step.value}i64"
        return f"(int64_t){step.value}LL" if step.value >= 0 else f"(-(int64_t){-step.value}LL)"
    if op in _BINARY:
        return f"{a[0]} {_BINARY[op]} {a[1]}"
    if op == "div":
        if rust:
            return f"(if {a[1]} == 0 {{ 0 }} else {{ {a[0]} / {a[1]} }})"
        return f"(({a[1]}) == 0 ? 0 : ({a[0]} / {a[1]}))"
    if op == "neg":
        return f"-{a[0]}"
    # A left shift of a negative value is undefined in C: multiply by the power of two.
    if op == "shl":
        return f"{a[0]} * ({one} << {step.value})"
    if op == "shr":
        return f"{a[0]} >> {step.value}"
    if op == "shl_var":
        return f"{a[0]} * ({one} << {clamp(a[1])})"
    if op == "shr_var":
        return f"{a[0]} >> {clamp(a[1])}"
    if op == "mask":
        return f"{a[0]} & {(1 << step.value) - 1}{'i64' if rust else 'LL'}"
    if op in _COMPARE:
        return (
            f"({a[0]} {_COMPARE[op]} {a[1]}) as i64"
            if rust
            else f"(int64_t)({a[0]} {_COMPARE[op]} {a[1]})"
        )
    if rust:
        return f"(if {a[0]} != 0 {{ {a[1]} }} else {{ {a[2]} }})"
    return f"(({a[0]}) != 0 ? ({a[1]}) : ({a[2]}))"  # select


def emit_c(
    program: Program,
    prefix: str,
    argument: str,
    lang: str = "c",
    externs: dict[str, str] | None = None,
) -> tuple[list[str], dict[str, list[int]], str]:
    """Emit the program as C or Rust statements.

    Parameters
    ----------
    program : Program
        The unit.
    prefix : str
        Unique local-name prefix for this call site.
    argument : str
        C/Rust expression of the signed input code.
    lang : str
        ``"c"`` or ``"rust"``.
    externs : dict[str, str], optional
        C/Rust expression of each named external input (64-bit).

    Returns
    -------
    tuple[list[str], dict[str, list[int]], str]
        Statements, the tables to declare (flattened ``a, b, c`` triples keyed by
        array name ``TABLE_PREFIX + table``, the same for every call site) and the
        output local's name.

    Raises
    ------
    ValueError
        For a language other than ``"c"`` or ``"rust"``.

    """
    if lang not in ("c", "rust"):
        raise ValueError(f"unknown language {lang!r}")
    rust = lang == "rust"
    statements: list[str] = []
    arrays: dict[str, list[int]] = {}
    name = {step.name: f"{prefix}_{step.name}" for step in program.steps}
    for step in program.steps:
        a = [name[arg] for arg in step.args]
        if step.op == "table":
            array = f"{TABLE_PREFIX}{step.table}"  # shared by every call site
            arrays.setdefault(array, [v for row in program.tables[step.table] for v in row])
            last = len(program.tables[step.table]) - 1
            if rust:
                index = f"(if {a[0]} < 0 {{ 0 }} else if {a[0]} > {last} {{ {last} }} else {{ {a[0]} }})"
                expression = f"{array}[({index}) as usize * 3 + {step.value}]"
            else:
                index = f"(({a[0]}) < 0 ? 0 : (({a[0]}) > {last} ? {last} : ({a[0]})))"
                expression = f"{array}[({index}) * 3 + {step.value}]"
        else:
            expression = _c_expression(step, a, argument, rust, externs or {})
        local = name[step.name]
        statements.append(
            f"let {local}: i64 = {expression};" if rust else f"int64_t {local} = {expression};"
        )
    return statements, arrays, name[program.output]


__all__ = ["TABLE_PREFIX", "Program", "Step", "emit_c", "emit_verilog"]
