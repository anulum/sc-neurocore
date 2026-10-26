# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Shared integer programs as typed native graphs

"""Lower function and counter-based RNG programs to native scalar instructions.

The existing Program remains the numerical source. Every step becomes typed
arithmetic, a comparison or a selection; no executable source or RTL string is
embedded in the graph. Coefficient selection uses balanced trees, sharing
constants and comparisons between columns. Both native hardware emitters can
therefore consume the same persisted graph.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol

from .function_unit_ir import Program, Step


class NativeWordBuilder(Protocol):
    """Graph-builder surface the native compiler lowers into.

    ``sc_neurocore_engine.ir.ScGraphBuilder`` satisfies it. The compiler names
    this structural type instead of the engine class so that its value IDs stay
    typed as ``int`` whether or not the optional engine is installed.
    """

    def constant_i64(self, value: int, ty: str = "i32") -> int:
        """Add an integer constant and return its value ID."""
        ...

    def input(self, name: str, ty: str = "rate") -> int:
        """Add a typed input port and return its value ID."""
        ...

    def output(self, name: str, source_id: int) -> int:
        """Expose an existing value as a named output port."""
        ...

    def runtime_check(
        self, condition: int, kind: str = "overflow", enable: int | None = None
    ) -> int:
        """Append an effectful simulation invariant."""
        ...

    def state(self, initial: int, ty: str) -> int:
        """Declare register storage with a constant reset."""
        ...

    def store(self, state: int, next: int, enable: int | None = None) -> int:
        """Bind one atomic update to a declared state register."""
        ...

    def word_delay(self, input: int, ty: str) -> int:
        """Capture an integer word for one clock with asynchronous zero reset."""
        ...

    def word_binary(self, kind: str, lhs: int, rhs: int, ty: str) -> int:
        """Append modulo-width integer arithmetic."""
        ...

    def word_unary(self, kind: str, input: int, ty: str) -> int:
        """Append integer negation or complement."""
        ...

    def word_compare(self, predicate: str, lhs: int, rhs: int) -> int:
        """Append an integer comparison returning Bool."""
        ...

    def word_resize(self, input: int, ty: str) -> int:
        """Resize an integer word explicitly."""
        ...

    def word_bitcast(self, input: int, ty: str) -> int:
        """Reinterpret a scalar word without changing its bits."""
        ...

    def word_select(self, condition: int, when_true: int, when_false: int, ty: str) -> int:
        """Select a scalar word using a Boolean condition."""
        ...

    def word_clamp(self, input: int, min: int, max: int, ty: str) -> int:
        """Saturate an integer word at explicit inclusive bounds."""
        ...


_ARITY = {
    "input": 0,
    "extern": 0,
    "const": 0,
    "add": 2,
    "sub": 2,
    "mul": 2,
    "div": 2,
    "xor": 2,
    "neg": 1,
    "shl": 1,
    "shr": 1,
    "shl_var": 2,
    "shr_var": 2,
    "mask": 1,
    "lt": 2,
    "le": 2,
    "eq": 2,
    "select": 3,
    "table": 1,
}


def _validate(program: Program, inputs: Sequence[int], externs: Mapping[str, int]) -> None:
    """Check the complete instruction stream before mutating a native builder."""
    program._check_input_count(len(inputs))
    defined: set[str] = set()
    for step in program.steps:
        if not step.name or step.name in defined:
            raise ValueError(f"program step name must be unique: {step.name!r}")
        if step.op not in _ARITY or len(step.args) != _ARITY[step.op]:
            raise ValueError(f"invalid program instruction: {step.op!r}")
        if any(arg not in defined for arg in step.args):
            raise ValueError(f"program step {step.name!r} reads an undefined value")
        if step.op == "input" and not 0 <= step.value < len(inputs):
            raise ValueError("program input index is out of range")
        if step.op in {"shl", "shr", "mask"} and step.value < 0:
            raise ValueError("program shift and mask sizes must be non-negative")
        if step.op == "const" and not -(1 << 63) < step.value < 1 << 63:
            raise ValueError("program constant does not fit its signed 64-bit contract")
        if step.op == "extern" and step.table not in externs:
            raise ValueError(f"missing program external binding: {step.table!r}")
        if step.op == "table":
            rows = program.tables.get(step.table)
            if not rows or not 0 <= step.value < 3:
                raise ValueError("program table needs non-empty rows and coefficient 0, 1 or 2")
            if any(len(row) != 3 or not -(1 << 63) < row[step.value] < 1 << 63 for row in rows):
                raise ValueError("program coefficient does not fit signed 64 bits")
        defined.add(step.name)
    if program.output not in defined:
        raise ValueError("program output must name a defined step")


class _ProgramLowering:
    """Append signed integer instructions and share repeated literal/table nodes."""

    def __init__(self, builder: NativeWordBuilder) -> None:
        """Bind the destination builder and caches for this program invocation."""
        self.builder = builder
        self.constants: dict[int, int] = {}
        self.table_conditions: dict[tuple[int, int], int] = {}
        self.lookups: dict[tuple[str, int, int], int] = {}

    def constant(self, value: int) -> int:
        """Return an existing or newly declared signed 64-bit constant."""
        if value not in self.constants:
            self.constants[value] = self.builder.constant_i64(value, "i64")
        return self.constants[value]

    def lookup(
        self,
        name: str,
        rows: tuple[tuple[int, int, int], ...],
        index: int,
        coefficient: int,
    ) -> int:
        """Select a coefficient, naturally clamping below and above the table."""
        key = (name, index, coefficient)
        if key in self.lookups:
            return self.lookups[key]

        def select(low: int, high: int) -> int:
            if high - low == 1:
                return self.constant(rows[low][coefficient])
            middle = (low + high) // 2
            condition_key = (index, middle)
            if condition_key not in self.table_conditions:
                self.table_conditions[condition_key] = self.builder.word_compare(
                    "lt", index, self.constant(middle)
                )
            return self.builder.word_select(
                self.table_conditions[condition_key],
                select(low, middle),
                select(middle, high),
                "i64",
            )

        result = select(0, len(rows))
        self.lookups[key] = result
        return result

    def emit(
        self,
        step: Step,
        values: Mapping[str, int],
        inputs: Sequence[int],
        externs: Mapping[str, int],
        tables: Mapping[str, tuple[tuple[int, int, int], ...]],
    ) -> int:
        """Lower one already validated instruction, retaining integer truth values."""
        b = self.builder
        operands = [values[name] for name in step.args]
        op = step.op
        if op == "input":
            return inputs[step.value]
        if op == "extern":
            return externs[step.table]
        if op == "const":
            return self.constant(step.value)
        if op in {"add", "sub", "mul", "div", "xor"}:
            return b.word_binary(op, operands[0], operands[1], "i64")
        if op == "neg":
            return b.word_unary("neg", operands[0], "i64")
        if op in {"lt", "le", "eq"}:
            condition = b.word_compare(op, operands[0], operands[1])
            return b.word_resize(condition, "i64")
        if op == "select":
            condition = b.word_compare("ne", operands[0], self.constant(0))
            return b.word_select(condition, operands[1], operands[2], "i64")
        if op == "mask":
            mask = (1 << step.value) - 1 if step.value < 64 else -1
            return b.word_binary("and", operands[0], self.constant(mask), "i64")
        if op in {"shl", "shr", "shl_var", "shr_var"}:
            if op.endswith("_var"):
                amount = b.word_clamp(operands[1], 0, 63, "i64")
            else:
                amount = self.constant(min(step.value, 64))
            return b.word_binary(
                "shl" if op.startswith("shl") else "ashr", operands[0], amount, "i64"
            )
        return self.lookup(step.table, tables[step.table], operands[0], step.value)


def lower_program(
    program: Program,
    builder: NativeWordBuilder,
    inputs: Sequence[int],
    externs: Mapping[str, int] | None = None,
) -> int:
    """Append a complete function/RNG integer program to a native graph.

    Parameters
    ----------
    program : Program
        Shared straight-line integer program from the existing function or
        random unit builders. Every intermediate must fit its signed 64-bit
        contract, as required by the existing compiled C/Rust/RTL targets.
    builder : NativeWordBuilder
        Caller-owned native graph builder, which remains usable afterward.
    inputs : sequence of int
        Native SSA identifiers of signed i64 operands, in program input order.
        Constant programs retain the original single-argument contract.
    externs : mapping of str to int, optional
        Native i64 SSA identifiers of named inputs such as step, seed and stream.
        Unsigned or fixed-point storage needs an explicit caller-side bitcast
        or resize before binding.

    Returns
    -------
    int
        Native SSA identifier of the program's signed i64 output.

    Raises
    ------
    ValueError
        If instructions, operand count, table metadata or bindings are invalid.
        These checks precede graph mutation. Native verification additionally
        checks that supplied SSA identifiers exist and have the required types.

    Notes
    -----
    All nineteen existing operations are lowered without a source/RTL escape
    hatch. Division by zero returns zero for evaluated but unselected branches;
    variable shifts clamp to 0..63; comparisons materialise integer 0 or 1.
    Negative table indices select the first row and large indices the last.
    The resulting graph can be printed, parsed and emitted by both native
    SystemVerilog and CIRCT backends. This does not select a neuron integrator
    or implement the surrounding state/event transaction.
    """
    bound = {} if externs is None else externs
    _validate(program, inputs, bound)
    lowering = _ProgramLowering(builder)
    values: dict[str, int] = {}
    for step in program.steps:
        values[step.name] = lowering.emit(step, values, inputs, bound, program.tables)
    return values[program.output]
