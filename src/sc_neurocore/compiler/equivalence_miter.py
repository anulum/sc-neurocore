# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Sequential equivalence miter construction

"""Build a sequential-equivalence miter for a compiled RTL module.

A *miter* instantiates two interface-compatible Verilog modules — a
device-under-test (the compiler's generated RTL) and an independent reference —
drives both with identical free inputs and a shared reset, and asserts their
outputs agree on every post-reset cycle. Feeding the miter to a bounded
model checker (see :mod:`sc_neurocore.compiler.equivalence_check`) then proves
the two modules compute the same function for *every* input sequence up to the
checked depth, rather than for the finite set a simulation happens to exercise.

This module is pure: it parses a module's port interface and emits the miter
Verilog. It runs no external tools.

The reset discipline avoids the two failure modes of a naive miter. It does not
use an ``initial`` block or a simulation-only ``always #5 clk`` construct (which
over-constrain the formal initial state into ``PREUNSAT``); instead a
free-running counter — initialised to zero, the one initial value the checker
honours — holds the active-low reset asserted for ``reset_cycles`` clocks, and
the equivalence assertions are gated on the post-reset window so the two modules
are compared only once both have been driven into their reset state.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass

__all__ = ["MiterPort", "parse_module_interface", "build_equivalence_miter"]

# One ANSI port declaration, e.g. ``input wire signed [DATA_WIDTH-1:0] leak_k``.
_PORT_RE = re.compile(
    r"\b(?P<direction>input|output)\b"
    r"(?:\s+(?:wire|reg|logic))?"
    r"(?P<signed>\s+signed)?"
    r"(?:\s*\[(?P<msb>[^:]+):(?P<lsb>[^\]]+)\])?"
    r"\s+(?P<name>[A-Za-z_]\w*)"
)


@dataclass(frozen=True)
class MiterPort:
    """One port of the module interface shared by the DUT and the reference.

    Attributes
    ----------
    name : str
        Port identifier.
    width : int
        Bit width (``1`` for a scalar port).
    signed : bool
        Whether the port is declared ``signed``.
    direction : str
        ``"input"`` or ``"output"``.
    """

    name: str
    width: int
    signed: bool
    direction: str

    def declaration(self, suffix: str = "") -> str:
        """Return a Verilog ``wire`` declaration for this port.

        Parameters
        ----------
        suffix : str
            Appended to the port name (e.g. ``"_dut"`` for a miter output wire).
        """
        signed = " signed" if self.signed else ""
        span = f" [{self.width - 1}:0]" if self.width > 1 else ""
        return f"wire{signed}{span} {self.name}{suffix}"


def _eval_width_expr(expr: str, params: dict[str, int]) -> int:
    """Evaluate a bit-index expression to an integer, substituting parameters.

    Supports integer literals, the parameter names in ``params``, and the
    ``+ - * // << >>`` operators — enough for the ``[DATA_WIDTH-1:0]`` bounds the
    compiler emits. Raises :class:`ValueError` on any unsupported construct so a
    malformed or parameter-dependent width can never silently resolve wrong.
    """

    def _eval(node: ast.AST) -> int:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, int) and not isinstance(node.value, bool):
                return node.value
            raise ValueError(f"non-integer literal in width expression: {node.value!r}")
        if isinstance(node, ast.Name):
            if node.id in params:
                return params[node.id]
            raise ValueError(f"unknown parameter in width expression: {node.id}")
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -_eval(node.operand)
        if isinstance(node, ast.BinOp):
            left, right = _eval(node.left), _eval(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.FloorDiv):
                return left // right
            if isinstance(node.op, ast.LShift):
                return left << right
            if isinstance(node.op, ast.RShift):
                return left >> right
        raise ValueError(f"unsupported width expression: {expr!r}")

    try:
        tree = ast.parse(expr.strip(), mode="eval")
    except SyntaxError as exc:  # pragma: no cover - defensive
        raise ValueError(f"unparsable width expression: {expr!r}") from exc
    return _eval(tree)


def _module_header(verilog: str, top: str) -> str:
    """Return the port-list body of ``module top ( ... );``.

    Skips an optional ``#( ... )`` parameter block. Raises :class:`ValueError`
    if the module or its port list cannot be located.
    """
    match = re.search(rf"\bmodule\s+{re.escape(top)}\b", verilog)
    if match is None:
        raise ValueError(f"module {top!r} not found in Verilog source")
    idx = match.end()
    # Skip an optional parameter block ``#( ... )``.
    hash_idx = verilog.find("#", idx)
    paren_idx = verilog.find("(", idx)
    if hash_idx != -1 and (paren_idx == -1 or hash_idx < paren_idx):
        depth = 0
        i = verilog.find("(", hash_idx)
        if i == -1:
            raise ValueError(f"malformed parameter block for module {top!r}")
        while i < len(verilog):
            if verilog[i] == "(":
                depth += 1
            elif verilog[i] == ")":
                depth -= 1
                if depth == 0:
                    idx = i + 1
                    break
            i += 1
        paren_idx = verilog.find("(", idx)
    if paren_idx == -1:
        raise ValueError(f"port list for module {top!r} not found")
    depth = 0
    i = paren_idx
    while i < len(verilog):
        if verilog[i] == "(":
            depth += 1
        elif verilog[i] == ")":
            depth -= 1
            if depth == 0:
                return verilog[paren_idx + 1 : i]
        i += 1
    raise ValueError(f"unterminated port list for module {top!r}")


def parse_module_interface(
    verilog: str, top: str, *, params: dict[str, int] | None = None
) -> list[MiterPort]:
    """Parse the ANSI port interface of a Verilog module.

    Parameters
    ----------
    verilog : str
        Verilog source containing the module.
    top : str
        Module name whose interface to parse.
    params : dict[str, int], optional
        Values for any parameters used in port width expressions (e.g.
        ``{"DATA_WIDTH": 16}``). Widths that reference an unlisted parameter
        raise :class:`ValueError`.

    Returns
    -------
    list[MiterPort]
        The ports in declaration order.
    """
    params = params or {}
    header = _module_header(verilog, top)
    ports: list[MiterPort] = []
    for decl in _PORT_RE.finditer(header):
        msb, lsb = decl.group("msb"), decl.group("lsb")
        if msb is None:
            width = 1
        else:
            width = _eval_width_expr(msb, params) - _eval_width_expr(lsb, params) + 1
            if width < 1:
                raise ValueError(f"non-positive width for port {decl.group('name')!r}")
        ports.append(
            MiterPort(
                name=decl.group("name"),
                width=width,
                signed=decl.group("signed") is not None,
                direction=decl.group("direction"),
            )
        )
    if not ports:
        raise ValueError(f"no ports parsed for module {top!r}")
    return ports


def _params_block(params: dict[str, int] | None) -> str:
    """Render a ``#(.NAME(value), ...)`` override block, or ``""`` if empty."""
    if not params:
        return ""
    body = ", ".join(f".{name}({value})" for name, value in params.items())
    return f" #({body})"


def _instance(top: str, inst: str, params: dict[str, int] | None, connections: list[str]) -> str:
    """Render a single module instantiation."""
    conn = ",\n        ".join(connections)
    return f"    {top}{_params_block(params)} {inst} (\n        {conn}\n    );"


def build_equivalence_miter(
    dut_top: str,
    ref_top: str,
    io_ports: list[MiterPort],
    *,
    miter_name: str = "equiv_miter",
    dut_params: dict[str, int] | None = None,
    ref_params: dict[str, int] | None = None,
    reset_cycles: int = 2,
    clock: str = "clk",
    reset_n: str = "rst_n",
) -> str:
    """Build a sequential-equivalence miter for two interface-compatible modules.

    Both ``dut_top`` and ``ref_top`` are instantiated with the same ``io_ports``;
    the miter exposes the clock and every non-reset input as free top-level
    inputs (the model checker explores all their values), derives an active-low
    reset held for ``reset_cycles`` clocks from a counter, and asserts that every
    output agrees between the two instances once reset is released.

    Parameters
    ----------
    dut_top, ref_top : str
        Module names of the device-under-test and the reference. Must differ so
        both sources can be read into one design.
    io_ports : list[MiterPort]
        The shared interface. Must contain ``clock`` and ``reset_n`` inputs and
        at least one output.
    miter_name : str
        Name of the generated miter module.
    dut_params, ref_params : dict[str, int], optional
        Parameter overrides applied to the respective instance (the two modules
        may take different parameter sets).
    reset_cycles : int
        Number of leading clocks to hold reset asserted before comparing.
    clock, reset_n : str
        Port names of the clock and active-low reset.

    Returns
    -------
    str
        The complete miter Verilog module.
    """
    if dut_top == ref_top:
        raise ValueError("dut_top and ref_top must differ to co-elaborate")
    if reset_cycles < 1:
        raise ValueError("reset_cycles must be at least 1")

    names = {p.name for p in io_ports}
    if clock not in names:
        raise ValueError(f"clock port {clock!r} not in io_ports")
    if reset_n not in names:
        raise ValueError(f"reset port {reset_n!r} not in io_ports")

    free_inputs = [p for p in io_ports if p.direction == "input" and p.name not in (clock, reset_n)]
    outputs = [p for p in io_ports if p.direction == "output"]
    if not outputs:
        raise ValueError("io_ports must contain at least one output to compare")

    # Miter port list: the clock plus every free (non-reset) input, all driven
    # freely by the checker as top-level inputs.
    header_ports = [f"input wire {clock}"]
    for p in free_inputs:
        signed = " signed" if p.signed else ""
        span = f" [{p.width - 1}:0]" if p.width > 1 else ""
        header_ports.append(f"input wire{signed}{span} {p.name}")
    header = ",\n    ".join(header_ports)

    # Output comparison wires, one pair per output.
    out_wires = "\n".join(
        f"    {p.declaration('_dut')};\n    {p.declaration('_ref')};" for p in outputs
    )

    def _connections(out_suffix: str) -> list[str]:
        conns = [f".{clock}({clock})", f".{reset_n}({reset_n})"]
        conns += [f".{p.name}({p.name})" for p in free_inputs]
        conns += [f".{p.name}({p.name}{out_suffix})" for p in outputs]
        return conns

    dut_inst = _instance(dut_top, "dut", dut_params, _connections("_dut"))
    ref_inst = _instance(ref_top, "ref_model", ref_params, _connections("_ref"))

    asserts = "\n".join(f"            assert({p.name}_dut == {p.name}_ref);" for p in outputs)

    return (
        "// SPDX-License-Identifier: AGPL-3.0-or-later\n"
        f"// SC-NeuroCore — auto-generated equivalence miter ({dut_top} vs {ref_top})\n"
        "`timescale 1ns / 1ps\n"
        "\n"
        f"module {miter_name}(\n    {header}\n);\n"
        f"    localparam integer RESET_CYCLES = {reset_cycles};\n"
        "    reg [7:0] rst_cnt = 0;\n"
        f"    always @(posedge {clock}) if (rst_cnt < 8'hFF) rst_cnt <= rst_cnt + 8'd1;\n"
        f"    wire {reset_n} = (rst_cnt >= RESET_CYCLES);\n"
        "\n"
        f"{out_wires}\n"
        "\n"
        f"{dut_inst}\n"
        "\n"
        f"{ref_inst}\n"
        "\n"
        f"    always @(posedge {clock}) begin\n"
        f"        if ({reset_n}) begin\n"
        f"{asserts}\n"
        "        end\n"
        "    end\n"
        "endmodule\n"
    )
