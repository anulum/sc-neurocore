# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Equation → Verilog RTL compiler

"""Compile arbitrary ODE neuron equations to synthesizable Verilog.

The only framework that goes from a string equation to FPGA hardware:

    from sc_neurocore.neurons.equation_builder import from_equations
    from sc_neurocore.compiler.equation_compiler import compile_to_verilog

    neuron = from_equations("dv/dt = -(v - E_L)/tau_m + I/C",
                            threshold="v > -50", reset="v = -65",
                            params=dict(E_L=-65, tau_m=10, C=1),
                            init=dict(v=-65))

    verilog = compile_to_verilog(neuron, module_name="my_lif")

All arithmetic uses Q8.8 signed fixed-point. Each ODE term becomes
a multiply-shift pipeline stage. Threshold and reset map to
combinational comparators and mux logic.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass

from ..neurons.equation_builder import EquationNeuron


@dataclass
class Q88:
    """Q8.8 fixed-point conversion: 8 integer bits, 8 fractional bits, signed."""

    data_width: int = 16
    fraction: int = 8

    def encode(self, value: float) -> int:
        raw = int(round(value * (1 << self.fraction)))
        mask = (1 << self.data_width) - 1
        return raw & mask

    def encode_signed_literal(self, value: float) -> str:
        raw = int(round(value * (1 << self.fraction)))
        if raw < 0:
            raw = raw & ((1 << self.data_width) - 1)
        return (
            f"{self.data_width}'sd{raw}"
            if raw >= 0
            else f"-{self.data_width}'sd{abs(int(round(value * (1 << self.fraction))))}"
        )


class _VerilogExprEmitter(ast.NodeVisitor):
    """Walk a Python AST and emit equivalent Verilog fixed-point expressions.

    Handles: +, -, *, /, **, unary minus, comparisons, names, constants.
    Multiplications emit wide product with arithmetic right shift.
    """

    def __init__(self, state_vars: set[str], param_map: dict[str, str], q: Q88):
        self.state_vars = state_vars
        self.param_map = param_map
        self.q = q
        self._mul_count = 0
        self.intermediates: list[str] = []

    def visit_BinOp(self, node: ast.BinOp) -> str:
        left = self.visit(node.left)
        right = self.visit(node.right)

        if isinstance(node.op, ast.Add):
            return f"({left} + {right})"
        elif isinstance(node.op, ast.Sub):
            return f"({left} - {right})"
        elif isinstance(node.op, ast.Mult):
            # Fixed-point multiply: (a * b) >>> FRACTION
            tmp = f"_mul{self._mul_count}"
            self._mul_count += 1
            self.intermediates.append(
                f"wire signed [{2 * self.q.data_width - 1}:0] {tmp} = {left} * {right};"
            )
            return f"({tmp} >>> {self.q.fraction})[{self.q.data_width - 1}:0]"
        elif isinstance(node.op, ast.Div):
            # Division by constant → multiply by reciprocal
            if isinstance(node.right, ast.Constant):
                recip = 1.0 / node.right.value
                recip_q = self.q.encode_signed_literal(recip)
                tmp = f"_mul{self._mul_count}"
                self._mul_count += 1
                self.intermediates.append(
                    f"wire signed [{2 * self.q.data_width - 1}:0] {tmp} = {left} * {recip_q};"
                )
                return f"({tmp} >>> {self.q.fraction})[{self.q.data_width - 1}:0]"
            return f"({left} / {right})"
        elif isinstance(node.op, ast.Pow):
            # x**2 → x * x, x**3 → x * x * x (small integer powers only)
            if isinstance(node.right, ast.Constant) and node.right.value == 2:
                tmp = f"_mul{self._mul_count}"
                self._mul_count += 1
                self.intermediates.append(
                    f"wire signed [{2 * self.q.data_width - 1}:0] {tmp} = {left} * {left};"
                )
                return f"({tmp} >>> {self.q.fraction})[{self.q.data_width - 1}:0]"
            elif isinstance(node.right, ast.Constant) and node.right.value == 3:
                sq = f"_mul{self._mul_count}"
                self._mul_count += 1
                self.intermediates.append(
                    f"wire signed [{2 * self.q.data_width - 1}:0] {sq} = {left} * {left};"
                )
                sq_trunc = f"({sq} >>> {self.q.fraction})[{self.q.data_width - 1}:0]"
                cu = f"_mul{self._mul_count}"
                self._mul_count += 1
                self.intermediates.append(
                    f"wire signed [{2 * self.q.data_width - 1}:0] {cu} = {sq_trunc} * {left};"
                )
                return f"({cu} >>> {self.q.fraction})[{self.q.data_width - 1}:0]"
            raise ValueError(
                f"Only integer powers 2 and 3 supported in Verilog, got {node.right.value}"
            )
        raise ValueError(f"Unsupported binary op: {type(node.op).__name__}")

    def visit_UnaryOp(self, node: ast.UnaryOp) -> str:
        operand = self.visit(node.operand)
        if isinstance(node.op, ast.USub):
            return f"(-{operand})"
        if isinstance(node.op, ast.UAdd):
            return operand
        raise ValueError(f"Unsupported unary op: {type(node.op).__name__}")

    def visit_Name(self, node: ast.Name) -> str:
        name = node.id
        if name in self.state_vars:
            return f"{name}_reg"
        if name in self.param_map:
            return self.param_map[name]
        if name == "I":
            return "I_t"
        return name

    def visit_Constant(self, node: ast.Constant) -> str:
        return self.q.encode_signed_literal(float(node.value))

    def visit_Compare(self, node: ast.Compare) -> str:
        left = self.visit(node.left)
        results = []
        for op, comp in zip(node.ops, node.comparators):
            right = self.visit(comp)
            if isinstance(op, ast.Gt):
                results.append(f"({left} > {right})")
            elif isinstance(op, ast.GtE):
                results.append(f"({left} >= {right})")
            elif isinstance(op, ast.Lt):
                results.append(f"({left} < {right})")
            elif isinstance(op, ast.LtE):
                results.append(f"({left} <= {right})")
            else:
                raise ValueError(f"Unsupported comparison: {type(op).__name__}")
        return " && ".join(results)

    def generic_visit(self, node):
        raise ValueError(f"Unsupported AST node for Verilog: {type(node).__name__}")


def _emit_expr(
    expr_str: str, state_vars: set[str], param_map: dict[str, str], q: Q88
) -> tuple[str, list[str]]:
    """Parse a Python expression string and return (verilog_expr, intermediate_wires)."""
    tree = ast.parse(expr_str, mode="eval")
    emitter = _VerilogExprEmitter(state_vars, param_map, q)
    result = emitter.visit(tree.body)
    return result, emitter.intermediates


def compile_to_verilog(
    neuron: EquationNeuron,
    module_name: str = "sc_equation_neuron",
    data_width: int = 16,
    fraction: int = 8,
) -> str:
    """Compile an EquationNeuron to synthesizable Verilog RTL.

    Parameters
    ----------
    neuron : EquationNeuron
        The neuron defined by arbitrary ODE strings.
    module_name : str
        Name of the generated Verilog module.
    data_width : int
        Bit width for fixed-point arithmetic (default 16 = Q8.8).
    fraction : int
        Number of fractional bits (default 8).

    Returns
    -------
    str
        Synthesizable Verilog source code.
    """
    q = Q88(data_width=data_width, fraction=fraction)
    state_vars = set(neuron.equations.keys())

    # Build parameter map: Python name → Verilog parameter name
    param_map: dict[str, str] = {}
    param_decls: list[str] = []
    for pname, pval in {**neuron.parameters, **neuron.constants}.items():
        vname = f"P_{pname.upper()}"
        param_map[pname] = vname
        q_val = q.encode(pval)
        param_decls.append(
            f"    parameter signed [{data_width - 1}:0] {vname} = {data_width}'sd{q_val}"
        )

    # Generate derivative expressions
    deriv_wires: list[str] = []
    deriv_assigns: list[str] = []
    all_intermediates: list[str] = []

    for var, expr_str in neuron.equations.items():
        vexpr, intermediates = _emit_expr(expr_str, state_vars, param_map, q)
        all_intermediates.extend(intermediates)
        # dv = expr * dt (multiply by dt in fixed-point)
        dt_literal = q.encode_signed_literal(neuron.dt)
        dt_tmp = f"_dt_mul_{var}"
        all_intermediates.append(
            f"wire signed [{2 * data_width - 1}:0] {dt_tmp} = ({vexpr}) * {dt_literal};"
        )
        deriv_name = f"d{var}"
        deriv_wires.append(
            f"wire signed [{data_width - 1}:0] {deriv_name} = ({dt_tmp} >>> {fraction})[{data_width - 1}:0];"
        )

    # Next-state computation
    next_wires: list[str] = []
    for var in neuron.equations:
        next_wires.append(f"wire signed [{data_width - 1}:0] {var}_next = {var}_reg + d{var};")

    # Threshold expression
    threshold_verilog = ""
    if neuron.threshold_expr:
        threshold_verilog, thr_intermediates = _emit_expr(
            neuron.threshold_expr, state_vars, param_map, q
        )
        all_intermediates.extend(thr_intermediates)

    # Reset assignments
    reset_assignments: list[str] = []
    for var, expr_str in neuron.reset_rules.items():
        rexpr, r_intermediates = _emit_expr(expr_str, state_vars, param_map, q)
        all_intermediates.extend(r_intermediates)
        reset_assignments.append(f"                    {var}_reg <= {rexpr};")

    # Build the Verilog module
    lines = [
        "// Auto-generated by SC-NeuroCore equation compiler",
        f"// Source: {neuron!r}",
        f"// Fixed-point: Q{data_width - fraction}.{fraction} ({data_width}-bit signed)",
        "`timescale 1ns / 1ps",
        "",
        f"module {module_name} #(",
    ]
    lines.append(",\n".join(param_decls))
    lines.append(")(")
    lines.append("    input wire clk,")
    lines.append("    input wire rst_n,")
    lines.append(f"    input wire signed [{data_width - 1}:0] I_t,")
    lines.append("    output reg spike_out,")

    # Output ports for each state variable
    for var in neuron.equations:
        lines.append(f"    output reg signed [{data_width - 1}:0] {var}_out,")
    # Remove trailing comma from last port
    lines[-1] = lines[-1].rstrip(",")
    lines.append(");")
    lines.append("")

    # State registers
    for var in neuron.equations:
        init_val = q.encode_signed_literal(neuron.initial_state.get(var, 0.0))
        lines.append(f"reg signed [{data_width - 1}:0] {var}_reg;")

    lines.append("")

    # Intermediate wires (multiply pipelines)
    for wire in all_intermediates:
        lines.append(wire)
    lines.append("")

    # Derivative wires
    for wire in deriv_wires:
        lines.append(wire)
    lines.append("")

    # Next-state wires
    for wire in next_wires:
        lines.append(wire)
    lines.append("")

    # Sequential logic
    lines.append("always @(posedge clk or negedge rst_n) begin")
    lines.append("    if (!rst_n) begin")
    for var in neuron.equations:
        init_val = q.encode_signed_literal(neuron.initial_state.get(var, 0.0))
        lines.append(f"        {var}_reg <= {init_val};")
        lines.append(f"        {var}_out <= {init_val};")
    lines.append("        spike_out <= 1'b0;")
    lines.append("    end else begin")

    if threshold_verilog:
        lines.append(f"        if ({threshold_verilog}) begin")
        lines.append("            spike_out <= 1'b1;")
        for assign in reset_assignments:
            lines.append(assign)
        # State vars not in reset keep their next value
        for var in neuron.equations:
            if var not in neuron.reset_rules:
                lines.append(f"            {var}_reg <= {var}_next;")
        for var in neuron.equations:
            reset_val = (
                f"{param_map.get(var + '_reset_val', var + '_next')}"
                if var in neuron.reset_rules
                else f"{var}_next"
            )
            lines.append(f"            {var}_out <= {var}_reg;")
        lines.append("        end else begin")
        lines.append("            spike_out <= 1'b0;")
        for var in neuron.equations:
            lines.append(f"            {var}_reg <= {var}_next;")
            lines.append(f"            {var}_out <= {var}_next;")
        lines.append("        end")
    else:
        lines.append("        spike_out <= 1'b0;")
        for var in neuron.equations:
            lines.append(f"        {var}_reg <= {var}_next;")
            lines.append(f"        {var}_out <= {var}_next;")

    lines.append("    end")
    lines.append("end")
    lines.append("")
    lines.append("endmodule")

    return "\n".join(lines)


def equation_to_fpga(
    *equation_strings: str,
    threshold: str | None = None,
    reset: str | None = None,
    params: dict[str, float] | None = None,
    init: dict[str, float] | None = None,
    dt: float = 0.1,
    module_name: str = "sc_equation_neuron",
) -> tuple[EquationNeuron, str]:
    """One-liner: ODE string → (Python neuron, Verilog RTL).

    >>> neuron, verilog = equation_to_fpga(
    ...     "dv/dt = -(v - E_L)/tau_m + I/C",
    ...     threshold="v > -50", reset="v = -65",
    ...     params=dict(E_L=-65, tau_m=10, C=1),
    ...     init=dict(v=-65),
    ... )
    """
    from ..neurons.equation_builder import from_equations

    neuron = from_equations(
        *equation_strings,
        threshold=threshold,
        reset=reset,
        params=params,
        init=init,
        dt=dt,
    )
    verilog = compile_to_verilog(neuron, module_name=module_name)
    return neuron, verilog
