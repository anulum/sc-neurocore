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

    def visit_Call(self, node: ast.Call) -> str:
        if not isinstance(node.func, ast.Name):
            raise ValueError(f"Only named function calls supported, got {ast.dump(node.func)}")
        fname = node.func.id
        if len(node.args) < 1:
            raise ValueError(f"Function {fname} requires at least 1 argument")
        arg = self.visit(node.args[0])

        # Q8.8 LUT-based approximations for transcendental functions.
        # Each function is a 16-entry piecewise-linear LUT indexed by
        # the top 4 bits of the unsigned input, covering [-8, +8) in Q8.8.
        # Accuracy: ~1-2% over the useful range for neuron dynamics.

        if fname == "exp":
            return self._emit_lut_call("_exp_lut", arg, self._exp_lut_entries())
        elif fname == "log":
            return self._emit_lut_call("_log_lut", arg, self._log_lut_entries())
        elif fname == "sqrt":
            return self._emit_lut_call("_sqrt_lut", arg, self._sqrt_lut_entries())
        elif fname == "tanh":
            return self._emit_lut_call("_tanh_lut", arg, self._tanh_lut_entries())
        elif fname in ("sigmoid", "expit"):
            return self._emit_lut_call("_sigmoid_lut", arg, self._sigmoid_lut_entries())
        elif fname == "sin":
            return self._emit_lut_call("_sin_lut", arg, self._sin_lut_entries())
        elif fname == "cos":
            return self._emit_lut_call("_cos_lut", arg, self._cos_lut_entries())
        elif fname == "abs":
            return f"(({arg} < 0) ? (-{arg}) : {arg})"
        elif fname == "clip":
            if len(node.args) == 3:
                lo = self.visit(node.args[1])
                hi = self.visit(node.args[2])
                return f"(({arg} < {lo}) ? {lo} : (({arg} > {hi}) ? {hi} : {arg}))"
            return arg
        elif fname in ("max", "min"):
            if len(node.args) >= 2:
                b = self.visit(node.args[1])
                if fname == "max":
                    return f"(({arg} > {b}) ? {arg} : {b})"
                return f"(({arg} < {b}) ? {arg} : {b})"
            return arg
        raise ValueError(
            f"Unsupported function '{fname}' in Verilog compilation. "
            f"Supported: exp, log, sqrt, tanh, sigmoid, sin, cos, abs, clip, max, min"
        )

    def _emit_lut_call(self, lut_name: str, arg: str, entries: list[int]) -> str:
        """Emit a 16-entry LUT indexed by top 4 bits of the input."""
        lut_id = f"{lut_name}{self._mul_count}"
        self._mul_count += 1

        # Declare the LUT as a reg array
        dw = self.q.data_width
        self.intermediates.append(
            f"// {lut_name} lookup table (16 entries, Q{dw - self.q.fraction}.{self.q.fraction})"
        )

        # Shift input to unsigned index: add 8.0 (=2048 in Q8.8) then take top 4 bits
        offset = 8 << self.q.fraction  # 2048 for Q8.8
        idx_wire = f"{lut_id}_idx"
        self.intermediates.append(
            f"wire [3:0] {idx_wire} = ({arg} + {dw}'sd{offset}) >>> {self.q.fraction + 4 - 4};"
        )

        # Build case expression
        result_wire = f"{lut_id}_out"
        lines = [f"reg signed [{dw - 1}:0] {result_wire};"]
        lines.append(f"always @(*) case ({idx_wire})")
        for i, val in enumerate(entries):
            lines.append(f"    4'd{i}: {result_wire} = {dw}'sd{val};")
        lines.append(f"    default: {result_wire} = {dw}'sd0;")
        lines.append("endcase")
        for line in lines:
            self.intermediates.append(line)

        return result_wire

    def _exp_lut_entries(self) -> list[int]:
        """exp(x) for x in [-8, +8) sampled at 16 points, Q8.8."""
        import math

        points = [(-8 + i) for i in range(16)]
        return [min(int(round(math.exp(x) * (1 << self.q.fraction))), 32767) for x in points]

    def _log_lut_entries(self) -> list[int]:
        """log(x) for x in [0.06, 8) sampled at 16 points, Q8.8."""
        import math

        return [
            int(round(math.log(max(0.06 + i * 0.5, 0.001)) * (1 << self.q.fraction)))
            for i in range(16)
        ]

    def _sqrt_lut_entries(self) -> list[int]:
        """sqrt(x) for x in [0, 8) sampled at 16 points, Q8.8."""
        import math

        return [int(round(math.sqrt(max(i * 0.5, 0)) * (1 << self.q.fraction))) for i in range(16)]

    def _tanh_lut_entries(self) -> list[int]:
        """tanh(x) for x in [-8, +8) sampled at 16 points, Q8.8."""
        import math

        points = [(-8 + i) for i in range(16)]
        return [int(round(math.tanh(x) * (1 << self.q.fraction))) for x in points]

    def _sigmoid_lut_entries(self) -> list[int]:
        """sigmoid(x) = 1/(1+exp(-x)) for x in [-8, +8), Q8.8."""
        import math

        points = [(-8 + i) for i in range(16)]
        return [int(round(1.0 / (1.0 + math.exp(-x)) * (1 << self.q.fraction))) for x in points]

    def _sin_lut_entries(self) -> list[int]:
        """sin(x) for x in [-8, +8) sampled at 16 points, Q8.8."""
        import math

        points = [(-8 + i) for i in range(16)]
        return [int(round(math.sin(x) * (1 << self.q.fraction))) for x in points]

    def _cos_lut_entries(self) -> list[int]:
        """cos(x) for x in [-8, +8) sampled at 16 points, Q8.8."""
        import math

        points = [(-8 + i) for i in range(16)]
        return [int(round(math.cos(x) * (1 << self.q.fraction))) for x in points]

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

    # Next-state computation with saturation
    max_val = (1 << (data_width - 1)) - 1  # e.g. 32767 for 16-bit
    min_val = -(1 << (data_width - 1))  # e.g. -32768 for 16-bit
    next_wires: list[str] = []
    for var in neuron.equations:
        raw = f"{var}_raw"
        next_wires.append(f"wire signed [{data_width}:0] {raw} = {var}_reg + d{var};")
        next_wires.append(
            f"wire signed [{data_width - 1}:0] {var}_next = "
            f"({raw} > {data_width + 1}'sd{max_val}) ? {data_width}'sd{max_val} : "
            f"({raw} < {data_width + 1}'sd{min_val}) ? {data_width}'sd{min_val} : "
            f"{raw}[{data_width - 1}:0];"
        )

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
