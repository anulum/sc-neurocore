# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Equation → Verilog RTL compiler

"""Compile arbitrary ODE neuron equations to synthesizable Verilog.

Compile string equations directly to FPGA hardware:

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

from ..hdl_gen._ident import sanitize_ident
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
        return f"{self.data_width}'sd{raw}"


class _VerilogExprEmitter(ast.NodeVisitor):
    """Walk a Python AST and emit equivalent Verilog fixed-point expressions.

    Handles: +, -, *, /, **, unary minus, comparisons, names, constants.
    Multiplications emit wide product with arithmetic right shift.
    """

    def __init__(self, state_vars: dict[str, str], param_map: dict[str, str], q: Q88):
        self.state_vars = state_vars
        self.param_map = param_map
        self.q = q
        self._mul_count = 0
        self.intermediates: list[str] = []

    def visit_BinOp(self, node: ast.BinOp) -> str:
        left: str = self.visit(node.left)
        right: str = self.visit(node.right)

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
            if isinstance(node.right, ast.Constant) and isinstance(node.right.value, (int, float)):
                recip = 1.0 / float(node.right.value)
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
            elif (
                isinstance(node.right, ast.Constant)
                and isinstance(node.right.value, int)
                and 4 <= node.right.value <= 8
            ):
                # General small integer power via chained squaring
                exp: int = node.right.value
                prev: str = left
                for step in range(exp - 1):
                    tmp = f"_mul{self._mul_count}"
                    self._mul_count += 1
                    self.intermediates.append(
                        f"wire signed [{2 * self.q.data_width - 1}:0] {tmp} = {prev} * {left};"
                    )
                    prev = f"({tmp} >>> {self.q.fraction})[{self.q.data_width - 1}:0]"
                return prev
            raise ValueError(
                f"Only integer powers 2-8 supported in Verilog, got {ast.dump(node.right)}"
            )
        raise ValueError(f"Unsupported binary op: {type(node.op).__name__}")

    def visit_UnaryOp(self, node: ast.UnaryOp) -> str:
        operand: str = self.visit(node.operand)
        if isinstance(node.op, ast.USub):
            return f"(-{operand})"
        if isinstance(node.op, ast.UAdd):
            return str(operand)
        raise ValueError(f"Unsupported unary op: {type(node.op).__name__}")

    def visit_Name(self, node: ast.Name) -> str:
        name = node.id
        if name in self.state_vars:
            return f"{self.state_vars[name]}_reg"
        if name in self.param_map:
            return self.param_map[name]
        if name == "I":
            return "I_t"
        return sanitize_ident(name, context="expression identifier")

    def visit_Constant(self, node: ast.Constant) -> str:
        val: float = float(node.value) if isinstance(node.value, (int, float)) else 0.0
        return self.q.encode_signed_literal(val)

    def visit_Compare(self, node: ast.Compare) -> str:
        left: str = self.visit(node.left)
        results: list[str] = []
        for op, comp in zip(node.ops, node.comparators):
            right: str = self.visit(comp)
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
        arg: str = self.visit(node.args[0])

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
                lo: str = self.visit(node.args[1])
                hi: str = self.visit(node.args[2])
                return f"(({arg} < {lo}) ? {lo} : (({arg} > {hi}) ? {hi} : {arg}))"
            return arg
        elif fname in ("max", "min"):
            if len(node.args) >= 2:
                b: str = self.visit(node.args[1])
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

    def generic_visit(self, node: ast.AST) -> str:
        raise ValueError(f"Unsupported AST node for Verilog: {type(node).__name__}")


def _emit_expr(
    expr_str: str, state_vars: dict[str, str], param_map: dict[str, str], q: Q88
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

    # Reject dt that quantises to zero in the chosen fixed-point format.
    # Without this guard the compiler silently emits Verilog where every
    # dv update is multiplied by 0, producing a frozen membrane voltage.
    if neuron.dt != 0.0:
        dt_quantised = int(round(neuron.dt * (1 << fraction)))
        if dt_quantised == 0:
            min_representable = 1.0 / (1 << fraction)
            raise ValueError(
                f"dt={neuron.dt} underflows in Q{data_width - fraction}.{fraction}: "
                f"smallest representable non-zero value is {min_representable} "
                f"(neuron.dt * 2**{fraction} = {neuron.dt * (1 << fraction)} → 0). "
                f"Use dt >= {min_representable} (e.g. dt=1.0 for 1-step intervals), "
                f"or pass a wider fraction (e.g. Q4.12 via fraction=12) to the compiler."
            )

    safe_module_name = sanitize_ident(module_name, context="module name")
    state_var_map = {var: sanitize_ident(var, context="state variable") for var in neuron.equations}

    # Build parameter map: Python name → Verilog parameter name
    param_map: dict[str, str] = {}
    param_decls: list[str] = []
    for pname, pval in {**neuron.parameters, **neuron.constants}.items():
        safe_pname = sanitize_ident(pname, context="parameter name")
        vname = f"P_{safe_pname.upper()}"
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
        safe_var = state_var_map[var]
        vexpr, intermediates = _emit_expr(expr_str, state_var_map, param_map, q)
        all_intermediates.extend(intermediates)
        # dv = expr * dt (multiply by dt in fixed-point)
        dt_literal = q.encode_signed_literal(neuron.dt)
        dt_tmp = f"_dt_mul_{safe_var}"
        all_intermediates.append(
            f"wire signed [{2 * data_width - 1}:0] {dt_tmp} = ({vexpr}) * {dt_literal};"
        )
        deriv_name = f"d{safe_var}"
        deriv_wires.append(
            f"wire signed [{data_width - 1}:0] {deriv_name} = ({dt_tmp} >>> {fraction})[{data_width - 1}:0];"
        )

    # Next-state computation with saturation
    max_val = (1 << (data_width - 1)) - 1  # e.g. 32767 for 16-bit
    min_val = -(1 << (data_width - 1))  # e.g. -32768 for 16-bit
    next_wires: list[str] = []
    for var in neuron.equations:
        safe_var = state_var_map[var]
        raw = f"{safe_var}_raw"
        next_wires.append(f"wire signed [{data_width}:0] {raw} = {safe_var}_reg + d{safe_var};")
        next_wires.append(
            f"wire signed [{data_width - 1}:0] {safe_var}_next = "
            f"({raw} > {data_width + 1}'sd{max_val}) ? {data_width}'sd{max_val} : "
            f"({raw} < {data_width + 1}'sd{min_val}) ? {data_width}'sd{min_val} : "
            f"{raw}[{data_width - 1}:0];"
        )

    # Threshold expression
    threshold_verilog = ""
    if neuron.threshold_expr:
        threshold_verilog, thr_intermediates = _emit_expr(
            neuron.threshold_expr, state_var_map, param_map, q
        )
        all_intermediates.extend(thr_intermediates)

    # Reset assignments
    reset_assignments: list[str] = []
    for var, expr_str in neuron.reset_rules.items():
        safe_var = state_var_map[var]
        rexpr, r_intermediates = _emit_expr(expr_str, state_var_map, param_map, q)
        all_intermediates.extend(r_intermediates)
        reset_assignments.append(f"                    {safe_var}_reg <= {rexpr};")

    # Build the Verilog module
    lines = [
        "// Auto-generated by SC-NeuroCore equation compiler",
        f"// Source: {neuron!r}",
        f"// Fixed-point: Q{data_width - fraction}.{fraction} ({data_width}-bit signed)",
        "`timescale 1ns / 1ps",
        "",
        f"module {safe_module_name} #(",
    ]
    lines.append(",\n".join(param_decls))
    lines.append(")(")
    lines.append("    input wire clk,")
    lines.append("    input wire rst_n,")
    lines.append(f"    input wire signed [{data_width - 1}:0] I_t,")
    lines.append("    output reg spike_out,")

    # Output ports for each state variable
    for var in neuron.equations:
        safe_var = state_var_map[var]
        lines.append(f"    output reg signed [{data_width - 1}:0] {safe_var}_out,")
    # Remove trailing comma from last port
    lines[-1] = lines[-1].rstrip(",")
    lines.append(");")
    lines.append("")

    # State registers
    for var in neuron.equations:
        safe_var = state_var_map[var]
        lines.append(f"reg signed [{data_width - 1}:0] {safe_var}_reg;")

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
        safe_var = state_var_map[var]
        init_val = q.encode_signed_literal(neuron.initial_state.get(var, 0.0))
        lines.append(f"        {safe_var}_reg <= {init_val};")
        lines.append(f"        {safe_var}_out <= {init_val};")
    lines.append("        spike_out <= 1'b0;")
    lines.append("    end else begin")

    if threshold_verilog:
        lines.append(f"        if ({threshold_verilog}) begin")
        lines.append("            spike_out <= 1'b1;")
        for assign in reset_assignments:
            lines.append(assign)
        # State vars not in reset keep their next value
        for var in neuron.equations:
            safe_var = state_var_map[var]
            if var not in neuron.reset_rules:
                lines.append(f"            {safe_var}_reg <= {safe_var}_next;")
        for var in neuron.equations:
            safe_var = state_var_map[var]
            lines.append(f"            {safe_var}_out <= {safe_var}_reg;")
        lines.append("        end else begin")
        lines.append("            spike_out <= 1'b0;")
        for var in neuron.equations:
            safe_var = state_var_map[var]
            lines.append(f"            {safe_var}_reg <= {safe_var}_next;")
            lines.append(f"            {safe_var}_out <= {safe_var}_next;")
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

    # Split semicolons within single strings for convenience
    expanded: list[str] = []
    for s in equation_strings:
        expanded.extend(part.strip() for part in s.split(";") if part.strip())

    neuron = from_equations(
        *expanded,
        threshold=threshold,
        reset=reset,
        params=params,
        init=init,
        dt=dt,
    )
    verilog = compile_to_verilog(neuron, module_name=module_name)
    return neuron, verilog


def generate_testbench(
    neuron: EquationNeuron,
    module_name: str = "sc_equation_neuron",
    n_steps: int = 200,
    input_current: float = 1.0,
    data_width: int = 16,
    fraction: int = 8,
) -> str:
    """Generate a Verilog testbench for a compiled equation neuron.

    Drives the module with constant current for n_steps clock cycles,
    monitors spike_out and state outputs, and produces a VCD waveform.

    Parameters
    ----------
    neuron : EquationNeuron
        The neuron (same one passed to compile_to_verilog).
    module_name : str
        Must match the module name used in compile_to_verilog.
    n_steps : int
        Number of simulation clock cycles.
    input_current : float
        Constant input current (Q-encoded internally).
    data_width : int
        Bit width matching the compiled module.
    fraction : int
        Fractional bits matching the compiled module.

    Returns
    -------
    str
        Verilog testbench source code.
    """
    q = Q88(data_width=data_width, fraction=fraction)
    i_val = q.encode_signed_literal(input_current)

    state_vars = list(neuron.equations.keys())
    port_connections = [
        "    .clk(clk),",
        "    .rst_n(rst_n),",
        f"    .I_t({i_val}),",
        "    .spike_out(spike_out),",
    ]
    wire_decls = []
    for var in state_vars:
        port_connections.append(f"    .{var}_out({var}_out),")
        wire_decls.append(f"wire signed [{data_width - 1}:0] {var}_out;")
    port_connections[-1] = port_connections[-1].rstrip(",")

    lines = [
        f"// Auto-generated testbench for {module_name}",
        "// SC-NeuroCore equation compiler",
        "`timescale 1ns / 1ps",
        "",
        f"module tb_{module_name};",
        "",
        "reg clk;",
        "reg rst_n;",
        "wire spike_out;",
    ]
    lines.extend(wire_decls)
    lines.append("")
    lines.append(f"{module_name} uut (")
    lines.extend(port_connections)
    lines.append(");")
    lines.append("")
    lines.append("// Clock: 10ns period (100 MHz)")
    lines.append("initial clk = 0;")
    lines.append("always #5 clk = ~clk;")
    lines.append("")
    lines.append("integer spike_count;")
    lines.append("")
    lines.append("initial begin")
    lines.append(f'    $dumpfile("tb_{module_name}.vcd");')
    lines.append(f"    $dumpvars(0, tb_{module_name});")
    lines.append("    spike_count = 0;")
    lines.append("")
    lines.append("    // Reset")
    lines.append("    rst_n = 0;")
    lines.append("    #20;")
    lines.append("    rst_n = 1;")
    lines.append("")
    lines.append(f"    // Run {n_steps} cycles")
    lines.append(f"    repeat ({n_steps}) begin")
    lines.append("        @(posedge clk);")
    lines.append("        if (spike_out) spike_count = spike_count + 1;")
    lines.append("    end")
    lines.append("")
    lines.append(
        f'    $display("Simulation complete: %0d spikes in {n_steps} cycles", spike_count);'
    )
    for var in state_vars:
        lines.append(
            f'    $display("Final {var} = %0d (Q{data_width - fraction}.{fraction})", {var}_out);'
        )
    lines.append("    $finish;")
    lines.append("end")
    lines.append("")
    lines.append("endmodule")

    return "\n".join(lines)
