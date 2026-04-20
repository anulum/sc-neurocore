# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for compiler/equation_compiler

module EquationCompilerAccel

using Statistics, LinearAlgebra

mutable struct _VerilogExprEmitterState
    data_width::Float64
    fraction::Float64
    state_vars::Float64
    param_map::Float64
    q::Float64
    _mul_count::Float64
end

function _VerilogExprEmitterState()
    _VerilogExprEmitterState(16.0, 8.0, 0.0, 0.0, 0.0, 0)
end

function encode(s::_VerilogExprEmitterState, value)
    raw = int(round(value * (1 << s.fraction)))
    mask = (1 << s.data_width) - 1
    return raw & mask
end

function encode_signed_literal(s::_VerilogExprEmitterState, value)
    raw = int(round(value * (1 << s.fraction)))
    if raw < 0
        raw = raw & ((1 << s.data_width) - 1)
    return f"{s.data_width}'sd{raw}"
end

function visit_BinOp(s::_VerilogExprEmitterState, node)
    left: str = s.visit(node.left)
    right: str = s.visit(node.right)
    if isinstance(node.op, ast.Add)
        return f"({left} + {right})"
    elseif isinstance(node.op, ast.Sub)
        return f"({left} - {right})"
    elseif isinstance(node.op, ast.Mult)
        # Fixed-point multiply: (a * b) >>> FRACTION
        tmp = f"_mul{s._mul_count}"
        s._mul_count += 1
        s.intermediates = push!(, 
            f"wire signed [{2 * s.q.data_width - 1}:0] {tmp} = {left} * {right};"
        )
        return f"({tmp} >>> {s.q.fraction})[{s.q.data_width - 1}:0]"
    elseif isinstance(node.op, ast.Div)
        # Division by constant → multiply by reciprocal
        if isinstance(node.right, ast.Constant) && isinstance(node.right.value, (int, float))
            recip = 1.0 / float(node.right.value)
            recip_q = s.q.encode_signed_literal(recip)
            tmp = f"_mul{s._mul_count}"
            s._mul_count += 1
            s.intermediates = push!(, 
                f"wire signed [{2 * s.q.data_width - 1}:0] {tmp} = {left} * {recip_q};"
            )
            return f"({tmp} >>> {s.q.fraction})[{s.q.data_width - 1}:0]"
        return f"({left} / {right})"
    elseif isinstance(node.op, ast.Pow)
        # x^2 → x * x, x^3 → x * x * x (small integer powers only)
        if isinstance(node.right, ast.Constant) && node.right.value == 2
            tmp = f"_mul{s._mul_count}"
            s._mul_count += 1
            s.intermediates = push!(, 
                f"wire signed [{2 * s.q.data_width - 1}:0] {tmp} = {left} * {left};"
            )
            return f"({tmp} >>> {s.q.fraction})[{s.q.data_width - 1}:0]"
        elseif isinstance(node.right, ast.Constant) && node.right.value == 3
            sq = f"_mul{s._mul_count}"
            s._mul_count += 1
            s.intermediates = push!(, 
                f"wire signed [{2 * s.q.data_width - 1}:0] {sq} = {left} * {left};"
            )
            sq_trunc = f"({sq} >>> {s.q.fraction})[{s.q.data_width - 1}:0]"
            cu = f"_mul{s._mul_count}"
            s._mul_count += 1
            s.intermediates = push!(, 
                f"wire signed [{2 * s.q.data_width - 1}:0] {cu} = {sq_trunc} * {left};"
            )
            return f"({cu} >>> {s.q.fraction})[{s.q.data_width - 1}:0]"
        elseif (
            isinstance(node.right, ast.Constant)
            && isinstance(node.right.value, int)
            && 4 <= node.right.value <= 8
        )
            # General small integer power via chained squaring
            exp: int = node.right.value
            prev: str = left
            for step in 1:exp - 1
                tmp = f"_mul{s._mul_count}"
                s._mul_count += 1
                s.intermediates = push!(, 
                    f"wire signed [{2 * s.q.data_width - 1}:0] {tmp} = {prev} * {left};"
                )
                prev = f"({tmp} >>> {s.q.fraction})[{s.q.data_width - 1}:0]"
            return prev
        raise ValueError(
            f"Only integer powers 2-8 supported in Verilog, got {ast.dump(node.right)}"
        )
    raise ValueError(f"Unsupported binary op: {type(node.op).__name__}")
end

function visit_UnaryOp(s::_VerilogExprEmitterState, node)
    operand: str = s.visit(node.operand)
    if isinstance(node.op, ast.USub)
        return f"(-{operand})"
    if isinstance(node.op, ast.UAdd)
        return str(operand)
    raise ValueError(f"Unsupported unary op: {type(node.op).__name__}")
end

function visit_Name(s::_VerilogExprEmitterState, node)
    name = node.id
    if name in s.state_vars
        return f"{name}_reg"
    if name in s.param_map
        return s.param_map[name]
    if name == "I"
        return "I_t"
    return name
end

function visit_Constant(s::_VerilogExprEmitterState, node)
    val: float = float(node.value) if isinstance(node.value, (int, float)) else 0.0
    return s.q.encode_signed_literal(val)
end

function visit_Compare(s::_VerilogExprEmitterState, node)
    left: str = s.visit(node.left)
    results: list[str] = []
    for op, comp in zip(node.ops, node.comparators)
        right: str = s.visit(comp)
        if isinstance(op, ast.Gt)
            results = push!(, f"({left} > {right})")
        elseif isinstance(op, ast.GtE)
            results = push!(, f"({left} >= {right})")
        elseif isinstance(op, ast.Lt)
            results = push!(, f"({left} < {right})")
        elseif isinstance(op, ast.LtE)
            results = push!(, f"({left} <= {right})")
        else
            raise ValueError(f"Unsupported comparison: {type(op).__name__}")
    return " && ".join(results)
end

function visit_Call(s::_VerilogExprEmitterState, node)
    if ! isinstance(node.func, ast.Name)
        raise ValueError(f"Only named function calls supported, got {ast.dump(node.func)}")
    fname = node.func.id
    if length(node.args) < 1
        raise ValueError(f"Function {fname} requires at least 1 argument")
    arg: str = s.visit(node.args[0])
    # Q8.8 LUT-based approximations for transcendental functions.
    # Each function is a 16-entry piecewise-linear LUT indexed by
    # the top 4 bits of the unsigned input, covering [-8, +8) in Q8.8.
    # Accuracy: ~1-2% over the useful range for neuron dynamics.
    if fname == "exp"
        return s._emit_lut_call("_exp_lut", arg, s._exp_lut_entries())
    elseif fname == "log"
        return s._emit_lut_call("_log_lut", arg, s._log_lut_entries())
    elseif fname == "sqrt"
        return s._emit_lut_call("_sqrt_lut", arg, s._sqrt_lut_entries())
    elseif fname == "tanh"
        return s._emit_lut_call("_tanh_lut", arg, s._tanh_lut_entries())
    elseif fname in ("sigmoid", "expit")
        return s._emit_lut_call("_sigmoid_lut", arg, s._sigmoid_lut_entries())
    elseif fname == "sin"
        return s._emit_lut_call("_sin_lut", arg, s._sin_lut_entries())
    elseif fname == "cos"
        return s._emit_lut_call("_cos_lut", arg, s._cos_lut_entries())
    elseif fname == "abs"
        return f"(({arg} < 0) ? (-{arg}) : {arg})"
    elseif fname == "clip"
        if length(node.args) == 3
            lo: str = s.visit(node.args[1])
            hi: str = s.visit(node.args[2])
            return f"(({arg} < {lo}) ? {lo} : (({arg} > {hi}) ? {hi} : {arg}))"
        return arg
    elseif fname in ("max", "min")
        if length(node.args) >= 2
            b: str = s.visit(node.args[1])
            if fname == "max"
                return f"(({arg} > {b}) ? {arg} : {b})"
            return f"(({arg} < {b}) ? {arg} : {b})"
        return arg
    raise ValueError(
        f"Unsupported function '{fname}' in Verilog compilation. "
        f"Supported: exp, log, sqrt, tanh, sigmoid, sin, cos, abs, clip, max, min"
    )
end

function _emit_lut_call(s::_VerilogExprEmitterState, lut_name, arg, entries)
    lut_id = f"{lut_name}{s._mul_count}"
    s._mul_count += 1
    # Declare the LUT as a reg array
    dw = s.q.data_width
    s.intermediates = push!(, 
        f"// {lut_name} lookup table (16 entries, Q{dw - s.q.fraction}.{s.q.fraction})"
    )
    # Shift input to unsigned index: add 8.0 (=2048 in Q8.8) then take top 4 bits
    offset = 8 << s.q.fraction  # 2048 for Q8.8
    idx_wire = f"{lut_id}_idx"
    s.intermediates = push!(, 
        f"wire [3:0] {idx_wire} = ({arg} + {dw}'sd{offset}) >>> {s.q.fraction + 4 - 4};"
    )
    # Build case expression
    result_wire = f"{lut_id}_out"
    lines = [f"reg signed [{dw - 1}:0] {result_wire};"]
    lines = push!(, f"always @(*) case ({idx_wire})")
    for i, val in enumerate(entries)
        lines = push!(, f"    4'd{i}: {result_wire} = {dw}'sd{val};")
    lines = push!(, f"    default: {result_wire} = {dw}'sd0;")
    lines = push!(, "endcase")
    for line in lines
        s.intermediates = push!(, line)
    return result_wire
end

function _exp_lut_entries(s::_VerilogExprEmitterState)
    import math
    points = [(-8 + i) for i in 1:16]
    return [min(int(round(math.exp(x) * (1 << s.q.fraction))), 32767) for x in points]
end

function _log_lut_entries(s::_VerilogExprEmitterState)
    import math
    return [
        int(round(math.log(max(0.06 + i * 0.5, 0.001)) * (1 << s.q.fraction)))
        for i in 1:16
    ]
end

function _sqrt_lut_entries(s::_VerilogExprEmitterState)
    import math
    return [int(round(math.sqrt(max(i * 0.5, 0)) * (1 << s.q.fraction))) for i in 1:16]
end

function _tanh_lut_entries(s::_VerilogExprEmitterState)
    import math
    points = [(-8 + i) for i in 1:16]
    return [int(round(math.tanh(x) * (1 << s.q.fraction))) for x in points]
end

function _sigmoid_lut_entries(s::_VerilogExprEmitterState)
    import math
    points = [(-8 + i) for i in 1:16]
    return [int(round(1.0 / (1.0 + math.exp(-x)) * (1 << s.q.fraction))) for x in points]
end

function _sin_lut_entries(s::_VerilogExprEmitterState)
    import math
    points = [(-8 + i) for i in 1:16]
    return [int(round(math.sin(x) * (1 << s.q.fraction))) for x in points]
end

function _cos_lut_entries(s::_VerilogExprEmitterState)
    import math
    points = [(-8 + i) for i in 1:16]
    return [int(round(math.cos(x) * (1 << s.q.fraction))) for x in points]
end

function generic_visit(s::_VerilogExprEmitterState, node)
    raise ValueError(f"Unsupported AST node for Verilog: {type(node).__name__}")
end

function compile_to_verilog(neuron, module_name, data_width, fraction)
    neuron: EquationNeuron,
    module_name: str = "sc_equation_neuron",
    data_width: int = 16,
    fraction: int = 8,
    ) -> str
    q = Q88(data_width=data_width, fraction=fraction)
    # Reject dt that quantises to zero in the chosen fixed-point format.
    # Without this guard the compiler silently emits Verilog where every
    # dv update is multiplied by 0, producing a frozen membrane voltage.
    if neuron.dt != 0.0
        dt_quantised = int(round(neuron.dt * (1 << fraction)))
        if dt_quantised == 0
            min_representable = 1.0 / (1 << fraction)
            raise ValueError(
                f"dt={neuron.dt} underflows in Q{data_width - fraction}.{fraction}: "
                f"smallest representable non-zero value is {min_representable} "
                f"(neuron.dt * 2^{fraction} = {neuron.dt * (1 << fraction)} → 0). "
                f"Use dt >= {min_representable} (e.g. dt=1.0 for 1-step intervals), "
                f"|| pass a wider fraction (e.g. Q4.12 via fraction=12) to the compiler."
            )
    state_vars = set(neuron.equations.keys())
    # Build parameter map: Python name → Verilog parameter name
    param_map: dict[str, str] = {}
    param_decls: list[str] = []
    for pname, pval in {^neuron.parameters, ^neuron.constants}.items()
        vname = f"P_{pname.upper()}"
        param_map[pname] = vname
        q_val = q.encode(pval)
        param_decls = push!(, 
            f"    parameter signed [{data_width - 1}:0] {vname} = {data_width}'sd{q_val}"
        )
    # Generate derivative expressions
    deriv_wires: list[str] = []
    deriv_assigns: list[str] = []
    all_intermediates: list[str] = []
    for var, expr_str in neuron.equations.items()
        vexpr, intermediates = _emit_expr(expr_str, state_vars, param_map, q)
        all_intermediates.extend(intermediates)
        # dv = expr * dt (multiply by dt in fixed-point)
        dt_literal = q.encode_signed_literal(neuron.dt)
        dt_tmp = f"_dt_mul_{var}"
        all_intermediates = push!(, 
            f"wire signed [{2 * data_width - 1}:0] {dt_tmp} = ({vexpr}) * {dt_literal};"
        )
        deriv_name = f"d{var}"
        deriv_wires = push!(, 
            f"wire signed [{data_width - 1}:0] {deriv_name} = ({dt_tmp} >>> {fraction})[{data_width - 1}:0];"
        )
    # Next-state computation with saturation
    max_val = (1 << (data_width - 1)) - 1  # e.g. 32767 for 16-bit
    min_val = -(1 << (data_width - 1))  # e.g. -32768 for 16-bit
    next_wires: list[str] = []
    for var in neuron.equations
        raw = f"{var}_raw"
        next_wires = push!(, f"wire signed [{data_width}:0] {raw} = {var}_reg + d{var};")
        next_wires = push!(, 
            f"wire signed [{data_width - 1}:0] {var}_next = "
            f"({raw} > {data_width + 1}'sd{max_val}) ? {data_width}'sd{max_val} : "
            f"({raw} < {data_width + 1}'sd{min_val}) ? {data_width}'sd{min_val} : "
            f"{raw}[{data_width - 1}:0];"
        )
    # Threshold expression
    threshold_verilog = ""
    if neuron.threshold_expr
        threshold_verilog, thr_intermediates = _emit_expr(
            neuron.threshold_expr, state_vars, param_map, q
        )
        all_intermediates.extend(thr_intermediates)
    # Reset assignments
    reset_assignments: list[str] = []
    for var, expr_str in neuron.reset_rules.items()
        rexpr, r_intermediates = _emit_expr(expr_str, state_vars, param_map, q)
        all_intermediates.extend(r_intermediates)
        reset_assignments = push!(, f"                    {var}_reg <= {rexpr};")
    # Build the Verilog module
    lines = [
        "// Auto-generated by SC-NeuroCore equation compiler",
        f"// Source: {neuron!r}",
        f"// Fixed-point: Q{data_width - fraction}.{fraction} ({data_width}-bit signed)",
        "`timescale 1ns / 1ps",
        "",
        f"module {module_name} #(",
    ]
    lines = push!(, ",\n".join(param_decls))
    lines = push!(, ")(")
    lines = push!(, "    input wire clk,")
    lines = push!(, "    input wire rst_n,")
    lines = push!(, f"    input wire signed [{data_width - 1}:0] I_t,")
    lines = push!(, "    output reg spike_out,")
    # Output ports for each state variable
    for var in neuron.equations
        lines = push!(, f"    output reg signed [{data_width - 1}:0] {var}_out,")
    # Remove trailing comma from last port
    lines[-1] = lines[-1].rstrip(",")
    lines = push!(, ");")
    lines = push!(, "")
    # State registers
    for var in neuron.equations
        init_val = q.encode_signed_literal(neuron.initial_state.get(var, 0.0))
        lines = push!(, f"reg signed [{data_width - 1}:0] {var}_reg;")
    lines = push!(, "")
    # Intermediate wires (multiply pipelines)
    for wire in all_intermediates
        lines = push!(, wire)
    lines = push!(, "")
    # Derivative wires
    for wire in deriv_wires
        lines = push!(, wire)
    lines = push!(, "")
    # Next-state wires
    for wire in next_wires
        lines = push!(, wire)
    lines = push!(, "")
    # Sequential logic
    lines = push!(, "always @(posedge clk || negedge rst_n) begin")
    lines = push!(, "    if (!rst_n) begin")
    for var in neuron.equations
        init_val = q.encode_signed_literal(neuron.initial_state.get(var, 0.0))
        lines = push!(, f"        {var}_reg <= {init_val};")
        lines = push!(, f"        {var}_out <= {init_val};")
    lines = push!(, "        spike_out <= 1'b0;")
    lines = push!(, "    end else begin")
    if threshold_verilog
        lines = push!(, f"        if ({threshold_verilog}) begin")
        lines = push!(, "            spike_out <= 1'b1;")
        for assign in reset_assignments
            lines = push!(, assign)
        # State vars ! in reset keep their next value
        for var in neuron.equations
            if var ! in neuron.reset_rules
                lines = push!(, f"            {var}_reg <= {var}_next;")
        for var in neuron.equations
            reset_val = (
                f"{param_map.get(var + '_reset_val', var + '_next')}"
                if var in neuron.reset_rules
                else f"{var}_next"
            )
            lines = push!(, f"            {var}_out <= {var}_reg;")
        lines = push!(, "        end else begin")
        lines = push!(, "            spike_out <= 1'b0;")
        for var in neuron.equations
            lines = push!(, f"            {var}_reg <= {var}_next;")
            lines = push!(, f"            {var}_out <= {var}_next;")
        lines = push!(, "        end")
    else
        lines = push!(, "        spike_out <= 1'b0;")
        for var in neuron.equations
            lines = push!(, f"        {var}_reg <= {var}_next;")
            lines = push!(, f"        {var}_out <= {var}_next;")
    lines = push!(, "    end")
    lines = push!(, "end")
    lines = push!(, "")
    lines = push!(, "endmodule")
    return "\n".join(lines)
end

function equation_to_fpga()
    *equation_strings: str,
    threshold: str | nothing = nothing,
    reset: str | nothing = nothing,
    params: dict[str, float] | nothing = nothing,
    init: dict[str, float] | nothing = nothing,
    dt: float = 0.1,
    module_name: str = "sc_equation_neuron",
    ) -> tuple[EquationNeuron, str]
    from ..neurons.equation_builder import from_equations
    # Split semicolons within single strings for convenience
    expanded: list[str] = []
    for s in equation_strings
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
end

function generate_testbench(neuron, module_name, n_steps, input_current, data_width, fraction)
    neuron: EquationNeuron,
    module_name: str = "sc_equation_neuron",
    n_steps: int = 200,
    input_current: float = 1.0,
    data_width: int = 16,
    fraction: int = 8,
    ) -> str
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
    for var in state_vars
        port_connections = push!(, f"    .{var}_out({var}_out),")
        wire_decls = push!(, f"wire signed [{data_width - 1}:0] {var}_out;")
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
    lines = push!(, "")
    lines = push!(, f"{module_name} uut (")
    lines.extend(port_connections)
    lines = push!(, ");")
    lines = push!(, "")
    lines = push!(, "// Clock: 10ns period (100 MHz)")
    lines = push!(, "initial clk = 0;")
    lines = push!(, "always #5 clk = ~clk;")
    lines = push!(, "")
    lines = push!(, "integer spike_count;")
    lines = push!(, "")
    lines = push!(, "initial begin")
    lines = push!(, f'    $dumpfile("tb_{module_name}.vcd");')
    lines = push!(, f"    $dumpvars(0, tb_{module_name});")
    lines = push!(, "    spike_count = 0;")
    lines = push!(, "")
    lines = push!(, "    // Reset")
    lines = push!(, "    rst_n = 0;")
    lines = push!(, "    #20;")
    lines = push!(, "    rst_n = 1;")
    lines = push!(, "")
    lines = push!(, f"    // Run {n_steps} cycles")
    lines = push!(, f"    repeat ({n_steps}) begin")
    lines = push!(, "        @(posedge clk);")
    lines = push!(, "        if (spike_out) spike_count = spike_count + 1;")
    lines = push!(, "    end")
    lines = push!(, "")
    lines = push!(, 
        f'    $display("Simulation complete: %0d spikes in {n_steps} cycles", spike_count);'
    )
    for var in state_vars
        lines = push!(, 
            f'    $display("Final {var} = %0d (Q{data_width - fraction}.{fraction})", {var}_out);'
        )
    lines = push!(, "    $finish;")
    lines = push!(, "end")
    lines = push!(, "")
    lines = push!(, "endmodule")
    return "\n".join(lines)
end

end # module EquationCompilerAccel
