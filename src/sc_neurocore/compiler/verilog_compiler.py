# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Verilog compiler implementation

"""Compile EquationNeuron to synthesizable Verilog RTL."""

from __future__ import annotations

from ..hdl_gen._ident import sanitize_ident
from ..neurons.equation_builder import EquationNeuron
from .verilog_compiler_config import Q88
from .verilog_expr_emitter import _emit_expr


def compile_to_verilog(
    neuron: EquationNeuron,
    module_name: str = "sc_equation_neuron",
    data_width: int = 16,
    fraction: int = 8,
    *,
    signed: bool = True,
    overflow: str = "saturate",
    rounding: str = "truncate",
    pipeline_stages: int = 0,
    pipeline_points: list[str] | None = None,
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
    signed : bool
        True for signed two's complement (default), False for unsigned.
    overflow : str
        Overflow mode: ``"saturate"`` (default), ``"wrap"``, or ``"trap"``.
    rounding : str
        Rounding mode: ``"truncate"`` (default), ``"nearest"``,
        ``"bankers"``, or ``"stochastic"``.
    pipeline_stages : int
        Number of pipeline register stages to insert at multiply outputs.
    pipeline_points : list[str], optional
        Explicit list of intermediate signal names.
    """
    q = Q88(
        data_width=data_width,
        fraction=fraction,
        signed=signed,
        overflow=overflow,
        rounding=rounding,
    )

    if neuron.dt != 0.0:
        dt_quantised = int(round(neuron.dt * (1 << fraction)))
        if dt_quantised == 0:
            min_representable = 1.0 / (1 << fraction)
            raise ValueError(
                f"dt={neuron.dt} underflows in Q{data_width - fraction}.{fraction}: "
                f"smallest representable non-zero value is {min_representable}. "
                f"Use dt=1.0 or another value >= {min_representable}, "
                "or increase fractional precision, e.g. fraction=12."
            )

    safe_module_name = sanitize_ident(module_name, context="module name")
    state_var_map = {var: sanitize_ident(var, context="state variable") for var in neuron.equations}

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

    deriv_wires: list[str] = []
    all_intermediates: list[str] = []
    all_pipeline_regs: list[str] = []
    _mc = 0
    _tc = 0

    use_pipeline = pipeline_stages > 0
    pp_set = set(pipeline_points) if pipeline_points and not use_pipeline else set()

    for var, expr_str in neuron.equations.items():
        safe_var = state_var_map[var]
        vexpr, intermediates, _mc, _tc, p_regs = _emit_expr(
            expr_str,
            state_var_map,
            param_map,
            q,
            mul_start=_mc,
            trunc_start=_tc,
            pipeline=use_pipeline,
            pipeline_points=pp_set,
        )
        all_intermediates.extend(intermediates)
        all_pipeline_regs.extend(p_regs)
        dt_literal = q.encode_signed_literal(neuron.dt)
        dt_tmp = f"_dt_mul_{safe_var}"
        dt_should_pipe = use_pipeline or dt_tmp in pp_set
        all_intermediates.append(
            f"wire signed [{2 * data_width - 1}:0] {dt_tmp} = ({vexpr}) * {dt_literal};"
        )
        if dt_should_pipe:
            dt_reg = f"{dt_tmp}_r"
            all_pipeline_regs.append(f"reg signed [{2 * data_width - 1}:0] {dt_reg};")
            deriv_name = f"d{safe_var}"
            deriv_trunc = f"_dt_trunc_{safe_var}"
            all_intermediates.append(
                f"wire signed [{data_width - 1}:0] {deriv_trunc} = ({dt_reg} >>> {fraction});"
            )
        else:
            deriv_name = f"d{safe_var}"
            deriv_trunc = f"_dt_trunc_{safe_var}"
            all_intermediates.append(
                f"wire signed [{data_width - 1}:0] {deriv_trunc} = ({dt_tmp} >>> {fraction});"
            )
        deriv_wires.append(f"wire signed [{data_width - 1}:0] {deriv_name} = {deriv_trunc};")

    sign_kw = "signed " if q.signed else ""
    if q.signed:
        max_val = (1 << (data_width - 1)) - 1
        min_val = -(1 << (data_width - 1))
    else:
        max_val = (1 << data_width) - 1
        min_val = 0

    next_wires: list[str] = []
    for var in neuron.equations:
        safe_var = state_var_map[var]
        raw = f"{safe_var}_raw"
        next_wires.append(f"wire {sign_kw}[{data_width}:0] {raw} = {safe_var}_reg + d{safe_var};")

        if q.overflow == "saturate":
            abs_min = abs(min_val)
            if q.signed:
                next_wires.append(
                    f"wire {sign_kw}[{data_width - 1}:0] {safe_var}_next = "
                    f"({raw} > {data_width + 1}'sd{max_val}) ? {data_width}'sd{max_val} : "
                    f"({raw} < (-{data_width + 1}'sd{abs_min})) ? (-{data_width}'sd{abs_min}) : "
                    f"{raw}[{data_width - 1}:0];"
                )
            else:
                next_wires.append(
                    f"wire [{data_width - 1}:0] {safe_var}_next = "
                    f"({raw} > {data_width + 1}'d{max_val}) ? {data_width}'d{max_val} : "
                    f"({raw}[{data_width}]) ? {data_width}'d0 : "
                    f"{raw}[{data_width - 1}:0];"
                )
        elif q.overflow == "wrap":
            next_wires.append(
                f"wire {sign_kw}[{data_width - 1}:0] {safe_var}_next = {raw}[{data_width - 1}:0];"
            )
        elif q.overflow == "trap":
            abs_min = abs(min_val)
            if q.signed:
                next_wires.append(
                    f"wire {sign_kw}[{data_width - 1}:0] {safe_var}_next = "
                    f"{raw}[{data_width - 1}:0];"
                )
                next_wires.append("// synthesis translate_off")
                next_wires.append(
                    f"always @(*) if ({raw} > {data_width + 1}'sd{max_val} || "
                    f"{raw} < (-{data_width + 1}'sd{abs_min})) "
                    f'$fatal(1, "OVERFLOW TRAP: {safe_var}_raw=%0d", {raw});'
                )
                next_wires.append("// synthesis translate_on")
            else:
                next_wires.append(
                    f"wire [{data_width - 1}:0] {safe_var}_next = {raw}[{data_width - 1}:0];"
                )
                next_wires.append("// synthesis translate_off")
                next_wires.append(
                    f"always @(*) if ({raw}[{data_width}]) "
                    f'$fatal(1, "OVERFLOW TRAP: {safe_var}_raw=%0d", {raw});'
                )
                next_wires.append("// synthesis translate_on")
        else:
            raise ValueError(f"Unknown overflow mode: {q.overflow!r}")

    threshold_verilog = ""
    if neuron.threshold_expr:
        thr_param_map = dict(param_map)
        for var in neuron.equations:
            safe_var = state_var_map[var]
            thr_param_map[var] = f"{safe_var}_next"
        threshold_verilog, thr_intermediates, _mc, _tc, thr_pregs = _emit_expr(
            neuron.threshold_expr,
            {},
            thr_param_map,
            q,
            mul_start=_mc,
            trunc_start=_tc,
        )
        all_intermediates.extend(thr_intermediates)
        all_pipeline_regs.extend(thr_pregs)

    reset_assignments: list[str] = []
    for var, expr_str in neuron.reset_rules.items():
        safe_var = state_var_map[var]
        rexpr, r_intermediates, _mc, _tc, r_pregs = _emit_expr(
            expr_str,
            state_var_map,
            param_map,
            q,
            mul_start=_mc,
            trunc_start=_tc,
        )
        all_intermediates.extend(r_intermediates)
        all_pipeline_regs.extend(r_pregs)
        reset_assignments.append(f"            {safe_var}_reg <= {rexpr};")

    total_pipeline_latency = len(all_pipeline_regs)

    lines = [
        "// Auto-generated by SC-NeuroCore equation compiler",
        f"// Source: {neuron!r}",
        f"// Fixed-point: Q{data_width - fraction}.{fraction} ({data_width}-bit signed)",
        "`timescale 1ns / 1ps",
        "",
        f"module {safe_module_name} #(",
    ]
    if param_decls:
        lines.append(",\n".join(param_decls))
        lines.append(")(")
    else:
        # No parameters: an empty #() clause is malformed; drop it entirely.
        lines[-1] = f"module {safe_module_name} ("
    lines.append("    input wire clk,")
    lines.append("    input wire rst_n,")
    lines.append(f"    input wire signed [{data_width - 1}:0] I_t,")
    lines.append("    output reg spike_out,")

    for var in neuron.equations:
        safe_var = state_var_map[var]
        lines.append(f"    output reg signed [{data_width - 1}:0] {safe_var}_out,")

    if total_pipeline_latency > 0:
        lat_w = max(1, (total_pipeline_latency).bit_length())
        lines.append(f"    output wire [{lat_w - 1}:0] latency,")

    lines[-1] = lines[-1].rstrip(",")
    lines.append(");")
    lines.append("")

    if total_pipeline_latency > 0:
        lat_w = max(1, (total_pipeline_latency).bit_length())
        lines.append(f"// Pipeline latency: {total_pipeline_latency} cycle(s)")
        lines.append(f"assign latency = {lat_w}'d{total_pipeline_latency};")
        lines.append("")

    for var in neuron.equations:
        safe_var = state_var_map[var]
        lines.append(f"reg signed [{data_width - 1}:0] {safe_var}_reg;")

    lines.append("")

    if all_pipeline_regs:
        lines.append("// Pipeline registers for multiply outputs")
        for reg_decl in all_pipeline_regs:
            lines.append(reg_decl)
        lines.append("")

    for wire in all_intermediates:
        lines.append(wire)
    lines.append("")

    for wire in deriv_wires:
        lines.append(wire)
    lines.append("")

    for wire in next_wires:
        lines.append(wire)
    lines.append("")

    if all_pipeline_regs:
        lines.append("// Pipeline register stage — register multiply outputs")
        lines.append("always @(posedge clk) begin")
        for reg_decl in all_pipeline_regs:
            reg_name = reg_decl.split()[-1].rstrip(";")
            wire_name = reg_name[:-2] if reg_name.endswith("_r") else reg_name
            lines.append(f"    {reg_name} <= {wire_name};")
        lines.append("end")
        lines.append("")

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
