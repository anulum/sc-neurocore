# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Verilog compiler implementation

"""Compile EquationNeuron to synthesizable Verilog RTL.

Two emitters share one combinational core (:func:`_build_neuron_core`):

* :func:`compile_to_verilog` — the per-instance module: internal state registers
  advanced in an ``always`` block. This is the bit-true, golden path used by the
  direct/AER interconnects.
* :func:`compile_to_datapath` — the same arithmetic as a **combinational
  processing element** with state carried on ports (``<var>_reg`` inputs,
  ``<var>_next_out`` outputs). One PE is time-multiplexed across many neurons in
  the folded interconnect, so the next-state arithmetic must be — and is, by
  construction — identical to the per-instance module.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ..hdl_gen._ident import sanitize_ident
from ..neurons.equation_builder import EquationNeuron
from .verilog_compiler_config import Q88
from .verilog_expr_emitter import _emit_expr


@dataclass
class _NeuronCore:
    """Shared combinational building blocks for one neuron's step.

    Both the registered per-instance module and the combinational datapath PE are
    assembled from these identical fragments, so their arithmetic is bit-for-bit
    the same. State variables are referenced as ``<safe_var>_reg`` throughout
    (the per-instance module declares those as registers; the datapath PE declares
    them as input ports).
    """

    state_var_map: dict[str, str]
    param_map: dict[str, str]
    param_decls: list[str]
    intermediates: list[str]
    pipeline_regs: list[str]
    deriv_wires: list[str]
    next_wires: list[str]
    threshold_verilog: str
    reset_assignments: list[str]

    @property
    def total_pipeline_latency(self) -> int:
        return len(self.pipeline_regs)


def _build_neuron_core(
    neuron: EquationNeuron,
    q: Q88,
    *,
    data_width: int,
    fraction: int,
    pipeline_stages: int,
    pipeline_points: list[str] | None,
) -> _NeuronCore:
    """Emit the combinational next-state + threshold + reset fragments.

    This is the logic shared verbatim by :func:`compile_to_verilog` and
    :func:`compile_to_datapath`; neither wraps nor mutates it differently, which
    is what guarantees bit-exact agreement between the per-instance module and the
    folded datapath.
    """
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

    return _NeuronCore(
        state_var_map=state_var_map,
        param_map=param_map,
        param_decls=param_decls,
        intermediates=all_intermediates,
        pipeline_regs=all_pipeline_regs,
        deriv_wires=deriv_wires,
        next_wires=next_wires,
        threshold_verilog=threshold_verilog,
        reset_assignments=reset_assignments,
    )


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
    core = _build_neuron_core(
        neuron,
        q,
        data_width=data_width,
        fraction=fraction,
        pipeline_stages=pipeline_stages,
        pipeline_points=pipeline_points,
    )
    state_var_map = core.state_var_map
    param_decls = core.param_decls
    all_intermediates = core.intermediates
    all_pipeline_regs = core.pipeline_regs
    deriv_wires = core.deriv_wires
    next_wires = core.next_wires
    threshold_verilog = core.threshold_verilog
    reset_assignments = core.reset_assignments
    total_pipeline_latency = core.total_pipeline_latency

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


def compile_to_datapath(
    neuron: EquationNeuron,
    module_name: str = "sc_equation_neuron_pe",
    data_width: int = 16,
    fraction: int = 8,
    *,
    signed: bool = True,
    overflow: str = "saturate",
    rounding: str = "truncate",
    param_ports: Sequence[str] = (),
) -> str:
    """Compile an EquationNeuron to a **combinational** processing element.

    State is carried on ports — ``<var>_reg`` inputs and ``<var>_next_out``
    outputs — instead of internal registers, so one PE can be time-multiplexed
    across many neurons (the folded interconnect stores per-neuron state in BRAM
    and streams it through this PE one neuron per cycle). ``spike_out`` and each
    ``<var>_next_out`` are the post-threshold, post-reset next values, computed by
    the same fragments as :func:`compile_to_verilog` — so the folded datapath is
    bit-for-bit identical to the per-instance module.

    ``param_ports`` names the parameters to carry on **input ports** instead of
    baking them as module ``parameter`` defaults. A folded population whose neurons
    have heterogeneous parameters drives these ports from a per-neuron parameter ROM
    (one value per neuron, streamed by the sequencer), the parameter-space analogue of
    the state BRAM. Because the arithmetic body references each parameter by the same
    ``P_<NAME>`` identifier whether it is a ``parameter`` or an ``input wire``, moving a
    parameter to a port changes only its declaration — the datapath stays bit-for-bit
    identical. Every name must be a real neuron parameter/constant; the rest stay baked.

    Pipelining is not supported here (a combinational PE has no register stages);
    the folded sequencer provides the one-cycle-per-neuron timing instead.
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
                f"smallest representable non-zero value is {min_representable}."
            )

    safe_module_name = sanitize_ident(module_name, context="module name")
    core = _build_neuron_core(
        neuron,
        q,
        data_width=data_width,
        fraction=fraction,
        pipeline_stages=0,
        pipeline_points=None,
    )
    state_var_map = core.state_var_map

    # Partition parameters into those baked as module ``parameter`` defaults and those
    # carried on input ports (``param_ports``), so a folded population can stream
    # per-neuron parameters through the PE. Both keep the same ``P_<NAME>`` identifier,
    # so only the declaration differs — the arithmetic body is untouched.
    unknown = [p for p in param_ports if p not in core.param_map]
    if unknown:
        raise ValueError(f"param_ports names are not parameters of {neuron!r}: {sorted(unknown)}")
    port_vnames = [core.param_map[p] for p in param_ports]
    baked_decls = [
        decl for decl in core.param_decls if not any(f" {vn} =" in decl for vn in port_vnames)
    ]

    lines = [
        "// Auto-generated by SC-NeuroCore equation compiler (folded datapath PE)",
        f"// Source: {neuron!r}",
        f"// Fixed-point: Q{data_width - fraction}.{fraction} ({data_width}-bit signed)",
        "// Combinational next-state: state carried on <var>_reg inputs / <var>_next_out outputs.",
        "`timescale 1ns / 1ps",
        "",
        f"module {safe_module_name} #(",
    ]
    if baked_decls:
        lines.append(",\n".join(baked_decls))
        lines.append(")(")
    else:
        lines[-1] = f"module {safe_module_name} ("
    lines.append(f"    input wire signed [{data_width - 1}:0] I_t,")
    # Heterogeneous parameters carried on input ports, streamed from a per-neuron ROM.
    for vname in port_vnames:
        lines.append(f"    input wire signed [{data_width - 1}:0] {vname},")
    # State carried in on ports (named <var>_reg so the shared core wires match).
    for var in neuron.equations:
        safe_var = state_var_map[var]
        lines.append(f"    input wire signed [{data_width - 1}:0] {safe_var}_reg,")
    lines.append("    output wire spike_out,")
    for var in neuron.equations:
        safe_var = state_var_map[var]
        lines.append(f"    output wire signed [{data_width - 1}:0] {safe_var}_next_out,")
    lines[-1] = lines[-1].rstrip(",")
    lines.append(");")
    lines.append("")

    for wire in core.intermediates:
        lines.append(wire)
    lines.append("")
    for wire in core.deriv_wires:
        lines.append(wire)
    lines.append("")
    for wire in core.next_wires:
        lines.append(wire)
    lines.append("")

    # Spike is the combinational threshold over the candidate next state.
    if core.threshold_verilog:
        lines.append(f"assign spike_out = ({core.threshold_verilog});")
    else:
        lines.append("assign spike_out = 1'b0;")

    # Next-state output: on spike, apply reset rules (or hold <var>_next when the
    # variable has no reset rule); otherwise advance to <var>_next. This mirrors
    # the per-instance always block exactly.
    reset_map: dict[str, str] = {}
    for assign in core.reset_assignments:
        # Each entry is "            <safe_var>_reg <= <expr>;"
        body = assign.strip().rstrip(";")
        lhs, rhs = body.split("<=", 1)
        safe_var = lhs.strip().removesuffix("_reg")
        reset_map[safe_var] = rhs.strip()

    for var in neuron.equations:
        safe_var = state_var_map[var]
        on_spike = reset_map.get(safe_var, f"{safe_var}_next")
        if core.threshold_verilog:
            lines.append(
                f"assign {safe_var}_next_out = spike_out ? ({on_spike}) : {safe_var}_next;"
            )
        else:
            lines.append(f"assign {safe_var}_next_out = {safe_var}_next;")

    lines.append("")
    lines.append("endmodule")

    return "\n".join(lines)
