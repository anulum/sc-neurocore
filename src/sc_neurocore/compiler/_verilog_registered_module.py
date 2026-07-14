# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Registered equation-neuron Verilog emission

"""Emit state-owning, clocked Verilog modules from equation neurons."""

from __future__ import annotations

from ..hdl_gen._ident import sanitize_ident
from ..neurons.equation_builder import EquationNeuron
from ._verilog_neuron_core import _build_neuron_core, _escape_threshold_wires
from .verilog_compiler_config import Q88


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
        Must be ``True``. Unsigned emission is rejected because the expression
        and derivative datapaths use signed fixed-point arithmetic.
    overflow : str
        Overflow mode: ``"saturate"`` (default), ``"wrap"``, or ``"trap"``.
    rounding : str
        Rounding mode: ``"truncate"`` (default), ``"nearest"``,
        or ``"bankers"``. ``"stochastic"`` is rejected because the generated
        datapath has no caller-owned product-rounding LFSR.
    pipeline_stages : int
        ``0`` disables global pipelining; any positive value registers multiply
        outputs. Must be non-negative and cannot be combined with
        ``pipeline_points``.
    pipeline_points : list[str], optional
        Unique intermediate multiply names to register instead of enabling the
        global pipeline.

    """
    if not signed:
        raise NotImplementedError(
            "unsigned equation-to-Verilog emission is not supported; signed must be True"
        )
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
    escape_probability_verilog = core.escape_probability_verilog
    reset_expressions = core.reset_expressions
    total_pipeline_latency = core.total_pipeline_latency
    stochastic_threshold = bool(escape_probability_verilog)
    if stochastic_threshold and total_pipeline_latency > 0:
        raise NotImplementedError(
            "stochastic-threshold RTL does not yet support multiply pipelining; "
            "use pipeline_stages=0"
        )
    if stochastic_threshold:
        initial_seed = neuron.stochastic_rng_initial_seed
        if initial_seed is None:
            raise ValueError("stochastic-threshold RTL has no initial RNG seed")
        param_decls = [
            *param_decls,
            f"    parameter [15:0] RNG_SEED = 16'h{initial_seed:04x}",
        ]
    # Mirror the Python golden's edge/level decision exactly (see EquationNeuron): edge
    # logic is engaged only for a crossing, non-resetting model, so reset-based models use
    # the identical level datapath whether they declare ``level`` or ``crossing``.
    edge_detection = bool(threshold_verilog) and getattr(neuron, "_edge_detection", False)

    # ``substeps`` advances the datapath this many clocks (one integration sub-step each)
    # before a single macro-step spike decision, mirroring ``EquationNeuron.step``'s inner
    # sub-stepping for the conductance hand models (e.g. 100 dt sub-steps per 1 ms macro
    # step). The state advances every clock; the threshold crossing is evaluated only on the
    # macro boundary against the condition at the previous macro boundary, so a repetitively
    # firing oscillator emits one spike per action potential. Supported only for the edge
    # (crossing, non-resetting) datapath with no multiply pipelining — the case the
    # conductance oscillators need; other combinations raise rather than emit a datapath that
    # silently disagrees with the golden.
    substeps = int(getattr(neuron, "substeps", 1))
    if substeps > 1:
        if not edge_detection:
            raise NotImplementedError(
                "substeps > 1 is only supported for edge-crossing (non-resetting) models; "
                f"got detection={neuron.detection!r} with reset rules {bool(neuron.reset_rules)}"
            )
        if total_pipeline_latency > 0:
            raise NotImplementedError(
                "substeps > 1 is not supported with multiply pipelining (pipeline_stages > 0)"
            )
    substep_cnt_width = max(1, (substeps - 1).bit_length()) if substeps > 1 else 1

    lines = [
        "// SPDX-License-Identifier: AGPL-3.0-or-later",
        "// Commercial license available",
        "// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.",
        "// © Code 2020–2026 Miroslav Šotek. All rights reserved.",
        "// ORCID: 0009-0009-3560-0851",
        "// Contact: www.anulum.li | protoscience@anulum.li",
        "// SC-NeuroCore — Generated fixed-point RTL",
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

    if stochastic_threshold:
        lines.append("reg [15:0] _escape_lfsr;")
        lines.append("function [15:0] _escape_advance;")
        lines.append("    input [15:0] value;")
        lines.append("    begin")
        lines.append(
            "        _escape_advance = {value[0] ^ value[2] ^ value[3] ^ value[5], value[15:1]};"
        )
        lines.append("    end")
        lines.append("endfunction")
        previous_sample = "_escape_lfsr"
        for advance in range(1, 9):
            sample_name = f"_escape_sample_{advance}"
            lines.append(f"wire [15:0] {sample_name} = _escape_advance({previous_sample});")
            previous_sample = sample_name
        lines.append("wire [15:0] _escape_sample = _escape_sample_8;")

    if edge_detection:
        # 1-bit history of the threshold condition for rising-edge (``crossing``) detection.
        lines.append("reg _thr_prev;")

    if substeps > 1:
        # Sub-step counter (0 .. substeps-1). ``_macro_boundary`` is the clock that completes
        # a macro step, at which the crossing is evaluated and ``_thr_prev`` is refreshed.
        lines.append(f"reg [{substep_cnt_width - 1}:0] _ss_cnt;")
        lines.append(f"wire _macro_boundary = (_ss_cnt == {substep_cnt_width}'d{substeps - 1});")

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

    if stochastic_threshold:
        lines.extend(
            _escape_threshold_wires(
                escape_probability_verilog,
                "_escape_sample",
                data_width=data_width,
                fraction=fraction,
            )
        )
        lines.append("")

    if all_pipeline_regs:
        # Reset the staging registers to 0 so an unfilled pipeline never injects X into the
        # state feedback; the fill counter below guarantees the state only advances once these
        # stages have drained, so the reset value is never read as a live increment.
        lines.append("// Pipeline register stage — register multiply outputs (reset to 0)")
        lines.append("always @(posedge clk or negedge rst_n) begin")
        lines.append("    if (!rst_n) begin")
        for reg_decl in all_pipeline_regs:
            reg_name = reg_decl.split()[-1].rstrip(";")
            lines.append(f"        {reg_name} <= 0;")
        lines.append("    end else begin")
        for reg_decl in all_pipeline_regs:
            reg_name = reg_decl.split()[-1].rstrip(";")
            wire_name = reg_name[:-2] if reg_name.endswith("_r") else reg_name
            lines.append(f"        {reg_name} <= {wire_name};")
        lines.append("    end")
        lines.append("end")
        lines.append("")

    # Build the single-step body once (8-space base indent). When pipelined it is gated behind
    # the fill counter so the recurrence only advances after the register stages have drained.
    step_lines: list[str] = []
    if substeps > 1:
        # Macro-step datapath: the state advances one sub-step every clock, and the crossing is
        # taken only when ``_ss_cnt`` completes a macro window. ``_thr_prev`` holds the condition
        # at the previous macro boundary (not per sub-step), so a repetitively firing oscillator
        # emits one spike per action potential — bit-matching ``EquationNeuron.step``'s macro
        # crossing. Guarded above to the edge (crossing, non-resetting), non-pipelined case.
        step_lines.append(
            f"        _ss_cnt <= _macro_boundary ? {substep_cnt_width}'d0 : "
            f"(_ss_cnt + {substep_cnt_width}'d1);"
        )
        for var in neuron.equations:
            safe_var = state_var_map[var]
            step_lines.append(f"        {safe_var}_reg <= {safe_var}_next;")
            step_lines.append(f"        {safe_var}_out <= {safe_var}_next;")
        step_lines.append("        if (_macro_boundary) begin")
        step_lines.append(
            f"            if (({threshold_verilog}) && !_thr_prev) spike_out <= 1'b1;"
        )
        step_lines.append("            else spike_out <= 1'b0;")
        step_lines.append(f"            _thr_prev <= ({threshold_verilog});")
        step_lines.append("        end else begin")
        step_lines.append("            spike_out <= 1'b0;")
        step_lines.append("        end")
    elif threshold_verilog or stochastic_threshold:
        if stochastic_threshold:
            spike_cond = "_escape_spike"
            step_lines.append("        _escape_lfsr <= _escape_sample;")
        elif edge_detection:
            # Edge detection: spike only on the rising transition of the condition. ``_thr_prev``
            # holds the condition evaluated on the previously committed (next) state, so a
            # non-resetting oscillator fires exactly once per upward crossing — bit-matching the
            # Python golden's ``active and not prev`` edge test. A reset that clears the condition
            # makes this identical to level detection, so reset-based models are unaffected.
            spike_cond = f"({threshold_verilog}) && !_thr_prev"
        else:
            # Unchanged level datapath — leave the condition exactly as before (the
            # expression emitter already parenthesises it) so reset models are untouched.
            spike_cond = threshold_verilog
        step_lines.append(f"        if ({spike_cond}) begin")
        step_lines.append("            spike_out <= 1'b1;")
        for var in neuron.equations:
            safe_var = state_var_map[var]
            on_spike = reset_expressions.get(safe_var, f"{safe_var}_next")
            step_lines.append(f"            {safe_var}_reg <= {on_spike};")
            step_lines.append(f"            {safe_var}_out <= {on_spike};")
        step_lines.append("        end else begin")
        step_lines.append("            spike_out <= 1'b0;")
        for var in neuron.equations:
            safe_var = state_var_map[var]
            step_lines.append(f"            {safe_var}_reg <= {safe_var}_next;")
            step_lines.append(f"            {safe_var}_out <= {safe_var}_next;")
        step_lines.append("        end")
        if edge_detection:
            # Track the condition on the just-computed next state for the next step's edge test
            # (pre-reset, matching the golden). A comparison expression is already 1 bit wide.
            step_lines.append(f"        _thr_prev <= ({threshold_verilog});")
    else:
        step_lines.append("        spike_out <= 1'b0;")
        for var in neuron.equations:
            step_lines.append(f"        {var}_reg <= {var}_next;")
            step_lines.append(f"        {var}_out <= {var}_next;")

    # Fill counter: a self-recurrent step whose increment takes ``total_pipeline_latency``
    # register stages must hold the state steady for that many cycles, else the increment
    # applied would reflect stale (mid-fill) products and break bit-exactness with the golden.
    # ``total_pipeline_latency`` (the register count) upper-bounds the true pipeline depth, so
    # holding for that many cycles guarantees every stage has drained before the state advances
    # — one logical step every ``latency + 1`` clocks, spikes only on the valid cycle.
    if total_pipeline_latency > 0:
        cnt_w = max(1, total_pipeline_latency.bit_length())
        lines.append(f"reg [{cnt_w - 1}:0] _pl_cnt;")
        lines.append(f"wire _pl_valid = (_pl_cnt == {cnt_w}'d{total_pipeline_latency});")
        lines.append("")

    lines.append("always @(posedge clk or negedge rst_n) begin")
    lines.append("    if (!rst_n) begin")
    for var in neuron.equations:
        safe_var = state_var_map[var]
        init_val = q.encode_signed_literal(neuron.initial_state.get(var, 0.0))
        lines.append(f"        {safe_var}_reg <= {init_val};")
        lines.append(f"        {safe_var}_out <= {init_val};")
    lines.append("        spike_out <= 1'b0;")
    if stochastic_threshold:
        lines.append("        _escape_lfsr <= (RNG_SEED == 16'd0) ? 16'hace1 : RNG_SEED;")
    if edge_detection:
        # Seed the edge history from the initial state (bit-matching the Python golden's
        # ``initial_threshold_active``); an oscillator starting below threshold seeds 0.
        thr_prev_init = "1'b1" if neuron.initial_threshold_active() else "1'b0"
        lines.append(f"        _thr_prev <= {thr_prev_init};")
    if substeps > 1:
        lines.append(f"        _ss_cnt <= {substep_cnt_width}'d0;")
    if total_pipeline_latency > 0:
        lines.append(f"        _pl_cnt <= {cnt_w}'d0;")
    lines.append("    end else begin")

    if total_pipeline_latency > 0:
        lines.append(f"        _pl_cnt <= _pl_valid ? {cnt_w}'d0 : (_pl_cnt + {cnt_w}'d1);")
        lines.append("        if (_pl_valid) begin")
        for sline in step_lines:
            lines.append("    " + sline)
        lines.append("        end else begin")
        lines.append("            spike_out <= 1'b0;")
        lines.append("        end")
    else:
        lines.extend(step_lines)

    lines.append("    end")
    lines.append("end")
    lines.append("")
    lines.append("endmodule")

    return "\n".join(lines)
