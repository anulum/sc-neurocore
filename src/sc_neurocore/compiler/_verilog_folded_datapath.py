# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Folded equation-neuron datapath emission

"""Emit combinational processing elements for folded neuron populations."""

from __future__ import annotations

from collections.abc import Sequence

from ..hdl_gen._ident import sanitize_ident
from ..neurons.equation_builder import EquationNeuron
from ._verilog_neuron_core import _build_neuron_core, _escape_threshold_wires
from .verilog_compiler_config import Q88


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
    Escape-rate models additionally expose the already-advanced 16-bit ``rng_sample``
    input: the folded population owns one LFSR state per neuron in BRAM and supplies
    that sample on the same cycle as the corresponding membrane state.

    Pipelining is not supported here (a combinational PE has no register stages);
    the folded sequencer provides the one-cycle-per-neuron timing instead.

    Unsigned emission is rejected because the expression and derivative
    datapaths use signed fixed-point arithmetic. Product rounding accepts
    ``"truncate"``, ``"nearest"``, or ``"bankers"``; ``"stochastic"`` is
    rejected because the PE has no caller-owned product-rounding LFSR.
    """
    if not signed:
        raise NotImplementedError(
            "unsigned equation-to-Verilog emission is not supported; signed must be True"
        )
    if isinstance(param_ports, (str, bytes)):
        raise TypeError("param_ports must be a sequence of parameter-name strings, not text")
    normalised_param_ports = tuple(param_ports)
    if any(type(name) is not str for name in normalised_param_ports):
        raise TypeError("param_ports entries must all be strings")
    if len(normalised_param_ports) != len(set(normalised_param_ports)):
        raise ValueError("param_ports must not contain duplicates")
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
    stochastic_threshold = bool(core.escape_probability_verilog)
    # Partition parameters into those baked as module ``parameter`` defaults and those
    # carried on input ports (``param_ports``), so a folded population can stream
    # per-neuron parameters through the PE. Both keep the same ``P_<NAME>`` identifier,
    # so only the declaration differs — the arithmetic body is untouched.
    unknown = [p for p in normalised_param_ports if p not in core.param_map]
    if unknown:
        raise ValueError(f"param_ports names are not parameters of {neuron!r}: {sorted(unknown)}")
    port_vnames = [core.param_map[p] for p in normalised_param_ports]
    baked_decls = [
        decl for decl in core.param_decls if not any(f" {vn} =" in decl for vn in port_vnames)
    ]

    lines = [
        "// SPDX-License-Identifier: AGPL-3.0-or-later",
        "// Commercial license available",
        "// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.",
        "// © Code 2020–2026 Miroslav Šotek. All rights reserved.",
        "// ORCID: 0009-0009-3560-0851",
        "// Contact: www.anulum.li | protoscience@anulum.li",
        "// SC-NeuroCore — Generated folded fixed-point RTL",
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
    if stochastic_threshold:
        lines.append("    input wire [15:0] rng_sample,")
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

    if stochastic_threshold:
        lines.extend(
            _escape_threshold_wires(
                core.escape_probability_verilog,
                "rng_sample",
                data_width=data_width,
                fraction=fraction,
            )
        )
        lines.append("")

    # Spike is the combinational threshold over the candidate next state.
    if stochastic_threshold:
        lines.append("assign spike_out = _escape_spike;")
    elif core.threshold_verilog:
        lines.append(f"assign spike_out = ({core.threshold_verilog});")
    else:
        lines.append("assign spike_out = 1'b0;")

    # Next-state output: on spike, apply reset rules (or hold <var>_next when the
    # variable has no reset rule); otherwise advance to <var>_next. This mirrors
    # the per-instance always block exactly.
    for var in neuron.equations:
        safe_var = state_var_map[var]
        on_spike = core.reset_expressions.get(safe_var, f"{safe_var}_next")
        if core.threshold_verilog or stochastic_threshold:
            lines.append(
                f"assign {safe_var}_next_out = spike_out ? ({on_spike}) : {safe_var}_next;"
            )
        else:
            lines.append(f"assign {safe_var}_next_out = {safe_var}_next;")

    lines.append("")
    lines.append("endmodule")

    return "\n".join(lines)
