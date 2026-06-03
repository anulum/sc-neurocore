# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Adaptive runtime precision (dual-datapath) compiler

"""Generate Verilog with parallel low/high precision neuron datapaths.

Produces a dual-datapath module that runs a low-precision (LP) and
high-precision (HP) neuron in parallel.  The HP datapath is the authoritative
scientific output.  The LP datapath and hysteresis comparator expose a
``use_hp`` telemetry signal that can be used by downstream tooling to measure
where a future clock-enable or state-transfer design would need HP precision.

This module deliberately does not gate clocks or switch output state between
unsynchronised datapaths.  A runtime power-saving design needs an explicit,
verified state-transfer protocol before it can claim numerical equivalence.

Supports all Q-format combinations available in the precision presets:
Q1.7, Q4.4, Q8.8, Q4.12, Q1.15, Q9.9, Q12.12, Q14.13, Q20.12, Q16.16,
Q8.24, Q18.18, plus block-floating formats (for metadata and parameter
precision planning, e.g. BFP16E3X32), plus arbitrary custom
(data_width, fraction) pairs.

Usage::

    from sc_neurocore.compiler.adaptive_runtime_precision import (
        compile_adaptive_precision,
    )
    from sc_neurocore.neurons.equation_builder import from_equations

    neuron = from_equations(
        "dv/dt = -(v - E_L)/tau_m + I/C",
        threshold="v > -50", reset="v = -65",
        params=dict(E_L=-65, tau_m=10, C=1),
        init=dict(v=-65),
    )

    verilog = compile_adaptive_precision(
        neuron,
        module_name="sc_lif_adaptive",
        lp_width=16, lp_frac=8,    # Q8.8
        hp_width=32, hp_frac=16,   # Q16.16
    )
"""

from __future__ import annotations

import json
import math
from typing import Any

from ..hdl_gen._ident import sanitize_ident
from ..neurons.equation_builder import EquationNeuron
from .equation_compiler import Q88, compile_to_verilog
from .quantizer import BlockFloatingMode, QFormat, parse_precision_format

# All valid LP/HP pair presets — any (data_width, fraction) pair is valid,
# but these are the canonical presets from PRECISION_PRESETS
PRECISION_PAIRS: list[tuple[tuple[int, int], tuple[int, int]]] = [
    # (LP, HP) — LP must have strictly fewer bits than HP
    ((8, 7), (16, 8)),  # Q1.7 → Q8.8
    ((8, 4), (16, 8)),  # Q4.4 → Q8.8
    ((8, 7), (16, 12)),  # Q1.7 → Q4.12
    ((8, 4), (16, 12)),  # Q4.4 → Q4.12
    ((16, 8), (32, 16)),  # Q8.8 → Q16.16 (default)
    ((16, 8), (32, 12)),  # Q8.8 → Q20.12
    ((16, 8), (32, 24)),  # Q8.8 → Q8.24
    ((16, 12), (32, 16)),  # Q4.12 → Q16.16
    ((16, 12), (32, 24)),  # Q4.12 → Q8.24
    ((16, 15), (32, 16)),  # Q1.15 → Q16.16
    ((18, 9), (32, 16)),  # Q9.9 → Q16.16
    ((18, 9), (36, 18)),  # Q9.9 → Q18.18
    ((24, 12), (32, 16)),  # Q12.12 → Q16.16
    ((24, 12), (36, 18)),  # Q12.12 → Q18.18
    ((27, 13), (36, 18)),  # Q14.13 → Q18.18
]


def _precision_label(parsed: QFormat | BlockFloatingMode, *, source: str) -> str:
    """Return a deterministic textual label for telemetry."""
    if isinstance(parsed, QFormat):
        return f"Q{parsed.integer_bits}.{parsed.fraction_bits}"

    label = parsed.label
    if source.upper().endswith(f"X{parsed.block_size}"):
        return label
    return f"{label}X{parsed.block_size}"


def _precision_manifest(
    parsed: QFormat | BlockFloatingMode,
    source: str,
    resolved_width: int,
    emitted_fraction: int,
    *,
    kind: str,
) -> dict[str, Any]:
    """Build deterministic metadata for precision contracts."""
    if isinstance(parsed, QFormat):
        return {
            "kind": kind,
            "source": source,
            "label": f"Q{parsed.integer_bits}.{parsed.fraction_bits}",
            "data_width": resolved_width,
            "fraction": parsed.fraction_bits,
            "signed": True,
            "emitted_fraction": emitted_fraction,
        }

    metadata = parsed.metadata.copy()
    metadata.update(
        {
            "kind": kind,
            "source": source,
            "label": _precision_label(parsed, source=source),
            "data_width": resolved_width,
            "fraction": emitted_fraction,
            "signed": True,
            "emitted_fraction": emitted_fraction,
        }
    )
    return metadata


def _coerce_precision(
    precision: str | None,
    *,
    default_width: int,
    default_frac: int,
    tag: str,
) -> tuple[int, int, str, dict[str, Any], QFormat | BlockFloatingMode]:
    """Resolve concrete fixed-point datapath parameters and telemetry metadata."""
    if precision is None:
        q = Q88(data_width=default_width, fraction=default_frac, signed=True)
        return default_width, default_frac, q.q_label, _precision_manifest(
            q,
            source=f"{tag}:fallback",
            resolved_width=default_width,
            emitted_fraction=default_frac,
            kind="fixed",
        ), q

    parsed = parse_precision_format(precision)
    if isinstance(parsed, QFormat):
        width = parsed.total_bits
        fraction = parsed.fraction_bits
        return width, fraction, _precision_label(parsed, source=precision), _precision_manifest(
            parsed,
            source=precision,
            resolved_width=width,
            emitted_fraction=fraction,
            kind="fixed",
        ), parsed

    # Block-floating is emitted as mantissa-width fixed datapath with
    # deterministic block metadata for parity tooling.
    width = parsed.mantissa_bits
    fraction = parsed.emit_fraction
    return width, fraction, _precision_label(parsed, source=precision), _precision_manifest(
        parsed,
        source=precision,
        resolved_width=width,
        emitted_fraction=fraction,
        kind="block_floating",
    ), parsed


def _validate_lp_hp(lp_width: int, lp_frac: int, hp_width: int, hp_frac: int) -> None:
    """Validate that the LP/HP pair is sensible."""
    if lp_width >= hp_width:
        raise ValueError(
            f"LP data_width ({lp_width}) must be strictly less than HP data_width ({hp_width})"
        )
    if lp_frac < 1:
        raise ValueError(f"LP fraction ({lp_frac}) must be >= 1")
    if hp_frac < 1:
        raise ValueError(f"HP fraction ({hp_frac}) must be >= 1")
    if lp_width < 2:
        raise ValueError(f"LP data_width ({lp_width}) must be >= 2")


def _validate_hysteresis(
    threshold_up_pct: float,
    threshold_down_pct: float,
) -> None:
    """Validate adaptive-precision hysteresis thresholds.

    HP must engage above the lower threshold and release below the lower
    threshold with a strict separation. The thresholds are expected to satisfy:
    0 < threshold_down_pct < threshold_up_pct < 1.
    """
    if not math.isfinite(threshold_up_pct) or not math.isfinite(threshold_down_pct):
        raise ValueError("Threshold percentages must be finite")

    if not (0.0 < threshold_up_pct < 1.0):
        raise ValueError("threshold_up_pct must satisfy 0 < threshold_up_pct < 1")

    if not (0.0 < threshold_down_pct < threshold_up_pct):
        raise ValueError(
            "threshold_down_pct must satisfy 0 < threshold_down_pct < threshold_up_pct"
        )


def compile_adaptive_precision(
    neuron: EquationNeuron,
    module_name: str = "sc_adaptive_neuron",
    lp_width: int = 16,
    lp_frac: int = 8,
    hp_width: int = 32,
    hp_frac: int = 16,
    *,
    lp_precision: str | None = None,
    hp_precision: str | None = None,
    threshold_up_pct: float = 0.8,
    threshold_down_pct: float = 0.5,
    signed: bool = True,
    overflow: str = "saturate",
    rounding: str = "truncate",
) -> str:
    """Compile an EquationNeuron to dual-datapath adaptive-precision Verilog.

    Generates two complete neuron datapaths (LP and HP).  HP is emitted as
    the authoritative output path, while LP drives only precision telemetry.

    Parameters
    ----------
    neuron : EquationNeuron
        The neuron defined by arbitrary ODE strings.
    module_name : str
        Name of the generated Verilog module.
    lp_width : int
        Low-precision total bit width (default 16).
    lp_frac : int
        Low-precision fractional bits (default 8).
    hp_width : int
        High-precision total bit width (default 32).
    hp_frac : int
        High-precision fractional bits (default 16).
    lp_precision : str | None
        Optional LP precision string, e.g. ``Q8.8`` or ``BFP16E3X32``.
        If provided, ``lp_width`` and ``lp_frac`` are ignored for LP datapath.
    hp_precision : str | None
        Optional HP precision string, e.g. ``Q16.16`` or ``BFP20E4X32``.
        If provided, ``hp_width`` and ``hp_frac`` are ignored for HP datapath.
    threshold_up_pct : float
        Fraction of LP range at which to switch to HP (default 0.8).
    threshold_down_pct : float
        Fraction of LP range at which to switch back to LP (default 0.5).
    signed : bool
        True for signed two's complement.
    overflow : str
        Overflow mode for both datapaths.
    rounding : str
        Rounding mode for both datapaths.

    Returns
    -------
    str
        Synthesisable Verilog source with HP-authoritative dual datapaths.
    """
    lp_width, lp_frac, lp_q_label, lp_metadata, lp_precision_obj = _coerce_precision(
        lp_precision,
        default_width=lp_width,
        default_frac=lp_frac,
        tag="lp",
    )
    hp_width, hp_frac, hp_q_label, hp_metadata, hp_precision_obj = _coerce_precision(
        hp_precision,
        default_width=hp_width,
        default_frac=hp_frac,
        tag="hp",
    )

    _validate_lp_hp(lp_width, lp_frac, hp_width, hp_frac)
    _validate_hysteresis(threshold_up_pct=threshold_up_pct, threshold_down_pct=threshold_down_pct)

    # Generate both datapaths as inner modules
    lp_module = f"{module_name}_lp"
    hp_module = f"{module_name}_hp"

    lp_verilog = compile_to_verilog(
        neuron,
        module_name=lp_module,
        data_width=lp_width,
        fraction=lp_frac,
        signed=signed,
        overflow=overflow,
        rounding=rounding,
    )

    hp_verilog = compile_to_verilog(
        neuron,
        module_name=hp_module,
        data_width=hp_width,
        fraction=hp_frac,
        signed=signed,
        overflow=overflow,
        rounding=rounding,
    )

    # Compute hysteresis thresholds
    q_lp = Q88(data_width=lp_width, fraction=lp_frac, signed=signed)
    max_q = int(q_lp.max_value * (1 << lp_frac))
    thresh_up = int(threshold_up_pct * max_q)
    thresh_down = int(threshold_down_pct * max_q)

    # Build the top-level wrapper
    safe_name = sanitize_ident(module_name, context="module name")
    state_vars = list(neuron.equations.keys())
    primary_var = state_vars[0]  # typically 'v'
    safe_primary = sanitize_ident(primary_var, context="state variable")

    lines: list[str] = []

    # Emit LP and HP sub-modules first
    lines.append("// " + "=" * 63)
    lines.append(f"// Low-Precision Datapath ({lp_q_label}, {lp_width}-bit)")
    lines.append("// " + "=" * 63)
    lines.append("")
    lines.append(lp_verilog)
    lines.append("")
    lines.append("// " + "=" * 63)
    lines.append(f"// High-Precision Datapath ({hp_q_label}, {hp_width}-bit)")
    lines.append("// " + "=" * 63)
    lines.append("")
    lines.append(hp_verilog)
    lines.append("")

    # Top-level wrapper
    lines.append("// " + "=" * 63)
    lines.append("// Adaptive Precision Wrapper — HP-authoritative telemetry")
    lines.append("// " + "=" * 63)
    lines.append("")
    lines.append("// Auto-generated by SC-NeuroCore adaptive precision compiler")
    lines.append(f"// LP: {lp_q_label} ({lp_width}-bit), HP: {hp_q_label} ({hp_width}-bit)")
    lines.append(
        f"// Hysteresis: switch-up at {threshold_up_pct * 100:.0f}% "
        f"of LP range, switch-down at {threshold_down_pct * 100:.0f}%"
    )
    lines.append(
        "// Meta: "
        f"lp_precision={lp_precision_obj.__class__.__name__}, "
        f"hp_precision={hp_precision_obj.__class__.__name__}"
    )
    lines.append(
        "// SC-NeuroCore Adaptive Precision Manifest: "
        + json.dumps(
            {
                "schema_version": "1.0",
                "kind": "adaptive_precision_v1",
                "module_name": safe_name,
                "primary_variable": primary_var,
                "threshold_up_pct": threshold_up_pct,
                "threshold_down_pct": threshold_down_pct,
                "signed": signed,
                "overflow": overflow,
                "rounding": rounding,
                "lp_precision": lp_metadata,
                "hp_precision": hp_metadata,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    lines.append("`timescale 1ns / 1ps")
    lines.append("")
    lines.append(f"module {safe_name} (")
    lines.append("    input wire clk,")
    lines.append("    input wire rst_n,")
    lines.append(f"    input wire signed [{hp_width - 1}:0] I_t,")
    lines.append("    output reg spike_out,")

    for var in state_vars:
        safe_var = sanitize_ident(var, context="state variable")
        lines.append(f"    output reg signed [{hp_width - 1}:0] {safe_var}_out,")

    lines.append("    output wire use_hp")
    lines.append(");")
    lines.append("")

    # Internal wires — LP datapath
    lines.append("// LP datapath signals")
    lines.append(f"wire signed [{lp_width - 1}:0] lp_I_t = I_t[{lp_width - 1}:0];")
    lines.append("wire lp_spike;")
    for var in state_vars:
        safe_var = sanitize_ident(var, context="state variable")
        lines.append(f"wire signed [{lp_width - 1}:0] lp_{safe_var}_out;")
    lines.append("")

    # Internal wires — HP datapath
    lines.append("// HP datapath signals")
    lines.append(f"wire signed [{hp_width - 1}:0] hp_I_t = I_t;")
    lines.append("wire hp_spike;")
    for var in state_vars:
        safe_var = sanitize_ident(var, context="state variable")
        lines.append(f"wire signed [{hp_width - 1}:0] hp_{safe_var}_out;")
    lines.append("")

    # Hysteresis precision switch logic
    lines.append("// Hysteresis precision telemetry")
    lines.append(
        f"// use_hp asserts when |{primary_var}| exceeds {threshold_up_pct * 100:.0f}% of LP range"
    )
    lines.append(
        f"// use_hp clears when |{primary_var}| falls below "
        f"{threshold_down_pct * 100:.0f}% of LP range"
    )
    lines.append(f"localparam signed [{lp_width - 1}:0] THRESH_UP = {lp_width}'sd{thresh_up};")
    lines.append(f"localparam signed [{lp_width - 1}:0] THRESH_DOWN = {lp_width}'sd{thresh_down};")
    lines.append("")
    lines.append("reg precision_mode;  // telemetry only; HP remains authoritative")
    lines.append("assign use_hp = precision_mode;")
    lines.append("")

    # Precision switch always block
    lines.append("always @(posedge clk or negedge rst_n) begin")
    lines.append("    if (!rst_n) begin")
    lines.append("        precision_mode <= 1'b0;")
    lines.append("    end else begin")
    lines.append(
        f"        if (!precision_mode && "
        f"($signed(lp_{safe_primary}_out) > THRESH_UP || "
        f"$signed(lp_{safe_primary}_out) < (-THRESH_UP))) begin"
    )
    lines.append("            precision_mode <= 1'b1;")
    lines.append("        end")
    lines.append(
        f"        else if (precision_mode && "
        f"($signed(lp_{safe_primary}_out) < THRESH_DOWN && "
        f"$signed(lp_{safe_primary}_out) > (-THRESH_DOWN))) begin"
    )
    lines.append("            precision_mode <= 1'b0;")
    lines.append("        end")
    lines.append("    end")
    lines.append("end")
    lines.append("")

    # LP datapath instance
    lp_safe = sanitize_ident(lp_module, context="module name")
    lines.append(f"// LP datapath instantiation ({lp_q_label})")
    lines.append(f"{lp_safe} lp_inst (")
    lines.append("    .clk(clk),")
    lines.append("    .rst_n(rst_n),")
    lines.append("    .I_t(lp_I_t),")
    lines.append("    .spike_out(lp_spike),")
    for var in state_vars:
        safe_var = sanitize_ident(var, context="state variable")
        lines.append(f"    .{safe_var}_out(lp_{safe_var}_out),")
    lines[-1] = lines[-1].rstrip(",")
    lines.append(");")
    lines.append("")

    # HP datapath instance
    hp_safe = sanitize_ident(hp_module, context="module name")
    lines.append(f"// HP datapath instantiation ({hp_q_label}, authoritative)")
    lines.append(f"{hp_safe} hp_inst (")
    lines.append("    .clk(clk),")
    lines.append("    .rst_n(rst_n),")
    lines.append("    .I_t(hp_I_t),")
    lines.append("    .spike_out(hp_spike),")
    for var in state_vars:
        safe_var = sanitize_ident(var, context="state variable")
        lines.append(f"    .{safe_var}_out(hp_{safe_var}_out),")
    lines[-1] = lines[-1].rstrip(",")
    lines.append(");")
    lines.append("")

    # Output register — HP authoritative
    lines.append("// Output register — HP datapath is authoritative")
    lines.append("always @(posedge clk or negedge rst_n) begin")
    lines.append("    if (!rst_n) begin")
    lines.append("        spike_out <= 1'b0;")

    for var in state_vars:
        safe_var = sanitize_ident(var, context="state variable")
        init_val = int(round(neuron.initial_state.get(var, 0.0) * (1 << hp_frac)))
        lines.append(f"        {safe_var}_out <= {hp_width}'sd{init_val};")

    lines.append("    end else begin")
    lines.append("        spike_out <= hp_spike;")
    for var in state_vars:
        safe_var = sanitize_ident(var, context="state variable")
        lines.append(f"            {safe_var}_out <= hp_{safe_var}_out;")
    lines.append("    end")
    lines.append("end")
    lines.append("")
    lines.append("endmodule")

    return "\n".join(lines)
