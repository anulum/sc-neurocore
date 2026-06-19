# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Compilation summary report

"""Generate a comprehensive human-readable compilation summary."""

from __future__ import annotations


def generate_compilation_summary(
    module_name: str,
    equations: dict[str, str],
    target: str,
    *,
    data_width: int = 16,
    fraction: int = 8,
    verilog_lines: int = 0,
) -> str:
    """Generate a comprehensive human-readable compilation summary.

    Produces a markdown document summarising all aspects of a compilation.

    Parameters
    ----------
    module_name : str
        Compiled module name.
    equations : dict[str, str]
        ODE equations compiled.
    target : str
        Target platform.
    data_width : int
        Total bit width.
    fraction : int
        Fractional bits.
    verilog_lines : int
        Lines of generated Verilog.

    Returns
    -------
    str
        Markdown compilation summary.
    """
    from ..platforms import get_profile
    from ..static_analysis import (
        compute_guard_bits,
        pipeline_stages_needed,
    )
    from ..static_analysis import (
        critical_path_depth as cpd,
    )

    profile = get_profile(target)
    freq = profile.max_freq_mhz or 100
    int_bits = data_width - fraction - 1

    # Compute metrics
    max_depth = 0
    max_guard = 0
    mul_count = 0
    add_count = 0
    for _sv, expr in equations.items():
        max_depth = max(max_depth, cpd(expr))
        max_guard = max(max_guard, compute_guard_bits(expr))
        mul_count += expr.count("*")
        add_count += expr.count("+") + expr.count("-")

    stages = pipeline_stages_needed(max_depth, freq)
    has_dsp = bool(profile.dsp_block)
    luts = add_count * data_width + (0 if has_dsp else mul_count * data_width * data_width // 4)
    dsps = mul_count if has_dsp else 0
    ffs = len(equations) * data_width

    lines = [
        "# SC-NeuroCore Compilation Summary",
        "",
        f"## Module: `{module_name}`",
        "",
        "### Equations",
        "",
    ]
    for sv, expr in equations.items():
        lines.append(f"- `{sv}' = {expr}`")

    lines.extend(
        [
            "",
            "### Target Platform",
            "",
            "| Property | Value |",
            "|----------|-------|",
            f"| Platform | {profile.name} |",
            f"| Vendor | {profile.vendor} |",
            f"| Family | {profile.family} |",
            f"| Class | {profile.platform_class} |",
            f"| Max Frequency | {freq} MHz |",
            f"| DSP Block | {profile.dsp_block or 'None'} |",
            "",
            "### Fixed-Point Configuration",
            "",
            "| Property | Value |",
            "|----------|-------|",
            f"| Format | Q{int_bits + 1}.{fraction} |",
            f"| Data Width | {data_width} bits |",
            f"| Integer Bits | {int_bits + 1} (incl. sign) |",
            f"| Fractional Bits | {fraction} |",
            f"| Overflow | {profile.overflow} |",
            f"| Rounding | {profile.rounding} |",
            f"| Guard Bits | {max_guard} |",
            f"| Max Representable | {(2.0**int_bits) - (2.0 ** (-fraction)):.4f} |",
            f"| LSB Resolution | {2.0 ** (-fraction):.2e} |",
            "",
            "### Resource Estimation",
            "",
            "| Resource | Count |",
            "|----------|------:|",
            f"| LUTs | {luts} |",
            f"| DSPs | {dsps} |",
            f"| Flip-Flops | {ffs} |",
            f"| Multiplies | {mul_count} |",
            f"| Adds/Subs | {add_count} |",
            "",
            "### Pipeline Analysis",
            "",
            "| Property | Value |",
            "|----------|------:|",
            f"| Critical Path Depth | {max_depth} DSP blocks |",
            f"| Pipeline Stages | {stages} |",
            f"| Total Latency | {stages + 1} clock cycles |",
            "",
        ]
    )

    if verilog_lines > 0:
        lines.extend(
            [
                "### Output",
                "",
                f"- Verilog: {verilog_lines} lines",
                "",
            ]
        )

    # Applicable features
    features = []
    if profile.platform_class == "photonic":
        features.append("MZI weight encoding (`encode_mzi_weights`)")
    if profile.platform_class == "in_memory":
        features.append("PIM layout planner (`plan_pim_layout`)")
    if profile.platform_class in ("fpga",):
        features.append("TMR wrapper (`generate_tmr_wrapper`)")
        features.append("Bitstream encryption (`generate_bitstream_encryption`)")
    if profile.platform_class == "neuromorphic":
        features.append("On-chip learning (`generate_learning_params`)")
    features.append("Model checksum (`embed_model_checksum`)")
    features.append("Quantisation sweep (`auto_quantisation_sweep`)")
    features.append("HLS-C++ export (`generate_hls_cpp`)")

    lines.extend(
        [
            "### Applicable Features",
            "",
        ]
    )
    for feat in features:
        lines.append(f"- {feat}")

    lines.extend(
        [
            "",
            "---",
            "*Generated by SC-NeuroCore Universal Neuromorphic Compiler*",
        ]
    )

    return "\n".join(lines)
