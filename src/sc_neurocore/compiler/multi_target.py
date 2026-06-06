# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Multi-target compilation comparison

"""Multi-target compilation utilities.

Compiles a neuron to multiple hardware profiles simultaneously and
generates comparison reports.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CompilationResult:
    """Per-target compilation result for comparison.

    Attributes
    ----------
    target : str
        Profile name.
    verilog_lines : int
        Lines of generated Verilog.
    data_width : int
        Total bit width.
    fraction : int
        Fractional bits.
    overflow : str
        Overflow mode.
    rounding : str
        Rounding mode.
    estimated_luts : int
        LUT estimate.
    estimated_dsps : int
        DSP block estimate.
    estimated_ffs : int
        Flip-flop estimate.
    guard_bits : int
        Required guard bits.
    max_freq_mhz : int | None
        Target max frequency, or None when unknown.
    """

    target: str
    verilog_lines: int
    data_width: int
    fraction: int
    overflow: str
    rounding: str
    estimated_luts: int
    estimated_dsps: int
    estimated_ffs: int
    guard_bits: int
    max_freq_mhz: int | None


def compile_multi_target(
    equations: dict[str, str],
    targets: list[str],
    module_name: str = "sc_neuron",
) -> list[CompilationResult]:
    """Compile a neuron to multiple targets and collect metrics.

    Parameters
    ----------
    equations : dict
        Variable name → ODE RHS expression.
    targets : list[str]
        Profile names to compile against.
    module_name : str
        Base module name.

    Returns
    -------
    list[CompilationResult]
        Per-target compilation results.
    """
    from sc_neurocore.compiler.platforms import get_profile
    from sc_neurocore.compiler.static_analysis import compute_guard_bits

    results = []
    for target_name in targets:
        profile = get_profile(target_name)

        # Estimate resources heuristically (no actual Verilog gen required)
        total_mul = 0
        total_add = 0
        max_guard = 0
        for expr in equations.values():
            # Count muls/adds from expression
            import ast

            tree = ast.parse(expr, mode="eval")
            for node in ast.walk(tree):
                if isinstance(node, ast.BinOp):
                    if isinstance(node.op, (ast.Mult, ast.Div)):
                        total_mul += 1
                    elif isinstance(node.op, (ast.Add, ast.Sub)):
                        total_add += 1
            g = compute_guard_bits(expr)
            max_guard = max(max_guard, g)

        dw = profile.data_width
        luts = total_add * dw + (total_mul * dw * dw // 4 if not profile.dsp_block else 0)
        dsps = total_mul if profile.dsp_block else 0
        ffs = len(equations) * dw + dw  # state regs + control
        verilog_lines = 30 + len(equations) * 15 + total_mul * 5

        results.append(
            CompilationResult(
                target=target_name,
                verilog_lines=verilog_lines,
                data_width=dw,
                fraction=profile.fraction,
                overflow=profile.overflow,
                rounding=profile.rounding,
                estimated_luts=max(luts, 1),
                estimated_dsps=dsps,
                estimated_ffs=max(ffs, 1),
                guard_bits=max_guard,
                max_freq_mhz=profile.max_freq_mhz,
            )
        )

    return results


def format_comparison_table(results: list[CompilationResult]) -> str:
    """Format multi-target results as a markdown comparison table.

    Parameters
    ----------
    results : list[CompilationResult]
        Per-target compilation results.

    Returns
    -------
    str
        Markdown table string.
    """
    lines = [
        "| Target | Bits | Frac | DSPs | LUTs | FFs | Fmax | Guard | Overflow | Rounding |",
        "|--------|-----:|-----:|-----:|-----:|----:|-----:|------:|----------|----------|",
    ]
    for r in results:
        freq = f"{r.max_freq_mhz}" if r.max_freq_mhz else "N/A"
        lines.append(
            f"| {r.target:16s} | {r.data_width:4d} | {r.fraction:4d} | "
            f"{r.estimated_dsps:4d} | {r.estimated_luts:4d} | {r.estimated_ffs:3d} | "
            f"{freq:>4s} | {r.guard_bits:5d} | {r.overflow:8s} | {r.rounding:8s} |"
        )
    return "\n".join(lines)
