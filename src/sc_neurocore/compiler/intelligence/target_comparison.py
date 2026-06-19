# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Target comparison report

"""Compilation comparison across multiple hardware targets."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TargetComparison:
    """Compilation comparison for one target.

    Attributes
    ----------
    target : str
        Platform name.
    data_width : int
        Selected data width.
    fraction : int
        Fractional bits.
    overflow : str
        Overflow mode.
    dsp_block : str
        DSP block type.
    max_freq_mhz : int | None
        Maximum frequency.
    estimated_luts : int
        Estimated LUT usage.
    estimated_dsps : int
        Estimated DSP usage.
    pipeline_stages : int
        Required pipeline stages.
    critical_path_depth : int
        DSP chain depth.
    """

    target: str
    data_width: int
    fraction: int
    overflow: str
    dsp_block: str
    max_freq_mhz: int | None
    estimated_luts: int
    estimated_dsps: int
    pipeline_stages: int
    critical_path_depth: int


def compare_targets(
    equations: dict[str, str],
    targets: list[str],
) -> list[TargetComparison]:
    """Compare compilation results across multiple hardware targets."""
    from ..platforms import get_profile
    from ..static_analysis import (
        critical_path_depth as cpd,
    )
    from ..static_analysis import (
        pipeline_stages_needed,
    )

    # Compute shared depth
    max_depth = 0
    mul_count = 0
    add_count = 0
    for _sv, expr in equations.items():
        max_depth = max(max_depth, cpd(expr))
        mul_count += expr.count("*")
        add_count += expr.count("+") + expr.count("-")

    results = []
    for tgt in targets:
        profile = get_profile(tgt)
        dw = profile.data_width
        frac = profile.fraction
        has_dsp = bool(profile.dsp_block)
        freq = profile.max_freq_mhz or 100

        # Resource estimation
        luts_per_add = dw
        luts_per_mul = 0 if has_dsp else (dw * dw // 4)
        luts = add_count * luts_per_add + mul_count * luts_per_mul
        dsps = mul_count if has_dsp else 0

        stages = pipeline_stages_needed(max_depth, freq)

        results.append(
            TargetComparison(
                target=tgt,
                data_width=dw,
                fraction=frac,
                overflow=profile.overflow,
                dsp_block=profile.dsp_block,
                max_freq_mhz=profile.max_freq_mhz,
                estimated_luts=luts,
                estimated_dsps=dsps,
                pipeline_stages=stages,
                critical_path_depth=max_depth,
            )
        )

    return results


def format_comparison_report(results: list[TargetComparison]) -> str:
    """Format a multi-target comparison as a markdown table."""
    lines = [
        "# SC-NeuroCore Multi-Target Comparison Report",
        "",
        "| Target | Width | Frac | Overflow | DSP | Freq (MHz) | LUTs | DSPs | Pipeline | Depth |",
        "|--------|------:|-----:|----------|-----|----------:|-----:|-----:|---------:|------:|",
    ]
    for r in results:
        freq_str = str(r.max_freq_mhz) if r.max_freq_mhz else "N/A"
        lines.append(
            f"| {r.target:20s} | {r.data_width:5d} | {r.fraction:4d} "
            f"| {r.overflow:8s} | {r.dsp_block or 'N/A':3s} | {freq_str:>10s} "
            f"| {r.estimated_luts:4d} | {r.estimated_dsps:4d} "
            f"| {r.pipeline_stages:8d} | {r.critical_path_depth:5d} |"
        )
    return "\n".join(lines)
