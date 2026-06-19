# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Auto-quantisation sweep

"""Auto-quantisation design-space exploration utilities.

Sweeps multiple data widths and fractional precisions to find the
optimal accuracy-vs-resource trade-off for a given ODE system.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QuantSweepResult:
    """Result of a quantisation sweep for one (width, fraction) pair.

    Attributes
    ----------
    data_width : int
        Total bit width tested.
    fraction : int
        Fractional bits tested.
    guard_bits : int
        Guard bits required.
    estimated_luts : int
        Estimated LUT usage.
    estimated_dsps : int
        Estimated DSP usage.
    estimated_ffs : int
        Estimated flip-flop usage.
    max_representable : float
        Maximum representable value.
    min_step : float
        Minimum step size (LSB resolution).
    """

    data_width: int
    fraction: int
    guard_bits: int
    estimated_luts: int
    estimated_dsps: int
    estimated_ffs: int
    max_representable: float
    min_step: float


def auto_quantisation_sweep(
    equations: dict[str, str],
    target: str = "artix7",
    *,
    widths: list[int] | None = None,
    fraction_ratio: float = 0.5,
) -> list[QuantSweepResult]:
    """Sweep data widths to find accuracy-vs-resource trade-offs.

    Compiles the same ODE equations at multiple quantisation levels
    (Q4.2 through Q32.16) and reports the resource cost and numerical
    precision for each. Enables rapid design-space exploration.

    Parameters
    ----------
    equations : dict[str, str]
        ODE equations mapping state variable names to expressions.
    target : str
        Target platform name for resource estimation.
    widths : list[int], optional
        Data widths to sweep. Defaults to ``[4, 8, 12, 16, 20, 24, 32]``.
    fraction_ratio : float
        Fraction of data_width used for fractional bits (default 0.5).

    Returns
    -------
    list[QuantSweepResult]
        Sweep results sorted by data_width (ascending).
    """
    from ..platforms import get_profile
    from ..static_analysis import compute_guard_bits

    if widths is None:
        widths = [4, 8, 12, 16, 20, 24, 32]

    profile = get_profile(target)
    has_dsp = bool(profile.dsp_block)

    results = []
    for dw in sorted(widths):
        frac = max(1, int(dw * fraction_ratio))

        # Guard bits from expression analysis
        guard = 0
        for _sv, expr in equations.items():
            g = compute_guard_bits(expr)
            guard = max(guard, g)

        # Count multiplies and adds in expressions
        mul_count = 0
        add_count = 0
        for _sv, expr in equations.items():
            mul_count += expr.count("*")
            add_count += expr.count("+") + expr.count("-")

        # Resource estimation heuristics
        luts_per_add = dw
        luts_per_mul = 0 if has_dsp else (dw * dw // 4)
        luts = add_count * luts_per_add + mul_count * luts_per_mul
        dsps = mul_count if has_dsp else 0
        ffs = len(equations) * dw

        # Numerical range
        int_bits = dw - frac - 1  # sign bit
        max_repr = (2.0**int_bits) - (2.0 ** (-frac))
        min_step = 2.0 ** (-frac)

        results.append(
            QuantSweepResult(
                data_width=dw,
                fraction=frac,
                guard_bits=guard,
                estimated_luts=luts,
                estimated_dsps=dsps,
                estimated_ffs=ffs,
                max_representable=max_repr,
                min_step=min_step,
            )
        )

    return results


def format_quantisation_report(results: list[QuantSweepResult]) -> str:
    """Format a quantisation sweep into a readable markdown table.

    Parameters
    ----------
    results : list[QuantSweepResult]
        Results from ``auto_quantisation_sweep()``.

    Returns
    -------
    str
        Markdown table comparing all quantisation levels.
    """
    lines = [
        "# SC-NeuroCore Quantisation Sweep Report",
        "",
        "| Width | Frac | Q-format | Guard | LUTs | DSPs | FFs | Max Value | LSB Step |",
        "|------:|-----:|----------|------:|-----:|-----:|----:|---------:|--------:|",
    ]
    for r in results:
        qfmt = f"Q{r.data_width - r.fraction}.{r.fraction}"
        lines.append(
            f"| {r.data_width:5d} | {r.fraction:4d} | {qfmt:8s} "
            f"| {r.guard_bits:5d} | {r.estimated_luts:4d} | {r.estimated_dsps:4d} "
            f"| {r.estimated_ffs:3d} | {r.max_representable:9.4f} "
            f"| {r.min_step:.2e} |"
        )
    return "\n".join(lines)
