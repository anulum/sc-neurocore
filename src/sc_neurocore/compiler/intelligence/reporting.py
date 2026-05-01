# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore

from __future__ import annotations
import hashlib
import json
from dataclasses import dataclass
from .security_and_compliance import estimate_carbon_footprint
from .verification_and_safety import predict_reliability

# 24. Multi-Target Comparison Report
# ═══════════════════════════════════════════════════════════════════════


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
    """Compare compilation results across multiple hardware targets.

    Compiles the same ODE equations for each target and reports
    resource usage, precision, and pipeline requirements side-by-side.

    Parameters
    ----------
    equations : dict[str, str]
        ODE equations.
    targets : list[str]
        List of target platform names.

    Returns
    -------
    list[TargetComparison]
        Comparison results for each target.
    """
    from ..platforms import get_profile
    from ..static_analysis import critical_path_depth as cpd
    from ..static_analysis import pipeline_stages_needed

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
    """Format a multi-target comparison as a markdown table.

    Parameters
    ----------
    results : list[TargetComparison]
        Results from ``compare_targets()``.

    Returns
    -------
    str
        Markdown comparison table.
    """
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


# ═══════════════════════════════════════════════════════════════════════
# 25. Compilation Summary Report
# ═══════════════════════════════════════════════════════════════════════


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

    Produces a markdown document summarising all aspects of a
    compilation: equations, target, precision, resources, pipeline,
    guard bits, and applicable strategic features.

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
        Lines of generated Verilog (0 if not counted).

    Returns
    -------
    str
        Markdown compilation summary.
    """
    from ..platforms import get_profile
    from ..static_analysis import (
        compute_guard_bits,
        critical_path_depth as cpd,
        pipeline_stages_needed,
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


# ═══════════════════════════════════════════════════════════════════════
# 28. Provenance Chain
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class ProvenanceRecord:
    """Cryptographic audit trail entry.

    Attributes
    ----------
    stage : str
        Pipeline stage name.
    input_hash : str
        SHA-256 of input artefact.
    output_hash : str
        SHA-256 of output artefact.
    timestamp : str
        ISO 8601 timestamp.
    parameters : dict
        Compilation parameters used.
    """

    stage: str
    input_hash: str
    output_hash: str
    timestamp: str
    parameters: dict


def generate_provenance_chain(
    module_name: str,
    equations: dict[str, str],
    verilog_source: str = "",
    *,
    target: str = "artix7",
    data_width: int = 16,
    fraction: int = 8,
) -> list[ProvenanceRecord]:
    """Generate a cryptographic provenance chain for compilation.

    Creates a full audit trail from source equations through
    compiled RTL, with SHA-256 hashes at every stage.

    Parameters
    ----------
    module_name : str
        Module name.
    equations : dict[str, str]
        Source ODE equations.
    verilog_source : str
        Generated Verilog (if available).
    target : str
        Target platform.
    data_width : int
        Fixed-point width.
    fraction : int
        Fractional bits.

    Returns
    -------
    list[ProvenanceRecord]
        Ordered provenance records.
    """
    import json
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc).isoformat()

    # Stage 1: Source equations
    eq_str = json.dumps(equations, sort_keys=True)
    eq_hash = hashlib.sha256(eq_str.encode()).hexdigest()

    # Stage 2: Compilation parameters
    params = {
        "module_name": module_name,
        "target": target,
        "data_width": data_width,
        "fraction": fraction,
    }
    params_str = json.dumps(params, sort_keys=True)
    params_hash = hashlib.sha256(params_str.encode()).hexdigest()

    # Stage 3: Verilog output
    v_hash = hashlib.sha256(verilog_source.encode()).hexdigest()

    chain = [
        ProvenanceRecord(
            stage="source_equations",
            input_hash="genesis",
            output_hash=eq_hash,
            timestamp=now,
            parameters={"equation_count": len(equations)},
        ),
        ProvenanceRecord(
            stage="compilation_config",
            input_hash=eq_hash,
            output_hash=params_hash,
            timestamp=now,
            parameters=params,
        ),
        ProvenanceRecord(
            stage="verilog_generation",
            input_hash=params_hash,
            output_hash=v_hash,
            timestamp=now,
            parameters={"verilog_lines": verilog_source.count("\n") + 1},
        ),
    ]

    return chain


def format_provenance_json(chain: list[ProvenanceRecord]) -> str:
    """Format provenance chain as JSON manifest.

    Parameters
    ----------
    chain : list[ProvenanceRecord]
        From ``generate_provenance_chain()``.

    Returns
    -------
    str
        JSON manifest.
    """

    data = {
        "sc_neurocore_provenance": {
            "version": "1.0",
            "chain": [
                {
                    "stage": r.stage,
                    "input_hash": r.input_hash,
                    "output_hash": r.output_hash,
                    "timestamp": r.timestamp,
                    "parameters": r.parameters,
                }
                for r in chain
            ],
        }
    }
    return json.dumps(data, indent=2)


# ═══════════════════════════════════════════════════════════════════════
# 38. Model Complexity Classifier
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class ModelComplexity:
    """Model compute-profile classification.

    Attributes
    ----------
    classification : str
        ``"compute_bound"``, ``"memory_bound"``, or ``"comm_bound"``.
    compute_ops : int
        Total arithmetic operations.
    memory_vars : int
        State variables (memory footprint proxy).
    comm_ratio : float
        Inter-variable coupling ratio.
    recommended_paradigm : str
        Best platform class.
    """

    classification: str
    compute_ops: int
    memory_vars: int
    comm_ratio: float
    recommended_paradigm: str


def classify_model_complexity(
    equations: dict[str, str],
) -> ModelComplexity:
    """Classify a model's compute profile.

    Determines whether the model is compute-bound, memory-bound,
    or communication-bound and recommends the best platform paradigm.

    Parameters
    ----------
    equations : dict[str, str]
        ODE equations.

    Returns
    -------
    ModelComplexity
        Classification with recommended paradigm.
    """
    num_vars = len(equations)
    total_ops = sum(
        e.count("+") + e.count("-") + e.count("*") + e.count("/") for e in equations.values()
    )

    # Communication: count cross-variable references
    cross_refs = 0
    for sv, expr in equations.items():
        for other_sv in equations:
            if other_sv != sv and other_sv in expr:
                cross_refs += 1

    comm_ratio = cross_refs / max(1, num_vars)

    if total_ops / max(1, num_vars) > 4:
        cls = "compute_bound"
        paradigm = "fpga"
    elif num_vars > 4 and total_ops / max(1, num_vars) <= 2:
        cls = "memory_bound"
        paradigm = "in_memory"
    elif comm_ratio > 1.5:
        cls = "comm_bound"
        paradigm = "cgra"
    else:
        cls = "compute_bound"
        paradigm = "fpga"

    return ModelComplexity(
        classification=cls,
        compute_ops=total_ops,
        memory_vars=num_vars,
        comm_ratio=round(comm_ratio, 2),
        recommended_paradigm=paradigm,
    )


# ═══════════════════════════════════════════════════════════════════════
# 48. Model Portability Scorer
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class PortabilityScore:
    """Cross-platform portability assessment.

    Attributes
    ----------
    score : float
        Portability score 0-100.
    compatible_profiles : int
        Number of compatible profiles.
    total_profiles : int
        Total profiles checked.
    blockers : list[str]
        Portability blockers.
    """

    score: float
    compatible_profiles: int
    total_profiles: int
    blockers: list[str]


def score_portability(
    equations: dict[str, str],
    *,
    min_data_width: int = 8,
) -> PortabilityScore:
    """Score how portable a model is across all profiles.

    Parameters
    ----------
    equations : dict[str, str]
        ODE equations.
    min_data_width : int
        Minimum acceptable data width.

    Returns
    -------
    PortabilityScore
        Portability assessment.
    """
    from sc_neurocore.compiler.platforms import (
        list_profile_names,
        get_profile,
    )

    total_ops = sum(e.count("*") + e.count("/") for e in equations.values())
    names = list_profile_names()
    compatible = 0
    blockers = []

    for n in names:
        p = get_profile(n)
        if p.data_width < min_data_width:
            continue
        if (
            total_ops > 3
            and not p.dsp_block
            and p.platform_class
            not in (
                "simulation",
                "biological",
                "dna_molecular",
            )
        ):
            continue
        compatible += 1

    if total_ops > 5:
        blockers.append("High arithmetic complexity limits low-width targets")
    if len(equations) > 4:
        blockers.append("Many state variables require large register files")

    pct = (compatible / len(names)) * 100 if names else 0
    return PortabilityScore(
        score=round(pct, 1),
        compatible_profiles=compatible,
        total_profiles=len(names),
        blockers=blockers,
    )


# ═══════════════════════════════════════════════════════════════════════
# 59. Compilation Report Generator
# ═══════════════════════════════════════════════════════════════════════


def generate_compilation_report(
    module_name: str,
    equations: dict[str, str],
    profile_name: str,
    *,
    include_carbon: bool = True,
    include_reliability: bool = True,
) -> str:
    """Generate comprehensive compilation report.

    Consolidates Verilog, timing, power, carbon, risk,
    and reliability into a single markdown document.

    Parameters
    ----------
    module_name : str
        Module name.
    equations : dict[str, str]
        ODE equations.
    profile_name : str
        Target profile.
    include_carbon : bool
        Include carbon footprint section.
    include_reliability : bool
        Include reliability prediction.

    Returns
    -------
    str
        Markdown report.
    """
    from sc_neurocore.compiler.platforms import get_profile

    p = get_profile(profile_name)

    sections = [
        "# SC-NeuroCore Compilation Report",
        "",
        f"## Target: `{profile_name}`",
        f"- **Vendor**: {p.vendor}",
        f"- **Family**: {p.family}",
        f"- **Class**: {p.platform_class}",
        f"- **Width**: {p.data_width}-bit Q{p.data_width - p.fraction}.{p.fraction}",
        f"- **Overflow**: {p.overflow} | **Rounding**: {p.rounding}",
        "",
        f"## Module: `{module_name}`",
        f"- **State variables**: {len(equations)}",
        f"- **Equations**: {', '.join(equations.keys())}",
        "",
    ]

    if include_carbon:
        c = estimate_carbon_footprint(profile_name)
        sections.extend(
            [
                "## Carbon Footprint",
                f"- Manufacturing: {c.manufacturing_kg_co2} kg CO₂",
                f"- Operation (5yr): {c.total_5yr_kg_co2} kg CO₂",
                "",
            ]
        )

    if include_reliability:
        r = predict_reliability(voltage_v=0.9, temperature_c=85)
        sections.extend(
            [
                "## Reliability",
                f"- MTTF: {r.mttf_years} years",
                f"- Failure mode: {r.failure_mode}",
                "",
            ]
        )

    sections.extend(
        [
            "---",
            f"*Generated by SC-NeuroCore — {len(list(equations))} equations, "
            f"target {profile_name}*",
        ]
    )

    return "\n".join(sections)


# ═══════════════════════════════════════════════════════════════════════
# 72. Multi-Objective Pareto Explorer
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class ParetoPoint:
    """A single Pareto-optimal design point.

    Attributes
    ----------
    config : dict
    power_mw : float
    area_luts : int
    latency_ns : float
    """

    config: dict
    power_mw: float
    area_luts: int
    latency_ns: float


def explore_pareto(
    equations: dict[str, str],
    *,
    widths: list[int] | None = None,
    pipeline_depths: list[int] | None = None,
) -> list[ParetoPoint]:
    """Explore power/area/latency Pareto frontier.

    Parameters
    ----------
    equations : dict[str, str]
        State variable equations.
    widths : list[int], optional
        Bit widths to sweep. Default: [8, 16, 24, 32].
    pipeline_depths : list[int], optional
        Pipeline stages to sweep. Default: [1, 2, 4].

    Returns
    -------
    list[ParetoPoint]
        Non-dominated design points.
    """
    if widths is None:
        widths = [8, 16, 24, 32]
    if pipeline_depths is None:
        pipeline_depths = [1, 2, 4]

    n_vars = len(equations)
    points = []
    for w in widths:
        for d in pipeline_depths:
            power = n_vars * (w / 8) ** 1.5 * (1.0 / d) * 10
            area = n_vars * w * d * 3
            latency = 1000.0 / (d * (32 / w))
            points.append(
                ParetoPoint(
                    config={"data_width": w, "pipeline_depth": d},
                    power_mw=round(power, 2),
                    area_luts=area,
                    latency_ns=round(latency, 2),
                )
            )

    # Filter non-dominated
    pareto = []
    for p in points:
        dominated = False
        for q in points:
            if (
                q.power_mw <= p.power_mw
                and q.area_luts <= p.area_luts
                and q.latency_ns <= p.latency_ns
                and (
                    q.power_mw < p.power_mw
                    or q.area_luts < p.area_luts
                    or q.latency_ns < p.latency_ns
                )
            ):
                dominated = True
                break
        if not dominated:
            pareto.append(p)

    return sorted(pareto, key=lambda p: p.power_mw)


# ═══════════════════════════════════════════════════════════════════════
