# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SNN Architecture Doctor

"""Run holistic diagnostics on an SNN architecture.

Takes layer sizes, weights, spike data, and a target FPGA. Produces a
diagnostic report covering: hardware utilization, coding efficiency,
weight health, spike statistics, and overprovisioning. Each finding
includes severity, metric, and a specific fix recommendation.

No SNN framework provides automated architecture-level diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class Severity(Enum):
    OK = "ok"
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class Diagnosis:
    """Single diagnostic finding."""

    category: str
    severity: Severity
    message: str
    suggestion: str
    metric: float = 0.0


@dataclass
class DiagnosticReport:
    """Full diagnostic report for an SNN architecture."""

    target: str
    findings: list[Diagnosis] = field(default_factory=list)

    def summary(self) -> str:
        lines = [f"SNN Architecture Doctor — target: {self.target}", ""]
        counts = {s: 0 for s in Severity}
        for f in self.findings:
            counts[f.severity] += 1

        lines.append(
            f"  {counts[Severity.CRITICAL]} critical, {counts[Severity.WARNING]} warning, "
            f"{counts[Severity.INFO]} info, {counts[Severity.OK]} ok"
        )
        lines.append("")

        for f in self.findings:
            if f.severity == Severity.OK:
                continue
            lines.append(f"  [{f.severity.value}] {f.category}: {f.message}")
            lines.append(f"    Fix: {f.suggestion}")
        return "\n".join(lines)

    @property
    def has_critical(self) -> bool:
        return any(f.severity == Severity.CRITICAL for f in self.findings)

    @property
    def score(self) -> int:
        """Health score 0-100. 100 = no issues."""
        penalty = sum(
            10 if f.severity == Severity.CRITICAL else 5 if f.severity == Severity.WARNING else 1
            for f in self.findings
            if f.severity != Severity.OK
        )
        return max(0, 100 - penalty)


def diagnose(
    layer_sizes: list[tuple[int, int]],
    weights: list[np.ndarray[Any, Any]] | None = None,
    spike_rates: list[np.ndarray[Any, Any]] | None = None,
    target: str = "ice40",
    bitstream_length: int = 256,
) -> DiagnosticReport:
    """Run architecture diagnostics.

    Parameters
    ----------
    layer_sizes : list of (n_inputs, n_neurons)
    weights : list of ndarray, optional
        Weight matrices per layer.
    spike_rates : list of ndarray, optional
        Per-neuron firing rates per layer (from profiling).
    target : str
        FPGA target for hardware checks.
    bitstream_length : int
        SC bitstream length.

    Returns
    -------
    DiagnosticReport
    """
    report = DiagnosticReport(target=target)

    # 1. Hardware utilization check
    _check_hardware(report, layer_sizes, target, bitstream_length)

    # 2. Weight health
    if weights is not None:
        _check_weights(report, weights)

    # 3. Spike rate health
    if spike_rates is not None:
        _check_spike_rates(report, spike_rates)

    # 4. Architecture balance
    _check_architecture(report, layer_sizes)

    # 5. Coding efficiency
    _check_coding_efficiency(report, layer_sizes, bitstream_length)

    return report


def _check_hardware(
    report: DiagnosticReport,
    layer_sizes: list[tuple[int, int]],
    target: str,
    bitstream_length: int,
) -> None:
    from sc_neurocore.energy.estimator import estimate

    est = estimate(layer_sizes, target=target, bitstream_length=bitstream_length)

    if not est.fits_on_target:
        report.findings.append(
            Diagnosis(
                category="hardware_fit",
                severity=Severity.CRITICAL,
                message=f"Network exceeds {target} capacity: {est.utilization_pct:.0f}% utilization",
                suggestion=f"Reduce layer widths, lower bitstream length, or use a larger FPGA. "
                f"Current: {est.total_luts} LUTs",
                metric=est.utilization_pct,
            )
        )
    elif est.utilization_pct > 80:
        report.findings.append(
            Diagnosis(
                category="hardware_fit",
                severity=Severity.WARNING,
                message=f"High utilization: {est.utilization_pct:.0f}%",
                suggestion="Consider pruning or reducing bitstream length for routing margin",
                metric=est.utilization_pct,
            )
        )
    elif est.utilization_pct < 20:
        report.findings.append(
            Diagnosis(
                category="hardware_overprovisioned",
                severity=Severity.INFO,
                message=f"Only {est.utilization_pct:.0f}% of {target} used",
                suggestion="Network could be larger or use a smaller FPGA to save cost/power",
                metric=est.utilization_pct,
            )
        )
    else:  # pragma: no cover
        report.findings.append(
            Diagnosis(
                category="hardware_fit",
                severity=Severity.OK,
                message=f"Fits on {target}: {est.utilization_pct:.0f}% utilization",
                suggestion="",
                metric=est.utilization_pct,
            )
        )


def _check_weights(report: DiagnosticReport, weights: list[np.ndarray[Any, Any]]) -> None:
    for i, w in enumerate(weights):
        # Near-zero weights (dead synapses)
        sparsity = float(np.mean(np.abs(w) < 1e-6))
        if sparsity > 0.9:
            report.findings.append(
                Diagnosis(
                    category="weight_sparsity",
                    severity=Severity.WARNING,
                    message=f"Layer {i}: {sparsity:.0%} near-zero weights",
                    suggestion="Apply structured pruning to reduce layer width and save hardware",
                    metric=sparsity,
                )
            )

        # Weight magnitude outliers
        abs_w = np.abs(w)
        if abs_w.max() > 10 * abs_w.mean() and abs_w.mean() > 0:
            ratio = float(abs_w.max() / abs_w.mean())
            report.findings.append(
                Diagnosis(
                    category="weight_outliers",
                    severity=Severity.WARNING,
                    message=f"Layer {i}: max weight is {ratio:.0f}x mean — outlier risk",
                    suggestion="Apply weight clipping or normalization",
                    metric=ratio,
                )
            )

        # SC range check: weights outside [0, 1] need rescaling
        if w.min() < -0.01 or w.max() > 1.01:
            report.findings.append(
                Diagnosis(
                    category="weight_sc_range",
                    severity=Severity.INFO,
                    message=f"Layer {i}: weights outside SC [0,1] range "
                    f"(min={w.min():.2f}, max={w.max():.2f})",
                    suggestion="Apply sigmoid or min-max normalization for SC encoding",
                    metric=float(max(abs(w.min()), abs(w.max()))),
                )
            )


def _check_spike_rates(report: DiagnosticReport, spike_rates: list[np.ndarray[Any, Any]]) -> None:
    for i, rates in enumerate(spike_rates):
        dead = float(np.mean(rates < 0.01))
        saturated = float(np.mean(rates > 0.95))

        if dead > 0.5:
            report.findings.append(
                Diagnosis(
                    category="dead_neurons",
                    severity=Severity.CRITICAL,
                    message=f"Layer {i}: {dead:.0%} neurons are dead (rate < 0.01)",
                    suggestion="Lower threshold, increase input gain, or add noise",
                    metric=dead,
                )
            )
        if saturated > 0.3:
            report.findings.append(
                Diagnosis(
                    category="saturated_neurons",
                    severity=Severity.WARNING,
                    message=f"Layer {i}: {saturated:.0%} neurons saturated (rate > 0.95)",
                    suggestion="Raise threshold or reduce input magnitude",
                    metric=saturated,
                )
            )
        mean_rate = float(rates.mean())
        if 0.05 < mean_rate < 0.5:
            report.findings.append(
                Diagnosis(
                    category="spike_efficiency",
                    severity=Severity.OK,
                    message=f"Layer {i}: mean firing rate {mean_rate:.2f} — good sparse coding",
                    suggestion="",
                    metric=mean_rate,
                )
            )


def _check_architecture(report: DiagnosticReport, layer_sizes: list[tuple[int, int]]) -> None:
    widths = [n for _, n in layer_sizes]

    # Bottleneck detection: sudden width reduction > 4x
    for i in range(len(widths) - 1):
        ratio = widths[i] / max(widths[i + 1], 1)
        if ratio > 4:
            report.findings.append(
                Diagnosis(
                    category="architecture_bottleneck",
                    severity=Severity.WARNING,
                    message=f"Layer {i}→{i + 1}: {ratio:.1f}x width reduction "
                    f"({widths[i]}→{widths[i + 1]})",
                    suggestion="Add intermediate layer or increase bottleneck width",
                    metric=ratio,
                )
            )

    # Single hidden layer might underfit
    if len(layer_sizes) == 1 and layer_sizes[0][1] < 16:
        report.findings.append(
            Diagnosis(
                category="architecture_capacity",
                severity=Severity.INFO,
                message=f"Single layer with {layer_sizes[0][1]} neurons — may underfit",
                suggestion="Add a hidden layer or increase width",
                metric=float(layer_sizes[0][1]),
            )
        )


def _check_coding_efficiency(
    report: DiagnosticReport,
    layer_sizes: list[tuple[int, int]],
    bitstream_length: int,
) -> None:
    total_neurons = sum(n for _, n in layer_sizes)

    if bitstream_length > 256 and total_neurons < 50:
        report.findings.append(
            Diagnosis(
                category="coding_overprovisioned",
                severity=Severity.INFO,
                message=f"L={bitstream_length} with only {total_neurons} neurons — precision may be wasted",
                suggestion=f"Try L={bitstream_length // 2} for 2x throughput with minimal accuracy loss",
                metric=float(bitstream_length),
            )
        )

    if bitstream_length < 64 and total_neurons > 200:
        report.findings.append(
            Diagnosis(
                category="coding_underprovisioned",
                severity=Severity.WARNING,
                message=f"L={bitstream_length} with {total_neurons} neurons — precision may be too low",
                suggestion="Increase bitstream length to 128+ for better accuracy",
                metric=float(bitstream_length),
            )
        )
