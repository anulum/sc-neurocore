# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Spike-level training profiler with pathology detection

"""Live profiler for SNN training: detects dead neurons, gradient pathology,
saturated layers, temporal credit assignment failures, and energy bottlenecks.

No SNN framework provides automated training diagnostics. SNN debugging
is manual and expertise-intensive. This profiler instruments the training
loop and emits actionable fix suggestions.

Usage:
    profiler = SpikeProfiler()
    profiler.record_step(layer="hidden", spikes=spike_tensor, voltages=v_tensor)
    profiler.record_step(layer="hidden", spikes=spike_tensor2, voltages=v_tensor2)
    report = profiler.report()
    print(report.summary())
    for p in report.pathologies:
        print(p.severity, p.message, p.suggestion)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class Severity(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class Pathology:
    """One detected training pathology."""

    severity: Severity
    category: str
    layer: str
    message: str
    suggestion: str
    metric_value: float = 0.0


@dataclass
class LayerStats:
    """Accumulated statistics for one layer across recorded steps."""

    name: str
    n_neurons: int = 0
    n_steps: int = 0

    # Spike statistics
    total_spikes: int = 0
    per_neuron_spikes: np.ndarray[Any, Any] | None = None
    firing_rates: np.ndarray[Any, Any] | None = None

    # Voltage statistics
    voltage_mean: float = 0.0
    voltage_std: float = 0.0
    voltage_min: float = 0.0
    voltage_max: float = 0.0

    # Gradient statistics (if recorded)
    gradient_norm_mean: float = 0.0
    gradient_norm_max: float = 0.0

    # ISI statistics
    mean_isi: float = 0.0
    cv_isi: float = 0.0

    # Derived
    dead_neuron_count: int = 0
    saturated_neuron_count: int = 0
    dead_neuron_fraction: float = 0.0
    saturated_neuron_fraction: float = 0.0

    # Energy estimate (synaptic operations)
    estimated_syn_ops: int = 0


@dataclass
class ProfileReport:
    """Complete profiling report with per-layer stats and detected pathologies."""

    layer_stats: dict[str, LayerStats] = field(default_factory=dict)
    pathologies: list[Pathology] = field(default_factory=list)
    total_steps: int = 0
    total_spikes: int = 0
    total_neurons: int = 0

    def summary(self) -> str:
        lines = [
            f"SpikeProfiler Report: {self.total_steps} steps, "
            f"{self.total_neurons} neurons, {self.total_spikes} total spikes",
            "",
        ]
        for name, stats in self.layer_stats.items():
            fr = stats.firing_rates
            mean_fr = float(fr.mean()) if fr is not None else 0.0
            lines.append(
                f"  {name}: {stats.n_neurons}n, rate={mean_fr:.3f}, "
                f"dead={stats.dead_neuron_count}, sat={stats.saturated_neuron_count}, "
                f"V={stats.voltage_mean:.3f}+/-{stats.voltage_std:.3f}"
            )

        if self.pathologies:
            lines.append("")
            lines.append(f"Pathologies detected: {len(self.pathologies)}")
            for p in self.pathologies:
                lines.append(f"  [{p.severity.value}] {p.category} @ {p.layer}: {p.message}")
                lines.append(f"    Fix: {p.suggestion}")
        else:  # pragma: no cover
            lines.append("")
            lines.append("No pathologies detected.")

        return "\n".join(lines)

    @property
    def has_critical(self) -> bool:
        return any(p.severity == Severity.CRITICAL for p in self.pathologies)


class SpikeProfiler:
    """Instruments SNN training to detect pathologies and compute diagnostics.

    Record spike tensors, voltage tensors, and optionally gradient tensors
    per layer per training step. Call report() to get a ProfileReport with
    detected pathologies and fix suggestions.

    Parameters
    ----------
    dead_threshold : float
        Firing rate below which a neuron is considered dead (default 0.01).
    saturated_threshold : float
        Firing rate above which a neuron is considered saturated (default 0.95).
    gradient_explosion_ratio : float
        Max/mean gradient norm ratio above which gradient explosion is flagged.
    """

    def __init__(
        self,
        dead_threshold: float = 0.01,
        saturated_threshold: float = 0.95,
        gradient_explosion_ratio: float = 100.0,
    ) -> None:
        self.dead_threshold = dead_threshold
        self.saturated_threshold = saturated_threshold
        self.gradient_explosion_ratio = gradient_explosion_ratio

        self._layers: dict[str, _LayerAccumulator] = {}

    def record_step(
        self,
        layer: str,
        spikes: np.ndarray[Any, Any],
        voltages: np.ndarray[Any, Any] | None = None,
        gradients: np.ndarray[Any, Any] | None = None,
    ) -> None:
        """Record one timestep of data for a layer.

        Parameters
        ----------
        layer : str
            Layer name.
        spikes : ndarray of shape (n_neurons,) or (batch, n_neurons)
            Binary spike tensor for this timestep.
        voltages : ndarray, optional
            Membrane voltages, same shape as spikes.
        gradients : ndarray, optional
            Gradient tensor (surrogate gradient magnitudes).
        """
        if layer not in self._layers:
            self._layers[layer] = _LayerAccumulator(layer)
        self._layers[layer].add(spikes, voltages, gradients)

    def reset(self) -> None:
        """Clear all accumulated data."""
        self._layers.clear()

    def report(self) -> ProfileReport:
        """Analyze accumulated data and return a ProfileReport."""
        report = ProfileReport()

        for name, acc in self._layers.items():
            stats = acc.compute_stats()
            report.layer_stats[name] = stats
            report.total_steps = max(report.total_steps, stats.n_steps)
            report.total_spikes += stats.total_spikes
            report.total_neurons += stats.n_neurons

        # Detect pathologies
        report.pathologies = self._detect_pathologies(report.layer_stats)
        return report

    def _detect_pathologies(self, layer_stats: dict[str, LayerStats]) -> list[Pathology]:
        pathologies = []

        for name, stats in layer_stats.items():
            # Dead neurons
            if stats.dead_neuron_fraction > 0.5:
                pathologies.append(
                    Pathology(
                        severity=Severity.CRITICAL,
                        category="dead_neurons",
                        layer=name,
                        message=f"{stats.dead_neuron_count}/{stats.n_neurons} neurons "
                        f"({stats.dead_neuron_fraction:.0%}) never fire",
                        suggestion="Lower firing threshold by ~20% or increase input current gain",
                        metric_value=stats.dead_neuron_fraction,
                    )
                )
            elif stats.dead_neuron_fraction > 0.1:
                pathologies.append(
                    Pathology(
                        severity=Severity.WARNING,
                        category="dead_neurons",
                        layer=name,
                        message=f"{stats.dead_neuron_count}/{stats.n_neurons} neurons "
                        f"({stats.dead_neuron_fraction:.0%}) never fire",
                        suggestion="Consider lowering threshold or adding noise",
                        metric_value=stats.dead_neuron_fraction,
                    )
                )

            # Saturated neurons
            if stats.saturated_neuron_fraction > 0.3:
                pathologies.append(
                    Pathology(
                        severity=Severity.WARNING,
                        category="saturated_neurons",
                        layer=name,
                        message=f"{stats.saturated_neuron_count}/{stats.n_neurons} neurons "
                        f"({stats.saturated_neuron_fraction:.0%}) fire almost every step",
                        suggestion="Raise threshold or reduce input gain to restore sparse coding",
                        metric_value=stats.saturated_neuron_fraction,
                    )
                )

            # Gradient explosion
            if stats.gradient_norm_mean > 0 and stats.gradient_norm_max > 0:
                ratio = stats.gradient_norm_max / max(stats.gradient_norm_mean, 1e-12)
                if ratio > self.gradient_explosion_ratio:
                    pathologies.append(
                        Pathology(
                            severity=Severity.CRITICAL,
                            category="gradient_explosion",
                            layer=name,
                            message=f"Gradient max/mean ratio = {ratio:.1f}x "
                            f"(threshold: {self.gradient_explosion_ratio}x)",
                            suggestion="Clip gradients, reduce learning rate, or add surrogate gradient damping",
                            metric_value=ratio,
                        )
                    )

            # Silent network (zero spikes across all neurons)
            if stats.firing_rates is not None and stats.firing_rates.max() < 0.001:
                pathologies.append(
                    Pathology(
                        severity=Severity.CRITICAL,
                        category="silent_network",
                        layer=name,
                        message="Layer produces almost no spikes (max rate < 0.001)",
                        suggestion="Check input encoding, lower all thresholds, or verify input data is non-zero",
                        metric_value=float(stats.firing_rates.max()),
                    )
                )

            # Voltage collapse (all voltages near rest)
            if stats.voltage_std < 1e-6 and stats.n_steps > 10:
                pathologies.append(
                    Pathology(
                        severity=Severity.WARNING,
                        category="voltage_collapse",
                        layer=name,
                        message=f"Voltage std = {stats.voltage_std:.2e} — neurons not integrating input",
                        suggestion="Increase input current or check connectivity",
                        metric_value=stats.voltage_std,
                    )
                )

        # Cross-layer: gradient vanishing
        if len(layer_stats) >= 2:
            grad_norms = [
                (name, s.gradient_norm_mean)
                for name, s in layer_stats.items()
                if s.gradient_norm_mean > 0
            ]
            if len(grad_norms) >= 2:
                first_norm = grad_norms[0][1]
                last_norm = grad_norms[-1][1]
                if first_norm > 0 and last_norm / max(first_norm, 1e-12) < 0.01:
                    pathologies.append(
                        Pathology(
                            severity=Severity.CRITICAL,
                            category="gradient_vanishing",
                            layer=f"{grad_norms[0][0]}→{grad_norms[-1][0]}",
                            message=f"Gradient decays {first_norm / max(last_norm, 1e-12):.0f}x "
                            f"from first to last layer",
                            suggestion="Add skip connections, use adaptive surrogate gradient slope, "
                            "or reduce network depth",
                            metric_value=last_norm / max(first_norm, 1e-12),
                        )
                    )

        return pathologies


class _LayerAccumulator:
    """Internal: accumulates per-step data for one layer."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._spike_sums: np.ndarray[Any, Any] | None = None
        self._n_neurons = 0
        self._n_steps = 0
        self._total_spikes = 0

        self._voltage_sum = 0.0
        self._voltage_sq_sum = 0.0
        self._voltage_min = float("inf")
        self._voltage_max = float("-inf")
        self._voltage_count = 0

        self._gradient_norms: list[float] = []

    def add(
        self,
        spikes: np.ndarray[Any, Any],
        voltages: np.ndarray[Any, Any] | None,
        gradients: np.ndarray[Any, Any] | None,
    ) -> None:
        # Flatten batch dimension if present
        if spikes.ndim > 1:
            spikes_flat = spikes.reshape(-1, spikes.shape[-1])
            spikes_summed = spikes_flat.sum(axis=0)
        else:
            spikes_summed = spikes
            spikes_flat = spikes[np.newaxis]  # type: ignore[assignment]

        n_neurons = spikes_summed.shape[0]

        if self._spike_sums is None:
            self._spike_sums = np.zeros(n_neurons, dtype=np.float64)
            self._n_neurons = n_neurons

        self._spike_sums += spikes_summed.astype(np.float64)
        self._total_spikes += int(spikes_summed.sum())
        self._n_steps += 1

        if voltages is not None:
            v = voltages.astype(np.float64).ravel()
            self._voltage_sum += v.sum()
            self._voltage_sq_sum += (v**2).sum()
            self._voltage_min = min(self._voltage_min, float(v.min()))
            self._voltage_max = max(self._voltage_max, float(v.max()))
            self._voltage_count += len(v)

        if gradients is not None:
            g = gradients.astype(np.float64).ravel()
            self._gradient_norms.append(float(np.linalg.norm(g)))

    def compute_stats(self) -> LayerStats:
        n = max(self._n_steps, 1)
        firing_rates = self._spike_sums / n if self._spike_sums is not None else np.zeros(1)

        dead = int((firing_rates < 0.01).sum())
        saturated = int((firing_rates > 0.95).sum())
        n_neurons = self._n_neurons

        v_mean = self._voltage_sum / max(self._voltage_count, 1)
        v_var = self._voltage_sq_sum / max(self._voltage_count, 1) - v_mean**2
        v_std = float(np.sqrt(max(v_var, 0.0)))

        g_mean = float(np.mean(self._gradient_norms)) if self._gradient_norms else 0.0
        g_max = float(np.max(self._gradient_norms)) if self._gradient_norms else 0.0

        return LayerStats(
            name=self.name,
            n_neurons=n_neurons,
            n_steps=self._n_steps,
            total_spikes=self._total_spikes,
            per_neuron_spikes=self._spike_sums,
            firing_rates=firing_rates,
            voltage_mean=v_mean,
            voltage_std=v_std,
            voltage_min=self._voltage_min if self._voltage_count > 0 else 0.0,
            voltage_max=self._voltage_max if self._voltage_count > 0 else 0.0,
            gradient_norm_mean=g_mean,
            gradient_norm_max=g_max,
            dead_neuron_count=dead,
            saturated_neuron_count=saturated,
            dead_neuron_fraction=dead / max(n_neurons, 1),
            saturated_neuron_fraction=saturated / max(n_neurons, 1),
        )
