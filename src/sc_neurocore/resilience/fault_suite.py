# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Hardware fault resilience testing suite

"""Systematic fault injection and degradation analysis for SNN deployments.

Fault models: stuck-at neurons, bit flips in weights, dead synapses,
noisy membrane potentials, SC bitstream correlation faults.
Generates degradation curves (accuracy vs fault rate) and per-layer
vulnerability heatmaps.

SpikeFI exists but is tied to SLAYER. This suite integrates with
SC-native stochastic computing fault models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class FaultType(Enum):
    STUCK_AT_ZERO = "stuck_at_0"
    STUCK_AT_ONE = "stuck_at_1"
    WEIGHT_BIT_FLIP = "weight_bit_flip"
    DEAD_SYNAPSE = "dead_synapse"
    NOISY_MEMBRANE = "noisy_membrane"
    BITSTREAM_BIAS = "bitstream_bias"


@dataclass
class FaultModel:
    """One fault injection configuration."""

    fault_type: FaultType
    rate: float  # fraction of affected elements (0.0-1.0)
    layer_index: int | None = None  # None = all layers
    seed: int = 42


@dataclass
class FaultResult:
    """Result of one fault injection run."""

    fault_type: FaultType
    fault_rate: float
    layer_index: int | None
    accuracy_before: float
    accuracy_after: float
    degradation: float  # accuracy_before - accuracy_after


@dataclass
class ResilienceReport:
    """Full fault resilience report."""

    results: list[FaultResult] = field(default_factory=list)

    def degradation_curve(self, fault_type: FaultType) -> list[tuple[float, float]]:
        """Get (fault_rate, degradation) pairs for one fault type."""
        points = [(r.fault_rate, r.degradation) for r in self.results if r.fault_type == fault_type]
        points.sort(key=lambda x: x[0])
        return points

    def most_vulnerable_layer(self) -> int | None:
        """Return the layer index with highest average degradation."""
        layer_deg: dict[int, list[float]] = {}
        for r in self.results:
            if r.layer_index is not None:
                layer_deg.setdefault(r.layer_index, []).append(r.degradation)
        if not layer_deg:  # pragma: no cover
            return None
        return max(layer_deg, key=lambda k: np.mean(layer_deg[k]))

    def summary(self) -> str:
        lines = [f"Fault Resilience Report: {len(self.results)} experiments"]
        by_type: dict[str, list[FaultResult]] = {}
        for r in self.results:
            by_type.setdefault(r.fault_type.value, []).append(r)
        for ft, results in by_type.items():
            mean_deg = np.mean([r.degradation for r in results])
            max_deg = max(r.degradation for r in results)
            lines.append(f"  {ft}: mean_deg={mean_deg:.3f}, max_deg={max_deg:.3f}")
        mvl = self.most_vulnerable_layer()
        if mvl is not None:
            lines.append(f"  Most vulnerable layer: {mvl}")
        return "\n".join(lines)


class FaultResilienceSuite:
    """Systematic fault injection and resilience analysis.

    Parameters
    ----------
    eval_fn : callable
        Function(weights) -> accuracy. Takes list of weight matrices,
        returns accuracy in [0, 1].
    weights : list of ndarray
        Baseline (unfaulted) weight matrices.
    """

    def __init__(self, eval_fn, weights: list[np.ndarray[Any, Any]]):  # type: ignore[no-untyped-def]
        self.eval_fn = eval_fn
        self.weights = [w.copy() for w in weights]
        self._baseline_accuracy: float | None = None

    @property
    def baseline_accuracy(self) -> float:
        if self._baseline_accuracy is None:
            self._baseline_accuracy = self.eval_fn(self.weights)
        return self._baseline_accuracy

    def inject_fault(self, fault: FaultModel) -> list[np.ndarray[Any, Any]]:
        """Apply a fault model to weights, return faulted copies."""
        rng = np.random.RandomState(fault.seed)
        faulted = [w.copy() for w in self.weights]

        layers = [fault.layer_index] if fault.layer_index is not None else list(range(len(faulted)))

        for i in layers:
            w = faulted[i]
            mask = rng.random(w.shape) < fault.rate

            if fault.fault_type == FaultType.STUCK_AT_ZERO:
                w[mask] = 0.0
            elif fault.fault_type == FaultType.STUCK_AT_ONE:
                w[mask] = 1.0
            elif fault.fault_type == FaultType.WEIGHT_BIT_FLIP:
                # Flip sign of affected weights
                w[mask] = -w[mask]
            elif fault.fault_type == FaultType.DEAD_SYNAPSE:
                w[mask] = 0.0
            elif fault.fault_type == FaultType.NOISY_MEMBRANE:
                noise = rng.randn(*w.shape) * fault.rate * np.std(w)
                w += noise * mask
            elif fault.fault_type == FaultType.BITSTREAM_BIAS:
                # SC-specific: shift probabilities toward 0.5
                w[mask] = w[mask] * (1 - fault.rate) + 0.5 * fault.rate

            faulted[i] = w
        return faulted

    def run_single(self, fault: FaultModel) -> FaultResult:
        """Run one fault injection experiment."""
        faulted = self.inject_fault(fault)
        acc_after = self.eval_fn(faulted)
        return FaultResult(
            fault_type=fault.fault_type,
            fault_rate=fault.rate,
            layer_index=fault.layer_index,
            accuracy_before=self.baseline_accuracy,
            accuracy_after=acc_after,
            degradation=self.baseline_accuracy - acc_after,
        )

    def sweep(
        self,
        fault_type: FaultType,
        rates: list[float] | None = None,
        per_layer: bool = False,
    ) -> ResilienceReport:
        """Sweep fault rate and optionally per-layer.

        Parameters
        ----------
        fault_type : FaultType
        rates : list of float
            Fault rates to test.
        per_layer : bool
            If True, test each layer independently.
        """
        if rates is None:  # pragma: no cover
            rates = [0.01, 0.05, 0.1, 0.2, 0.5]

        report = ResilienceReport()

        if per_layer:
            for layer_idx in range(len(self.weights)):
                for rate in rates:
                    fault = FaultModel(fault_type=fault_type, rate=rate, layer_index=layer_idx)
                    report.results.append(self.run_single(fault))
        else:
            for rate in rates:
                fault = FaultModel(fault_type=fault_type, rate=rate)
                report.results.append(self.run_single(fault))

        return report

    def full_audit(self) -> ResilienceReport:
        """Run all fault types at standard rates, per-layer."""
        report = ResilienceReport()
        rates = [0.01, 0.05, 0.1, 0.2]
        for ft in FaultType:
            for layer_idx in range(len(self.weights)):
                for rate in rates:
                    fault = FaultModel(fault_type=ft, rate=rate, layer_index=layer_idx)
                    report.results.append(self.run_single(fault))
        return report
