# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — NeuroBench-compatible metrics

"""NeuroBench-compatible metrics: accuracy, compute complexity, spike counts.

Follows the NeuroBench algorithm track specification:
- Correctness metrics: accuracy, mAP, MSE (task-specific)
- Complexity metrics: synaptic operations, activation sparsity,
  total parameters, classification latency

Reference: NeuroBench (Nature Communications 2025)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class BenchmarkResult:
    """NeuroBench-compatible benchmark result."""

    task: str
    model: str
    accuracy: float
    total_parameters: int
    synaptic_operations: int
    activation_sparsity: float
    total_spikes: int
    timesteps: int
    latency_ms: float
    energy_nj: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)

    def to_neurobench_json(self) -> str:
        """Export as NeuroBench-compatible JSON."""
        result = {
            "task": self.task,
            "model": self.model,
            "metrics": {
                "correctness": {
                    "accuracy": self.accuracy,
                },
                "complexity": {
                    "total_parameters": self.total_parameters,
                    "synaptic_operations": self.synaptic_operations,
                    "activation_sparsity": self.activation_sparsity,
                    "total_spikes": self.total_spikes,
                    "timesteps": self.timesteps,
                },
                "system": {
                    "latency_ms": self.latency_ms,
                    "energy_nj": self.energy_nj,
                },
            },
            "framework": "sc-neurocore",
        }
        result["metrics"].update(self.extra)  # type: ignore[attr-defined]
        return json.dumps(result, indent=2)

    def summary(self) -> str:
        lines = [
            f"NeuroBench Result: {self.task} / {self.model}",
            f"  Accuracy:          {self.accuracy:.4f}",
            f"  Parameters:        {self.total_parameters:,}",
            f"  Synaptic ops:      {self.synaptic_operations:,}",
            f"  Sparsity:          {self.activation_sparsity:.2%}",
            f"  Total spikes:      {self.total_spikes:,}",
            f"  Timesteps:         {self.timesteps}",
            f"  Latency:           {self.latency_ms:.2f} ms",
        ]
        if self.energy_nj > 0:
            lines.append(f"  Energy:            {self.energy_nj:.2f} nJ")
        return "\n".join(lines)


def compute_metrics(
    predictions: np.ndarray[Any, Any],
    targets: np.ndarray[Any, Any],
    spike_counts: np.ndarray[Any, Any] | None = None,
    weights: list[np.ndarray[Any, Any]] | None = None,
    timesteps: int = 1,
    latency_ms: float = 0.0,
    task: str = "classification",
    model: str = "sc_neurocore",
) -> BenchmarkResult:
    """Compute NeuroBench-compatible metrics from model outputs.

    Parameters
    ----------
    predictions : ndarray
        Model predictions (class indices for classification).
    targets : ndarray
        Ground truth labels.
    spike_counts : ndarray, optional
        Per-sample total spike counts.
    weights : list of ndarray, optional
        Weight matrices for parameter counting.
    timesteps : int
        Number of simulation timesteps.
    latency_ms : float
        Inference latency in milliseconds.
    task : str
        Task name for the report.
    model : str
        Model name for the report.

    Returns
    -------
    BenchmarkResult
    """
    accuracy = float(np.mean(predictions == targets))

    total_params = sum(w.size for w in weights) if weights else 0

    if spike_counts is not None:
        total_spikes = int(spike_counts.sum())
        n_samples = len(predictions)
        sparsity = 1.0 - (total_spikes / max(total_params * timesteps * n_samples, 1))
    else:
        total_spikes = 0
        sparsity = 0.0

    # Synaptic operations: each spike activates fan-out synapses
    syn_ops = total_spikes * (total_params // max(timesteps, 1)) if weights else 0

    return BenchmarkResult(
        task=task,
        model=model,
        accuracy=accuracy,
        total_parameters=total_params,
        synaptic_operations=syn_ops,
        activation_sparsity=max(0.0, min(1.0, sparsity)),
        total_spikes=total_spikes,
        timesteps=timesteps,
        latency_ms=latency_ms,
    )
