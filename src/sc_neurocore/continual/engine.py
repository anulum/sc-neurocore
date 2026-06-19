# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Continual learning engine with EWC + on-chip plasticity

"""Continual learning pipeline: train with backprop, deploy with local plasticity.

Combines Elastic Weight Consolidation (EWC) for catastrophic forgetting
protection with STDP-based local learning rules that can run on-chip.
The pipeline extracts per-synapse plasticity parameters from the trained
model and emits a deployment config including active learning rules.

No framework provides the integrated pipeline from "trained model" to
"deployed model with active on-chip plasticity."

Reference:
  Kirkpatrick et al. 2017 — "Overcoming catastrophic forgetting" (EWC)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class PlasticityConfig:
    """Per-layer on-chip plasticity configuration.

    Extracted from training for hardware deployment.

    Parameters
    ----------
    layer_name : str
    rule : str
        Plasticity rule: 'stdp', 'r_stdp', 'homeostatic', 'none'.
    tau_pre : float
        Pre-synaptic trace time constant (ms).
    tau_post : float
        Post-synaptic trace time constant (ms).
    lr_potentiation : float
        Potentiation learning rate (A+).
    lr_depression : float
        Depression learning rate (A-).
    w_min : float
        Minimum weight.
    w_max : float
        Maximum weight.
    homeostatic_target : float
        Target firing rate for homeostatic regulation.
    """

    layer_name: str
    rule: str = "stdp"
    tau_pre: float = 20.0
    tau_post: float = 20.0
    lr_potentiation: float = 0.01
    lr_depression: float = 0.012
    w_min: float = 0.0
    w_max: float = 1.0
    homeostatic_target: float = 0.1


@dataclass
class ContinualReport:
    """Report from a continual learning session."""

    tasks_trained: int = 0
    ewc_lambda: float = 0.0
    fisher_computed: bool = False
    plasticity_configs: list[PlasticityConfig] = field(default_factory=list)
    accuracy_per_task: list[float] = field(default_factory=list)

    def summary(self) -> str:
        """Render a multi-line human-readable continual-learning report."""
        lines = [
            f"Continual Learning Report: {self.tasks_trained} tasks",
            f"  EWC lambda: {self.ewc_lambda}",
            f"  Fisher diagonal: {'computed' if self.fisher_computed else 'not computed'}",
            f"  Plasticity configs: {len(self.plasticity_configs)} layers",
        ]
        for i, acc in enumerate(self.accuracy_per_task):
            lines.append(f"  Task {i}: accuracy = {acc:.4f}")
        return "\n".join(lines)


class ContinualLearner:
    """Continual learning engine with EWC and on-chip plasticity extraction.

    Parameters
    ----------
    weights : list of ndarray
        Initial trained weight matrices per layer.
    layer_names : list of str
        Names for each layer.
    ewc_lambda : float
        Regularization strength for EWC (0 = no protection).
    plasticity_rule : str
        Default on-chip plasticity rule for all layers.
    """

    def __init__(
        self,
        weights: list[np.ndarray[Any, Any]],
        layer_names: list[str] | None = None,
        ewc_lambda: float = 1000.0,
        plasticity_rule: str = "stdp",
    ):
        self.weights = [w.copy() for w in weights]
        self.layer_names = layer_names or [f"layer_{i}" for i in range(len(weights))]
        self.ewc_lambda = ewc_lambda
        self.plasticity_rule = plasticity_rule

        self._fisher_diag: list[np.ndarray[Any, Any]] | None = None
        self._star_weights: list[np.ndarray[Any, Any]] | None = None
        self._task_count = 0
        self._accuracy_history: list[float] = []

    def compute_fisher(self, gradients_per_sample: list[list[np.ndarray[Any, Any]]]) -> None:
        """Compute Fisher Information diagonal from per-sample gradients.

        Parameters
        ----------
        gradients_per_sample : list of (list of ndarray)
            Outer list: samples. Inner list: gradient per layer.
            Each ndarray has same shape as the corresponding weight matrix.
        """
        n_layers = len(self.weights)
        fisher = [np.zeros_like(w) for w in self.weights]

        for sample_grads in gradients_per_sample:
            for i in range(min(len(sample_grads), n_layers)):
                fisher[i] += sample_grads[i] ** 2

        n_samples = max(len(gradients_per_sample), 1)
        self._fisher_diag = [f / n_samples for f in fisher]
        self._star_weights = [w.copy() for w in self.weights]

    def ewc_penalty(self) -> float:
        """Compute EWC regularization penalty."""
        if self._fisher_diag is None or self._star_weights is None:
            return 0.0
        penalty = 0.0
        for w, w_star, fisher in zip(self.weights, self._star_weights, self._fisher_diag):
            penalty += float(np.sum(fisher * (w - w_star) ** 2))
        return 0.5 * self.ewc_lambda * penalty

    def register_task(self, accuracy: float) -> None:
        """Register completion of a task."""
        self._task_count += 1
        self._accuracy_history.append(accuracy)

    def update_weights(self, new_weights: list[np.ndarray[Any, Any]]) -> None:
        """Update weights (e.g., after training on a new task)."""
        self.weights = [w.copy() for w in new_weights]

    def extract_plasticity_configs(self) -> list[PlasticityConfig]:
        """Extract per-layer plasticity parameters for on-chip deployment.

        Derives STDP parameters from weight statistics:
        - LR proportional to weight variance (active synapses learn faster)
        - Bounds from weight range
        - Homeostatic target from mean firing rate proxy
        """
        configs = []
        for i, (w, name) in enumerate(zip(self.weights, self.layer_names)):
            w_std = float(np.std(w))
            w_range = float(w.max() - w.min())
            lr_scale = min(w_std * 0.1, 0.05)

            configs.append(
                PlasticityConfig(
                    layer_name=name,
                    rule=self.plasticity_rule,
                    tau_pre=20.0,
                    tau_post=20.0,
                    lr_potentiation=lr_scale,
                    lr_depression=lr_scale * 1.2,
                    w_min=float(w.min()),
                    w_max=float(w.max()),
                    homeostatic_target=0.1,
                )
            )
        return configs

    def report(self) -> ContinualReport:
        """Generate a continual learning report."""
        configs = self.extract_plasticity_configs()
        return ContinualReport(
            tasks_trained=self._task_count,
            ewc_lambda=self.ewc_lambda,
            fisher_computed=self._fisher_diag is not None,
            plasticity_configs=configs,
            accuracy_per_task=list(self._accuracy_history),
        )
