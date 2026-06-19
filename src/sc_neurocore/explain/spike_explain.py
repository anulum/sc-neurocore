# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Spike-level explainability methods

"""Multi-method SNN explainability for network decision attribution.

Provided methods:
  - SpikeAttributor: backward attribution from output to input spikes
  - TemporalSaliency: perturbation-based — which input spikes matter most
  - CausalImportance: forward intervention — silence each neuron, measure impact

No SNN framework provides a unified, multi-method XAI toolkit.

Reference:
  Nguyen et al. 2024 — "Temporal Spike Attribution" (TSA)
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class ExplanationResult:
    """Result of an explanation method."""

    method: str
    importance_map: np.ndarray[Any, Any]  # (T, N) importance scores
    top_spikes: list[tuple[int, int, float]] = field(default_factory=list)
    summary_text: str = ""

    def top_k(self, k: int = 10) -> list[tuple[int, int, float]]:
        """Return top-k most important (timestep, neuron_id, score) tuples."""
        flat = self.importance_map.ravel()
        indices = np.argsort(flat)[::-1][:k]
        T = self.importance_map.shape[0]
        results = []
        for idx in indices:
            t = idx // self.importance_map.shape[1]
            n = idx % self.importance_map.shape[1]
            results.append((int(t), int(n), float(flat[idx])))
        return results

    def summary(self) -> str:
        """Render a human-readable report of the top attributed inputs."""
        top = self.top_k(5)
        lines = [f"Explanation ({self.method}):"]
        for t, n, score in top:
            lines.append(f"  t={t}, neuron={n}: importance={score:.4f}")
        return "\n".join(lines)


class SpikeAttributor:
    """Backward spike attribution via eligibility-trace chain.

    Traces the contribution of each input spike to the output
    through intermediate layers using eligibility trace products.
    Approximation of temporal backpropagation attribution.

    Parameters
    ----------
    decay : float
        Temporal decay factor for backward attribution (0-1).
    """

    def __init__(self, decay: float = 0.9):
        self.decay = decay

    def attribute(
        self,
        spikes: np.ndarray[Any, Any],
        weights: list[np.ndarray[Any, Any]],
        output_neuron: int = 0,
    ) -> ExplanationResult:
        """Compute per-input-spike attribution scores.

        Parameters
        ----------
        spikes : ndarray of shape (T, N_input)
            Input spike trains.
        weights : list of ndarray
            Weight matrices [W1, W2, ...] where W_i is (n_out, n_in).
        output_neuron : int
            Which output neuron to explain.

        Returns
        -------
        ExplanationResult with importance_map of shape (T, N_input)
        """
        T, N_in = spikes.shape
        importance = np.zeros((T, N_in))

        # Backward through weight chain: output_neuron → input
        # Attribution = product of weight paths * temporal decay
        attribution_weights = np.ones(N_in)
        for w in reversed(weights):
            if output_neuron < w.shape[0]:
                row = np.abs(w[output_neuron])
                if row.shape[0] == attribution_weights.shape[0]:
                    attribution_weights = attribution_weights * row
                else:
                    attribution_weights = np.abs(w[output_neuron])
                output_neuron = 0  # reset for next layer

        # Temporal attribution: weight input spikes by attribution + decay
        for t in range(T):
            time_weight = self.decay ** (T - 1 - t)
            importance[t] = spikes[t].astype(np.float64) * attribution_weights * time_weight

        # Normalize
        max_val = importance.max()
        if max_val > 0:
            importance /= max_val

        return ExplanationResult(
            method="spike_attribution",
            importance_map=importance,
        )


class TemporalSaliency:
    """Perturbation-based temporal saliency.

    For each input spike, measure the change in output when that spike
    is removed. Spikes whose removal causes large output change are
    salient (important).

    Parameters
    ----------
    run_fn : callable
        Function that takes input spikes (T, N) and returns output
        spike counts or rates (N_output,).
    """

    def __init__(self, run_fn: Callable[[np.ndarray[Any, Any]], np.ndarray[Any, Any]]):
        self.run_fn = run_fn

    def explain(
        self,
        spikes: np.ndarray[Any, Any],
        output_neuron: int = 0,
    ) -> ExplanationResult:
        """Compute perturbation-based saliency for each input spike.

        Parameters
        ----------
        spikes : ndarray of shape (T, N)
        output_neuron : int

        Returns
        -------
        ExplanationResult
        """
        T, N = spikes.shape
        baseline_output = self.run_fn(spikes)
        if baseline_output.ndim > 0:
            baseline_val = float(baseline_output[output_neuron])
        else:
            baseline_val = float(baseline_output)

        importance = np.zeros((T, N))

        # Find spike locations to perturb
        spike_locs = np.argwhere(spikes > 0)

        for t, n in spike_locs:
            perturbed = spikes.copy()
            perturbed[t, n] = 0
            perturbed_output = self.run_fn(perturbed)
            if perturbed_output.ndim > 0:
                new_val = float(perturbed_output[output_neuron])
            else:
                new_val = float(perturbed_output)
            importance[t, n] = abs(baseline_val - new_val)

        max_val = importance.max()
        if max_val > 0:
            importance /= max_val

        return ExplanationResult(
            method="temporal_saliency",
            importance_map=importance,
        )


class CausalImportance:
    """Causal importance via forward intervention.

    Silence each neuron (clamp to zero) across all timesteps and
    measure the impact on classification output. Builds a per-neuron
    causal importance score.

    Parameters
    ----------
    run_fn : callable
        Function that takes input spikes (T, N) and returns output (N_output,).
    """

    def __init__(self, run_fn: Callable[[np.ndarray[Any, Any]], np.ndarray[Any, Any]]):
        self.run_fn = run_fn

    def explain(
        self,
        spikes: np.ndarray[Any, Any],
        output_neuron: int = 0,
    ) -> ExplanationResult:
        """Compute causal importance by silencing each neuron.

        Parameters
        ----------
        spikes : ndarray of shape (T, N)
        output_neuron : int

        Returns
        -------
        ExplanationResult with importance_map of shape (1, N)
        """
        T, N = spikes.shape
        baseline_output = self.run_fn(spikes)
        if baseline_output.ndim > 0:
            baseline_val = float(baseline_output[output_neuron])
        else:
            baseline_val = float(baseline_output)

        neuron_importance = np.zeros(N)

        for n in range(N):
            silenced = spikes.copy()
            silenced[:, n] = 0
            silenced_output = self.run_fn(silenced)
            if silenced_output.ndim > 0:
                new_val = float(silenced_output[output_neuron])
            else:
                new_val = float(silenced_output)
            neuron_importance[n] = abs(baseline_val - new_val)

        max_val = neuron_importance.max()
        if max_val > 0:
            neuron_importance /= max_val

        importance_map = np.tile(neuron_importance, (1, 1))

        return ExplanationResult(
            method="causal_importance",
            importance_map=importance_map,
        )
