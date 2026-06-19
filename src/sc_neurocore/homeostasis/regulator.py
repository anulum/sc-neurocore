# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Network-wide homeostatic regulation

"""Network-wide homeostatic regulation for deploy-and-forget stability.

Auto-adjusts thresholds, learning rates, and inhibition to maintain
stability without external supervision.

Includes sleep consolidation: periodic synaptic renormalization +
spontaneous replay for memory consolidation without external input.

Reference: Sleep-Based Homeostatic Regularization (arXiv Jan 2026)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class StabilityMetrics:
    """Network stability measurements."""

    mean_firing_rate: float = 0.0
    rate_variance: float = 0.0
    ei_ratio: float = 1.0
    weight_norm: float = 0.0
    is_stable: bool = True
    adjustments_made: list[str] = field(default_factory=list)

    def summary(self) -> str:
        """Render a multi-line human-readable network-stability report."""
        status = "STABLE" if self.is_stable else "UNSTABLE"
        lines = [
            f"Network Stability: {status}",
            f"  Mean firing rate: {self.mean_firing_rate:.4f}",
            f"  Rate variance: {self.rate_variance:.4f}",
            f"  E/I ratio: {self.ei_ratio:.2f}",
            f"  Weight norm: {self.weight_norm:.4f}",
        ]
        if self.adjustments_made:  # pragma: no cover
            lines.append(f"  Adjustments: {', '.join(self.adjustments_made)}")
        return "\n".join(lines)


class NetworkRegulator:
    """Network-wide homeostatic regulator.

    Monitors population firing rates and adjusts thresholds, learning rates,
    and weights to maintain target activity levels.

    Parameters
    ----------
    target_rate : float
        Target mean firing rate (spikes per step).
    rate_tolerance : float
        Acceptable deviation from target (fraction).
    threshold_step : float
        Per-step threshold adjustment magnitude.
    lr_scale_factor : float
        Multiplicative LR adjustment factor.
    """

    def __init__(
        self,
        target_rate: float = 0.1,
        rate_tolerance: float = 0.5,
        threshold_step: float = 0.01,
        lr_scale_factor: float = 0.95,
    ):
        if not np.isfinite(target_rate) or target_rate < 0.0:
            raise ValueError("target_rate must be finite and non-negative")
        if not np.isfinite(rate_tolerance) or not 0.0 <= rate_tolerance <= 1.0:
            raise ValueError("rate_tolerance must be finite and within [0, 1]")
        if not np.isfinite(threshold_step) or threshold_step < 0.0:
            raise ValueError("threshold_step must be finite and non-negative")
        if not np.isfinite(lr_scale_factor) or not 0.0 < lr_scale_factor <= 1.0:
            raise ValueError("lr_scale_factor must be finite and within (0, 1]")

        self.target_rate = target_rate
        self.rate_tolerance = rate_tolerance
        self.threshold_step = threshold_step
        self.lr_scale_factor = lr_scale_factor

    def regulate(
        self,
        firing_rates: np.ndarray[Any, Any],
        thresholds: np.ndarray[Any, Any],
        learning_rate: float,
        weights: list[np.ndarray[Any, Any]] | None = None,
    ) -> tuple[np.ndarray[Any, Any], float, StabilityMetrics]:
        """Apply homeostatic regulation.

        Parameters
        ----------
        firing_rates : ndarray of shape (N,)
            Current per-neuron firing rates.
        thresholds : ndarray of shape (N,)
            Current per-neuron thresholds.
        learning_rate : float
            Current learning rate.
        weights : list of ndarray, optional
            Weight matrices for norm monitoring.

        Returns
        -------
        (new_thresholds, new_lr, StabilityMetrics)
        """
        self._validate_regulate_inputs(firing_rates, thresholds, learning_rate, weights)

        mean_rate = float(firing_rates.mean())
        rate_var = float(firing_rates.var())
        metrics = StabilityMetrics(
            mean_firing_rate=mean_rate,
            rate_variance=rate_var,
        )

        if weights:
            metrics.weight_norm = float(np.mean([np.linalg.norm(w) for w in weights]))

        new_thresholds = thresholds.copy()
        new_lr = learning_rate

        lo = self.target_rate * (1 - self.rate_tolerance)
        hi = self.target_rate * (1 + self.rate_tolerance)

        # Too active → raise thresholds
        if mean_rate > hi:
            new_thresholds += self.threshold_step
            metrics.adjustments_made.append(f"thresholds +{self.threshold_step:.3f}")
            metrics.is_stable = False

        # Too quiet → lower thresholds
        elif mean_rate < lo:
            new_thresholds -= self.threshold_step
            metrics.adjustments_made.append(f"thresholds -{self.threshold_step:.3f}")
            metrics.is_stable = False

        # High variance → reduce LR
        if rate_var > self.target_rate * 2:
            new_lr *= self.lr_scale_factor
            metrics.adjustments_made.append(f"lr *{self.lr_scale_factor}")

        return new_thresholds, new_lr, metrics

    @staticmethod
    def _validate_regulate_inputs(
        firing_rates: np.ndarray[Any, Any],
        thresholds: np.ndarray[Any, Any],
        learning_rate: float,
        weights: list[np.ndarray[Any, Any]] | None,
    ) -> None:
        if not isinstance(firing_rates, np.ndarray) or firing_rates.ndim != 1:
            raise ValueError("regulate firing_rates must be a one-dimensional array")
        if not np.all(np.isfinite(firing_rates)) or np.any(firing_rates < 0.0):
            raise ValueError("regulate firing_rates must be finite and non-negative")
        if not isinstance(thresholds, np.ndarray) or thresholds.ndim != 1:
            raise ValueError("regulate thresholds must be a one-dimensional array")
        if thresholds.shape != firing_rates.shape:
            raise ValueError("regulate thresholds must match firing_rates shape")
        if not np.all(np.isfinite(thresholds)):
            raise ValueError("regulate thresholds must be finite")
        if not np.isfinite(learning_rate) or learning_rate < 0.0:
            raise ValueError("regulate learning_rate must be finite and non-negative")
        if weights is not None:
            for weight in weights:
                if not isinstance(weight, np.ndarray) or not np.all(np.isfinite(weight)):
                    raise ValueError("weights must be finite numpy arrays")


class SleepConsolidation:
    """Sleep-phase synaptic renormalization for memory consolidation.

    During sleep: suppress external input, apply power-law weight decay,
    allow spontaneous replay through recurrent dynamics.

    Reference: Sleep-Based Homeostatic Regularization (arXiv Jan 2026)

    Parameters
    ----------
    decay_exponent : float
        Power-law exponent for weight decay (higher = more aggressive).
    noise_amplitude : float
        Spontaneous activity noise during sleep.
    duration_fraction : float
        Sleep duration as fraction of epoch (0.1 = 10% of time sleeping).
    """

    def __init__(
        self,
        decay_exponent: float = 0.5,
        noise_amplitude: float = 0.01,
        duration_fraction: float = 0.1,
    ):
        if not np.isfinite(decay_exponent) or decay_exponent < 0.0:
            raise ValueError("decay_exponent must be finite and non-negative")
        if not np.isfinite(noise_amplitude) or noise_amplitude < 0.0:
            raise ValueError("noise_amplitude must be finite and non-negative")
        if not np.isfinite(duration_fraction) or not 0.0 < duration_fraction <= 1.0:
            raise ValueError("duration_fraction must be finite and within (0, 1]")

        self.decay_exponent = decay_exponent
        self.noise_amplitude = noise_amplitude
        self.duration_fraction = duration_fraction

    def apply(
        self,
        weights: list[np.ndarray[Any, Any]],
        seed: int = 42,
    ) -> list[np.ndarray[Any, Any]]:
        """Apply sleep consolidation to weights.

        High-activity synapses (large |w|) undergo proportionally more decay.
        Low-activity synapses are relatively preserved.

        Parameters
        ----------
        weights : list of ndarray

        Returns
        -------
        list of ndarray
            Renormalized weights.
        """
        self._validate_weights(weights)

        rng = np.random.RandomState(seed)
        consolidated = []
        for w in weights:
            abs_w = np.abs(w)
            # Power-law decay: larger weights decay more
            max_w = max(abs_w.max(), 1e-8)
            relative = abs_w / max_w
            decay_factor = 1.0 - self.duration_fraction * (relative**self.decay_exponent)
            decay_factor = np.clip(decay_factor, 0.5, 1.0)

            # Apply decay
            w_new = w * decay_factor

            # Add spontaneous replay noise
            w_new += rng.randn(*w.shape) * self.noise_amplitude

            consolidated.append(w_new)
        return consolidated

    def should_sleep(self, epoch: int, total_epochs: int) -> bool:
        """Determine if this epoch should include a sleep phase."""
        if type(epoch) is not int or epoch < 0:
            raise ValueError("epoch must be a non-negative integer")
        if type(total_epochs) is not int or total_epochs <= 0:
            raise ValueError("epoch total_epochs must be a positive integer")

        interval = max(1, int(1.0 / self.duration_fraction))
        return epoch > 0 and epoch % interval == 0

    @staticmethod
    def _validate_weights(weights: list[np.ndarray[Any, Any]]) -> None:
        if not isinstance(weights, list) or len(weights) == 0:
            raise ValueError("weights must be a non-empty list of numpy arrays")
        for weight in weights:
            if not isinstance(weight, np.ndarray):
                raise ValueError("weights must contain only numpy arrays")
            if not np.all(np.isfinite(weight)):
                raise ValueError("weights must be finite")
