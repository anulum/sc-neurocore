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


_MAX_RANDOM_SEED = 2**32 - 1


def _validate_real_scalar(
    name: str,
    value: object,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    lower_open: bool = False,
    upper_open: bool = False,
) -> float:
    """Return a finite real scalar after rejecting bool aliases and bounds drift."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{name} must be a finite real scalar")

    scalar = float(value)
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and (scalar <= minimum if lower_open else scalar < minimum):
        qualifier = "greater than" if lower_open else "at least"
        raise ValueError(f"{name} must be {qualifier} {minimum:g}")
    if maximum is not None and (scalar >= maximum if upper_open else scalar > maximum):
        qualifier = "less than" if upper_open else "at most"
        raise ValueError(f"{name} must be {qualifier} {maximum:g}")
    return scalar


def _validate_finite_numeric_array(
    name: str,
    value: object,
    *,
    ndim: int | None = None,
    non_empty: bool = False,
    non_negative: bool = False,
) -> np.ndarray[Any, Any]:
    """Return a finite numeric array after enforcing shape and value contracts."""
    if not isinstance(value, np.ndarray):
        raise ValueError(f"{name} must be a numpy array")
    if ndim is not None and value.ndim != ndim:
        raise ValueError(f"{name} must be a {ndim}-dimensional array")
    if non_empty and value.size == 0:
        raise ValueError(f"{name} must be a non-empty array")
    if np.issubdtype(value.dtype, np.bool_) or not np.issubdtype(value.dtype, np.number):
        raise ValueError(f"{name} must be a finite numeric array")
    if not bool(np.all(np.isfinite(value))):
        raise ValueError(f"{name} must be finite")
    if non_negative and bool(np.any(value < 0.0)):
        raise ValueError(f"{name} must be non-negative")
    return value


def _max_abs_weight(weight: np.ndarray[Any, Any]) -> float:
    """Return the maximum absolute weight without relying on ndarray reductions."""
    return max(float(abs(value)) for value in weight.flat)


def _validate_seed(seed: int) -> int:
    """Return a NumPy RandomState seed after rejecting bool aliases and wraparound."""
    if type(seed) is not int or not 0 <= seed <= _MAX_RANDOM_SEED:
        raise ValueError("seed must be an integer in [0, 4294967295]")
    return seed


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
        """Render a multi-line human-readable network-stability report.

        Returns
        -------
        str
            Text report containing stability status, firing-rate statistics,
            E/I ratio, weight norm, and any applied regulation actions.
        """
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
    ) -> None:
        self.target_rate = _validate_real_scalar("target_rate", target_rate, minimum=0.0)
        self.rate_tolerance = _validate_real_scalar(
            "rate_tolerance",
            rate_tolerance,
            minimum=0.0,
            maximum=1.0,
        )
        self.threshold_step = _validate_real_scalar(
            "threshold_step",
            threshold_step,
            minimum=0.0,
        )
        self.lr_scale_factor = _validate_real_scalar(
            "lr_scale_factor",
            lr_scale_factor,
            minimum=0.0,
            maximum=1.0,
            lower_open=True,
        )

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
            norms = [float(np.linalg.norm(w)) for w in weights]
            metrics.weight_norm = float(np.mean(norms))

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
        _validate_finite_numeric_array(
            "regulate firing_rates",
            firing_rates,
            ndim=1,
            non_empty=True,
            non_negative=True,
        )
        _validate_finite_numeric_array(
            "regulate thresholds",
            thresholds,
            ndim=1,
            non_empty=True,
        )
        if thresholds.shape != firing_rates.shape:
            raise ValueError("regulate thresholds must match firing_rates shape")
        _validate_real_scalar("regulate learning_rate", learning_rate, minimum=0.0)
        if weights is not None:
            for weight in weights:
                _validate_finite_numeric_array(
                    "weights",
                    weight,
                    non_empty=True,
                )


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
    ) -> None:
        self.decay_exponent = _validate_real_scalar(
            "decay_exponent",
            decay_exponent,
            minimum=0.0,
        )
        self.noise_amplitude = _validate_real_scalar(
            "noise_amplitude",
            noise_amplitude,
            minimum=0.0,
        )
        self.duration_fraction = _validate_real_scalar(
            "duration_fraction",
            duration_fraction,
            minimum=0.0,
            maximum=1.0,
            lower_open=True,
        )

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
            Non-empty finite numeric weight arrays.
        seed : int, default=42
            Deterministic NumPy ``RandomState`` seed in ``[0, 2**32 - 1]``.

        Returns
        -------
        list of ndarray
            Renormalized weights.
        """
        self._validate_weights(weights)
        rng = np.random.RandomState(_validate_seed(seed))

        consolidated = []
        for w in weights:
            abs_w = np.abs(w)
            # Power-law decay: larger weights decay more
            max_w = max(_max_abs_weight(w), 1e-8)
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
        """Determine if this epoch should include a sleep phase.

        Parameters
        ----------
        epoch : int
            Zero-based epoch index.
        total_epochs : int
            Positive total epoch count for caller-side schedule validation.

        Returns
        -------
        bool
            ``True`` when the epoch is a positive multiple of the interval
            implied by ``duration_fraction``.
        """
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
            if isinstance(weight, np.ndarray) and weight.size == 0:
                raise ValueError("weights must contain non-empty arrays")
            _validate_finite_numeric_array("weights", weight, non_empty=True)
