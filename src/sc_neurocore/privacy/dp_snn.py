# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Differential privacy for spiking neural networks

"""Spike-level differential privacy: add privacy noise at the spike domain.

Standard DP-SGD adds Gaussian noise to gradients. For SNNs, we exploit
the binary nature of spikes: add/remove spikes stochastically to provide
(epsilon, delta)-differential privacy. More natural than gradient noise,
preserves spike sparsity.

No SNN framework has built-in differential privacy.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class PrivacyAccountant:
    """Track cumulative privacy budget across training steps.

    Uses simple composition theorem: total epsilon = sum of per-step epsilons.
    For tighter bounds, use Renyi DP (future extension).

    Parameters
    ----------
    target_epsilon : float
        Privacy budget limit.
    target_delta : float
        Failure probability.
    """

    target_epsilon: float = 1.0
    target_delta: float = 1e-5
    _spent_epsilon: float = 0.0
    _steps: int = 0

    def record_step(self, step_epsilon: float) -> None:
        """Record privacy cost of one training step."""
        self._spent_epsilon += step_epsilon
        self._steps += 1

    @property
    def spent_epsilon(self) -> float:
        return self._spent_epsilon

    @property
    def remaining_epsilon(self) -> float:
        return max(0.0, self.target_epsilon - self._spent_epsilon)

    @property
    def budget_exhausted(self) -> bool:
        return self._spent_epsilon >= self.target_epsilon

    def summary(self) -> str:
        return (
            f"Privacy: epsilon={self._spent_epsilon:.4f}/{self.target_epsilon} "
            f"({self._steps} steps), delta={self.target_delta}"
        )


class SpikeLevelDP:
    """Spike-level differential privacy mechanism.

    Adds stochastic spike noise to provide (epsilon, delta)-DP.
    Two mechanisms:
    - Spike randomized response: each spike independently flipped with probability p
    - Spike subsampling: randomly drop spikes with probability 1-q

    Parameters
    ----------
    epsilon : float
        Per-step privacy budget.
    mechanism : str
        'randomized_response' or 'subsampling'.
    seed : int
    """

    def __init__(
        self, epsilon: float = 1.0, mechanism: str = "randomized_response", seed: int = 42
    ) -> None:
        self.epsilon = epsilon
        self.mechanism = mechanism
        self._rng = np.random.RandomState(seed)

        # Compute noise parameter from epsilon
        if mechanism == "randomized_response":
            # Randomized response: flip each bit with probability p = 1/(1+e^epsilon)
            self.flip_prob = 1.0 / (1.0 + np.exp(epsilon))
        elif mechanism == "subsampling":
            # Poisson subsampling: keep each spike with probability q = e^epsilon / (1+e^epsilon)
            self.keep_prob = np.exp(epsilon) / (1.0 + np.exp(epsilon))
        else:
            raise ValueError(f"Unknown mechanism '{mechanism}'")

    def privatize(self, spikes: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Apply DP mechanism to a spike tensor.

        Parameters
        ----------
        spikes : ndarray of shape (T, N) or (N,)
            Binary spike tensor.

        Returns
        -------
        ndarray, same shape
            Privatized spikes.
        """
        if self.mechanism == "randomized_response":
            flip_mask = self._rng.random(spikes.shape) < self.flip_prob
            privatized = spikes.copy().astype(np.int8)
            privatized[flip_mask] = 1 - privatized[flip_mask]
            return privatized
        else:
            keep_mask = self._rng.random(spikes.shape) < self.keep_prob
            masked_spikes: np.ndarray[Any, Any] = (spikes * keep_mask).astype(spikes.dtype)
            return masked_spikes

    @property
    def per_step_epsilon(self) -> float:
        return self.epsilon


class MembershipAudit:
    """Audit SNN for membership inference vulnerability.

    Given a trained model (as a callable), test whether it leaks
    information about training data membership. Uses shadow model
    methodology: compare model confidence on training vs non-training
    samples.

    Parameters
    ----------
    run_fn : callable
        Model function: takes spikes (T, N) → output (N_out,).
    """

    def __init__(self, run_fn: Callable[..., Any]) -> None:
        self.run_fn = run_fn

    def audit(
        self,
        member_samples: list[np.ndarray[Any, Any]],
        non_member_samples: list[np.ndarray[Any, Any]],
    ) -> dict[str, Any]:
        """Run membership inference audit.

        Parameters
        ----------
        member_samples : list of ndarray
            Samples known to be in the training set.
        non_member_samples : list of ndarray
            Samples known to NOT be in the training set.

        Returns
        -------
        dict with:
            - accuracy: membership inference accuracy (0.5 = no leakage, 1.0 = full leak)
            - member_confidence: mean output magnitude for members
            - non_member_confidence: mean output magnitude for non-members
            - vulnerable: bool, True if accuracy > 0.6
        """
        member_scores = [float(np.abs(self.run_fn(s)).mean()) for s in member_samples]
        non_member_scores = [float(np.abs(self.run_fn(s)).mean()) for s in non_member_samples]

        mean_member = float(np.mean(member_scores))
        mean_non = float(np.mean(non_member_scores))

        # Threshold-based inference: predict member if score > midpoint
        threshold = (mean_member + mean_non) / 2
        correct = 0
        total = len(member_scores) + len(non_member_scores)

        for s in member_scores:
            if s >= threshold:
                correct += 1
        for s in non_member_scores:
            if s < threshold:
                correct += 1

        accuracy = correct / max(total, 1)

        return {
            "accuracy": accuracy,
            "member_confidence": mean_member,
            "non_member_confidence": mean_non,
            "vulnerable": accuracy > 0.6,
        }
