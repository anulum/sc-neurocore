# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Spike-based self-supervised learning

"""Contrastive self-supervised learning for SNNs.

SpikeContrastiveLoss: InfoNCE-style loss for spike representations.
CSDPRule: Contrastive Signal-Dependent Plasticity — biologically
plausible local learning rule (Science Advances 2024).

No SNN library ships self-supervised learning utilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


class SpikeContrastiveLoss:
    """InfoNCE contrastive loss adapted for spike representations.

    Computes similarity between spike-rate vectors from two augmented
    views of the same input. Positive pairs = same input, different
    augmentation. Negative pairs = different inputs.

    Parameters
    ----------
    temperature : float
        Contrastive temperature scaling.
    """

    def __init__(self, temperature: float = 0.5):
        self.temperature = temperature

    def compute(
        self,
        view_a: np.ndarray[Any, Any],
        view_b: np.ndarray[Any, Any],
    ) -> float:
        """Compute contrastive loss for a batch of spike-rate pairs.

        Parameters
        ----------
        view_a : ndarray of shape (batch, n_features)
            Spike rates from augmentation A.
        view_b : ndarray of shape (batch, n_features)
            Spike rates from augmentation B.

        Returns
        -------
        float — InfoNCE loss
        """
        batch = view_a.shape[0]
        if batch < 2:
            return 0.0

        # Normalize
        a_norm = view_a / np.clip(np.linalg.norm(view_a, axis=1, keepdims=True), 1e-8, None)
        b_norm = view_b / np.clip(np.linalg.norm(view_b, axis=1, keepdims=True), 1e-8, None)

        # Similarity matrix
        sim = a_norm @ b_norm.T / self.temperature

        # InfoNCE: positive = diagonal, negatives = off-diagonal
        # log softmax along rows
        exp_sim = np.exp(sim - sim.max(axis=1, keepdims=True))
        log_prob = np.log(
            np.clip(
                np.diag(exp_sim) / exp_sim.sum(axis=1),
                1e-10,
                None,
            )
        )
        return -float(log_prob.mean())


@dataclass
class CSDPRule:
    """Contrastive Signal-Dependent Plasticity.

    Local learning rule: weight update depends on (pre, post, contrastive_signal).
    Positive phase: present real data → Hebbian update.
    Negative phase: present corrupted data → anti-Hebbian update.

    Generalizes Forward-Forward to spiking circuits.

    Reference: Ororbia 2024, Science Advances

    Parameters
    ----------
    lr : float
        Learning rate.
    decay : float
        Weight decay for regularization.
    """

    lr: float = 0.01
    decay: float = 0.001

    def positive_update(
        self,
        weights: np.ndarray[Any, Any],
        pre_spikes: np.ndarray[Any, Any],
        post_spikes: np.ndarray[Any, Any],
    ) -> np.ndarray[Any, Any]:
        """Hebbian update from positive (real) data.

        dW = lr * (post @ pre^T) - decay * W
        """
        dW = self.lr * np.outer(post_spikes, pre_spikes) - self.decay * weights
        return weights + dW

    def negative_update(
        self,
        weights: np.ndarray[Any, Any],
        pre_spikes: np.ndarray[Any, Any],
        post_spikes: np.ndarray[Any, Any],
    ) -> np.ndarray[Any, Any]:
        """Anti-Hebbian update from negative (corrupted) data.

        dW = -lr * (post @ pre^T)
        """
        dW = -self.lr * np.outer(post_spikes, pre_spikes)
        return weights + dW

    def contrastive_step(
        self,
        weights: np.ndarray[Any, Any],
        pos_pre: np.ndarray[Any, Any],
        pos_post: np.ndarray[Any, Any],
        neg_pre: np.ndarray[Any, Any],
        neg_post: np.ndarray[Any, Any],
    ) -> np.ndarray[Any, Any]:
        """Full contrastive update: positive + negative phase."""
        w = self.positive_update(weights, pos_pre, pos_post)
        w = self.negative_update(w, neg_pre, neg_post)
        return w

    def goodness(self, activations: np.ndarray[Any, Any]) -> float:
        """Compute 'goodness' score (sum of squared activations).

        Positive data should have high goodness, negative data low.
        """
        return float(np.sum(activations**2))
