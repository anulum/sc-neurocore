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
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _finite_scalar(value: float, name: str, *, positive: bool = False) -> float:
    scalar = float(value)
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    if positive and scalar <= 0.0:
        raise ValueError(f"{name} must be positive")
    if not positive and scalar < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return scalar


def _as_float_matrix(values: np.ndarray[Any, Any], name: str) -> FloatArray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a 2-D array")
    if matrix.shape[1] == 0:
        raise ValueError(f"{name} must contain at least one feature")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values")
    return matrix


def _as_float_vector(values: np.ndarray[Any, Any], name: str) -> FloatArray:
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1:
        raise ValueError(f"{name} must be a 1-D array")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    return vector


def _as_weight_matrix(values: np.ndarray[Any, Any]) -> FloatArray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("weights must be a 2-D array")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("weights must contain only finite values")
    return matrix


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

    def __init__(self, temperature: float = 0.5) -> None:
        self.temperature = _finite_scalar(temperature, "temperature", positive=True)

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
        float
            Mean InfoNCE loss. Single-item batches return ``0.0`` because no
            in-batch negatives are available.

        Raises
        ------
        ValueError
            If either view is not a finite 2-D array with the same shape.
        """
        view_a_matrix = _as_float_matrix(view_a, "view_a")
        view_b_matrix = _as_float_matrix(view_b, "view_b")
        if view_a_matrix.shape != view_b_matrix.shape:
            raise ValueError("view_a and view_b must have the same shape")

        batch = view_a_matrix.shape[0]
        if batch < 2:
            return 0.0

        a_norm = view_a_matrix / np.clip(
            np.linalg.norm(view_a_matrix, axis=1, keepdims=True),
            1e-8,
            None,
        )
        b_norm = view_b_matrix / np.clip(
            np.linalg.norm(view_b_matrix, axis=1, keepdims=True),
            1e-8,
            None,
        )

        sim = a_norm @ b_norm.T / self.temperature
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

    def __post_init__(self) -> None:
        """Validate the scalar learning-rule parameters."""
        self.lr = _finite_scalar(self.lr, "lr")
        self.decay = _finite_scalar(self.decay, "decay")

    def positive_update(
        self,
        weights: np.ndarray[Any, Any],
        pre_spikes: np.ndarray[Any, Any],
        post_spikes: np.ndarray[Any, Any],
    ) -> np.ndarray[Any, Any]:
        """Hebbian update from positive (real) data.

        dW = lr * (post @ pre^T) - decay * W
        """
        weights_matrix, pre_vector, post_vector = self._validate_update_inputs(
            weights,
            pre_spikes,
            post_spikes,
        )
        dW = self.lr * np.outer(post_vector, pre_vector) - self.decay * weights_matrix
        return weights_matrix + dW

    def negative_update(
        self,
        weights: np.ndarray[Any, Any],
        pre_spikes: np.ndarray[Any, Any],
        post_spikes: np.ndarray[Any, Any],
    ) -> np.ndarray[Any, Any]:
        """Anti-Hebbian update from negative (corrupted) data.

        dW = -lr * (post @ pre^T)
        """
        weights_matrix, pre_vector, post_vector = self._validate_update_inputs(
            weights,
            pre_spikes,
            post_spikes,
        )
        dW = -self.lr * np.outer(post_vector, pre_vector)
        return weights_matrix + dW

    def contrastive_step(
        self,
        weights: np.ndarray[Any, Any],
        pos_pre: np.ndarray[Any, Any],
        pos_post: np.ndarray[Any, Any],
        neg_pre: np.ndarray[Any, Any],
        neg_post: np.ndarray[Any, Any],
    ) -> np.ndarray[Any, Any]:
        """Apply one positive phase followed by one negative phase.

        Parameters
        ----------
        weights : ndarray of shape (n_post, n_pre)
            Current synaptic weight matrix.
        pos_pre, neg_pre : ndarray of shape (n_pre,)
            Presynaptic spike-rate vectors for real and corrupted samples.
        pos_post, neg_post : ndarray of shape (n_post,)
            Postsynaptic activation vectors for real and corrupted samples.

        Returns
        -------
        ndarray of shape (n_post, n_pre)
            Updated weights after Hebbian and anti-Hebbian phases.
        """
        w = self.positive_update(weights, pos_pre, pos_post)
        w = self.negative_update(w, neg_pre, neg_post)
        return w

    def goodness(self, activations: np.ndarray[Any, Any]) -> float:
        """Compute 'goodness' score (sum of squared activations).

        Positive data should have high goodness, negative data low.
        """
        values = np.asarray(activations, dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise ValueError("activations must contain only finite values")
        return float(np.sum(values**2))

    @staticmethod
    def _validate_update_inputs(
        weights: np.ndarray[Any, Any],
        pre_spikes: np.ndarray[Any, Any],
        post_spikes: np.ndarray[Any, Any],
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        weights_matrix = _as_weight_matrix(weights)
        pre_vector = _as_float_vector(pre_spikes, "pre_spikes")
        post_vector = _as_float_vector(post_spikes, "post_spikes")
        expected_shape = (post_vector.shape[0], pre_vector.shape[0])
        if weights_matrix.shape != expected_shape:
            raise ValueError("weights must have shape (len(post_spikes), len(pre_spikes))")
        return weights_matrix, pre_vector, post_vector
