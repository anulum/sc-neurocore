# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Learnable spike pattern predictor

"""Online-learnable autoregressive spike predictor for codec integration.

Predicts multi-channel spike patterns from recent history using a linear
autoregressive model trained online via LMS (Least Mean Squares). No
backprop, no gradients, no batches — updates one sample at a time.

For codec integration: encoder and decoder both maintain identical
SpikePredictor instances. Both see the same spike history (encoder uses
actual spikes, decoder recovers actual spikes via XOR then updates).
Deterministic: same history → same prediction → lossless roundtrip.

This is the learnable version of PredictiveWorldModel (LGSSM).
"""

from __future__ import annotations

from dataclasses import dataclass

from typing import Any
import numpy as np


try:
    from sc_neurocore_engine import (
        py_update_spike_world_model as py_update_spike_world_model,
    )
except ImportError:
    py_update_spike_world_model = None


@dataclass
class SpikePredictor:
    """Online autoregressive spike pattern predictor.

    Learns to predict spike[t] from spike[t-K:t] per channel.
    Weight matrix W of shape (N, N*K) maps flattened history to
    per-channel firing probabilities. Binary prediction via threshold.

    Training: LMS update after each timestep.
        W += lr * outer(error, history)
    where error = actual - predicted_prob.

    Parameters
    ----------
    n_channels : int
        Number of spike channels.
    history_len : int
        Number of past timesteps to use as context (K).
    lr : float
        LMS learning rate.
    threshold : float
        Probability threshold for binary prediction.
    seed : int
        RNG seed for weight initialization.
    """

    n_channels: int
    history_len: int = 8
    lr: float = 0.01
    threshold: float = 0.5
    seed: int = 42

    def __post_init__(self) -> None:
        """Seed the RNG and initialise the predictor weights."""
        rng = np.random.RandomState(self.seed)
        n_features = self.n_channels * self.history_len
        # Small random weights — predict from history
        self.W = rng.randn(self.n_channels, n_features) * 0.01
        self.bias = np.zeros(self.n_channels)
        # Circular buffer for history
        self._history = np.zeros((self.history_len, self.n_channels), dtype=np.float64)
        self._t = 0

    def _features(self) -> np.ndarray[Any, Any]:
        """Flatten history buffer into feature vector."""
        # Ordered: oldest first
        indices = [(self._t + i) % self.history_len for i in range(self.history_len)]
        return self._history[indices].ravel()

    def predict_probs(self) -> np.ndarray[Any, Any]:
        """Predict per-channel firing probabilities from history."""
        return self._probabilities(self._features())

    def _probabilities(self, features: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Evaluate the model using an already acquired history vector."""
        logits = self.W @ features + self.bias
        # Sigmoid activation
        probs: np.ndarray[Any, Any] = 1.0 / (1.0 + np.exp(-np.clip(logits, -20, 20)))
        return probs

    def predict(self) -> np.ndarray[Any, Any]:
        """Predict binary spike pattern."""
        return (self.predict_probs() > self.threshold).astype(np.int8)

    def update(self, actual: np.ndarray[Any, Any]) -> None:
        """Update weights with observed spike pattern (LMS rule).

        Parameters
        ----------
        actual : ndarray of shape (n_channels,), binary
        """
        features = self._features()
        self._learn(actual, features, self.predict_probs())

    def _learn(
        self,
        actual: np.ndarray[Any, Any],
        features: np.ndarray[Any, Any],
        probabilities: np.ndarray[Any, Any],
        *,
        use_native: bool = False,
    ) -> None:
        """Update from the exact history and probabilities used to make the prediction."""
        sample = actual.astype(np.float64)
        if use_native and py_update_spike_world_model is not None:
            py_update_spike_world_model(self.W, self.bias, features, probabilities, sample, self.lr)
        else:
            error = sample - probabilities
            self.W += self.lr * np.outer(error, features)
            self.bias += self.lr * error
        self._history[self._t % self.history_len] = sample
        self._t += 1

    def reset(self) -> None:
        """Reset to initial state (same seed → same weights)."""
        self.__post_init__()


def predict_and_xor_world_model(
    spikes: np.ndarray[Any, Any],
    n_channels: int,
    history_len: int = 8,
    lr: float = 0.01,
    threshold: float = 0.5,
    seed: int = 42,
    *,
    use_native: bool = True,
) -> tuple[np.ndarray[Any, Any], int]:
    """World-model predict-XOR loop for codec compression.

    Use the available native weight update when use_native is true.
    Returns the residual raster and correct prediction count.
    """
    T = spikes.shape[0]
    if spikes.size == 0:
        return np.empty_like(spikes), 0
    predictor = SpikePredictor(
        n_channels=n_channels,
        history_len=history_len,
        lr=lr,
        threshold=threshold,
        seed=seed,
    )

    errors = np.empty_like(spikes)
    correct = 0

    for t in range(T):
        features = predictor._features()
        probabilities = predictor._probabilities(features)
        predicted = (probabilities > threshold).astype(np.int8)
        errors[t] = spikes[t] ^ predicted
        correct += n_channels - int(np.count_nonzero(errors[t]))
        predictor._learn(spikes[t], features, probabilities, use_native=use_native)

    return errors, correct


def xor_and_recover_world_model(
    errors: np.ndarray[Any, Any],
    n_channels: int,
    history_len: int = 8,
    lr: float = 0.01,
    threshold: float = 0.5,
    seed: int = 42,
    *,
    use_native: bool = True,
) -> np.ndarray[Any, Any]:
    """Recover world-model residuals with the same optional native weight update."""
    T = errors.shape[0]
    if errors.size == 0:
        return np.empty(errors.shape, dtype=np.int8)
    predictor = SpikePredictor(
        n_channels=n_channels,
        history_len=history_len,
        lr=lr,
        threshold=threshold,
        seed=seed,
    )

    spikes = np.empty((T, errors.shape[1]), dtype=np.int8)

    for t in range(T):
        features = predictor._features()
        probabilities = predictor._probabilities(features)
        predicted = (probabilities > threshold).astype(np.int8)
        actual = errors[t] ^ predicted
        spikes[t] = actual
        predictor._learn(actual, features, probabilities, use_native=use_native)

    return spikes
