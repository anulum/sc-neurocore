# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Equilibrium Propagation (EP) prototype
#
# Reference implementation of Scellier & Bengio 2017 — a
# backpropagation-free learning algorithm suitable for analogue
# and stochastic computing hardware.
#
# EP replaces the error-backpropagation chain rule with a
# two-phase settle-and-nudge protocol:
#   Phase 1 (Free):  Network settles to energy minimum → s*
#   Phase 2 (Nudge):  Output is nudged toward target → s_β
#   Weight update:  ΔW ∝ (s_β · s_βᵀ − s* · s*ᵀ) / β
#
# Advantages for SC hardware:
#   - No backward pass (no gradient routing network needed)
#   - Only local Hebbian-like products (naturally SC-compatible)
#   - β can be implemented as a small analog perturbation
#
# Reference: Scellier, B. & Bengio, Y. (2017). Front. Comp. Neurosci. 11:24.
#            DOI: 10.3389/fncom.2017.00024

"""Equilibrium Propagation — backprop-free learning for SC hardware.

This is a research prototype for evaluating EP feasibility in the
SC-NeuroCore stochastic computing pipeline. It is NOT integrated
into the main compiler yet.

Usage::

    from sc_neurocore.training.equilibrium_propagation import EPNetwork

    net = EPNetwork(layer_sizes=[784, 500, 10])
    net.train(x_batch, y_batch, beta=1.0, lr=0.01, n_settle=20)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def _rho(x: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Hard-sigmoid activation (hardware-friendly, no exp)."""
    return np.clip(x, 0.0, 1.0)


def _rho_prime(x: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Derivative of hard-sigmoid."""
    return np.where((x > 0.0) & (x < 1.0), 1.0, 0.0)


class EPNetwork:
    """Multi-layer Equilibrium Propagation network.

    Parameters
    ----------
    layer_sizes : list of int
        Number of units per layer (including input and output).
    rng_seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        layer_sizes: list[int],
        rng_seed: int = 42,
    ) -> None:
        self.layer_sizes = layer_sizes
        self.n_layers = len(layer_sizes)
        self.rng = np.random.default_rng(rng_seed)

        # Xavier initialisation
        self.weights: list[np.ndarray[Any, Any]] = []
        self.biases: list[np.ndarray[Any, Any]] = []
        for i in range(self.n_layers - 1):
            fan_in = layer_sizes[i]
            fan_out = layer_sizes[i + 1]
            scale = np.sqrt(2.0 / (fan_in + fan_out))
            self.weights.append(self.rng.normal(0, scale, (fan_in, fan_out)).astype(np.float64))
            self.biases.append(np.zeros(fan_out, dtype=np.float64))

        logger.info(
            "EPNetwork: %s (%d params)",
            " → ".join(str(s) for s in layer_sizes),
            sum(w.size + b.size for w, b in zip(self.weights, self.biases)),
        )

    def _energy(self, states: list[np.ndarray[Any, Any]]) -> float:
        """Compute the Hopfield energy of the network."""
        E = 0.0
        for i in range(self.n_layers - 1):
            E -= float(np.sum(_rho(states[i]) @ self.weights[i] * _rho(states[i + 1])))
            E -= float(np.sum(self.biases[i] * _rho(states[i + 1])))
        return E

    def _settle(
        self,
        x: np.ndarray[Any, Any],
        *,
        n_steps: int = 20,
        epsilon: float = 0.5,
        beta: float = 0.0,
        target: np.ndarray[Any, Any] | None = None,
    ) -> list[np.ndarray[Any, Any]]:
        """Settle the network to (near) equilibrium.

        Parameters
        ----------
        x : array, shape (n_input,)
            Input pattern (clamped).
        n_steps : int
            Number of relaxation steps.
        epsilon : float
            Integration step size.
        beta : float
            Nudging strength (0 = free phase, >0 = nudged phase).
        target : array, optional
            Target output (required when beta > 0).
        """
        # Initialise states
        states: list[np.ndarray[Any, Any]] = [x.copy()]
        for i in range(1, self.n_layers):
            states.append(np.zeros(self.layer_sizes[i], dtype=np.float64))

        for _step in range(n_steps):
            for i in range(1, self.n_layers):
                # Gradient of internal energy w.r.t. s_i
                grad = self.biases[i - 1].copy()
                grad += _rho(states[i - 1]) @ self.weights[i - 1]
                if i < self.n_layers - 1:
                    grad += self.weights[i] @ _rho(states[i + 1])

                # Nudging term for output layer
                if beta > 0 and i == self.n_layers - 1 and target is not None:
                    grad += beta * (target - states[i])

                # Update state
                states[i] += epsilon * (-states[i] + _rho_prime(states[i]) * grad)

        return states

    def train(
        self,
        x_batch: np.ndarray[Any, Any],
        y_batch: np.ndarray[Any, Any],
        *,
        beta: float = 1.0,
        lr: float = 0.01,
        n_settle: int = 20,
        epsilon: float = 0.5,
    ) -> float:
        """Train on a mini-batch using the EP two-phase protocol.

        Parameters
        ----------
        x_batch : array, shape (batch, n_input)
        y_batch : array, shape (batch, n_output)
        beta : float
            Nudging strength.
        lr : float
            Learning rate.
        n_settle : int
            Steps per phase.
        epsilon : float
            Relaxation step size.

        Returns
        -------
        float
            Mean squared error over the batch.
        """
        batch_size = x_batch.shape[0]
        total_mse = 0.0

        # Accumulate weight/bias deltas
        dW = [np.zeros_like(w) for w in self.weights]
        dB = [np.zeros_like(b) for b in self.biases]

        for b_idx in range(batch_size):
            x = x_batch[b_idx]
            y = y_batch[b_idx]

            # Free phase
            s_free = self._settle(x, n_steps=n_settle, epsilon=epsilon, beta=0.0)

            # Nudged phase
            s_nudge = self._settle(x, n_steps=n_settle, epsilon=epsilon, beta=beta, target=y)

            # Contrastive Hebbian update: ΔW ∝ (nudge - free) / β
            for i in range(self.n_layers - 1):
                free_pre = _rho(s_free[i])
                free_post = _rho(s_free[i + 1])
                nudge_pre = _rho(s_nudge[i])
                nudge_post = _rho(s_nudge[i + 1])

                dW[i] += np.outer(nudge_pre, nudge_post) - np.outer(free_pre, free_post)
                dB[i] += nudge_post - free_post

            # MSE
            output = _rho(s_free[-1])
            total_mse += float(np.mean((output - y) ** 2))

        # Apply updates
        for i in range(self.n_layers - 1):
            self.weights[i] += lr / (beta * batch_size) * dW[i]
            self.biases[i] += lr / (beta * batch_size) * dB[i]

        return float(total_mse / batch_size)

    def predict(self, x: np.ndarray[Any, Any], n_settle: int = 20) -> np.ndarray[Any, Any]:
        """Predict output by settling in free phase."""
        states = self._settle(x, n_steps=n_settle, beta=0.0)
        return _rho(states[-1])

    def get_params(self) -> dict[str, Any]:
        """Return serialisable parameter dict."""
        return {
            "layer_sizes": self.layer_sizes,
            "weights": [w.tolist() for w in self.weights],
            "biases": [b.tolist() for b in self.biases],
        }
