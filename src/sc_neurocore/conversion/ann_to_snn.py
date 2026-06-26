# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — ANN-to-SNN conversion engine

"""Convert trained PyTorch ANNs to rate-coded spiking neural networks.

The conversion replaces ReLU activations with IF (integrate-and-fire)
neurons and uses weight/threshold normalization to preserve accuracy.
Rate coding: ANN activation a maps to spike rate a/theta over T steps.

Pipeline:
    1. Extract weights and biases from PyTorch Sequential model
    2. Compute per-layer activation statistics (max, percentile)
    3. Normalize weights so that max activation = threshold
    4. Build an SNN with IF neurons that reproduces the ANN output
       as spike counts over T timesteps

Reference: Diehl et al. 2015 — "Fast-classifying, high-accuracy spiking
deep networks through weight and threshold balancing"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np

try:
    import torch
    import torch.nn as nn

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@dataclass
class ConvertedSNN:
    """Rate-coded SNN converted from an ANN.

    Attributes
    ----------
    weights : list of ndarray
        Per-layer weight matrices.
    biases : list of ndarray or None
        Per-layer biases (None if absent).
    thresholds : list of float
        Per-layer firing thresholds after normalization.
    T : int
        Number of simulation timesteps.
    n_layers : int
        Number of layers.
    """

    weights: list[np.ndarray[Any, Any]]
    biases: list[np.ndarray[Any, Any] | None]
    thresholds: list[float]
    T: int
    n_layers: int = field(init=False)

    def __post_init__(self) -> None:
        """Derive the layer count from the converted weight stack."""
        self.n_layers = len(self.weights)

    def run(self, x: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Run the converted SNN for T timesteps on input x.

        Parameters
        ----------
        x : ndarray of shape (n_input,) or (batch, n_input)
            Input values in [0, 1]. Converted to Poisson spike trains.

        Returns
        -------
        ndarray of shape (n_output,) or (batch, n_output)
            Output spike counts over T timesteps (unnormalized).
        """
        squeeze = x.ndim == 1
        if squeeze:
            x = x[np.newaxis]

        batch = x.shape[0]
        rng = np.random.RandomState(42)

        # Initialize membrane voltages
        voltages = [np.zeros((batch, w.shape[0])) for w in self.weights]
        spike_counts = np.zeros((batch, self.weights[-1].shape[0]))

        for t in range(self.T):
            # Rate-code input: spike with probability proportional to x
            input_spikes = (rng.random(x.shape) < x).astype(np.float64)

            layer_input = input_spikes
            for i, (w, b, theta) in enumerate(zip(self.weights, self.biases, self.thresholds)):
                current = layer_input @ w.T
                if b is not None:
                    current += b / self.T
                voltages[i] += current
                spikes = (voltages[i] >= theta).astype(np.float64)
                voltages[i] -= spikes * theta
                layer_input = spikes

                if i == self.n_layers - 1:
                    spike_counts += spikes

        if squeeze:
            spike_counts = spike_counts[0]
        return spike_counts

    def classify(self, x: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Run SNN and return predicted class indices."""
        counts = self.run(x)
        predictions: np.ndarray[Any, Any] = np.argmax(counts, axis=-1)
        return predictions


def _extract_layers(model: Any) -> list[tuple[np.ndarray[Any, Any], np.ndarray[Any, Any] | None]]:
    """Extract (weight, bias) pairs from a PyTorch Sequential model."""
    layers = []
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            w = module.weight.detach().cpu().numpy()
            b = module.bias.detach().cpu().numpy() if module.bias is not None else None
            layers.append((w, b))
    return layers


def _compute_max_activations(
    model: Any, calibration_data: torch.Tensor, percentile: float = 99.9
) -> list[float]:
    """Run calibration data through model, record per-layer max activation."""
    maxes = []
    hooks = []
    activations = []

    def hook_fn(module: Any, inp: Any, out: Any) -> None:
        activations.append(out.detach().cpu())

    for module in model.modules():
        if isinstance(module, (nn.ReLU, nn.ReLU6)):
            hooks.append(module.register_forward_hook(hook_fn))

    with torch.no_grad():
        model(calibration_data)

    for h in hooks:
        h.remove()

    for act in activations:
        val = float(np.percentile(act.numpy(), percentile))
        maxes.append(max(val, 1e-6))

    return maxes


def convert(
    model: object,
    calibration_data: object = None,
    T: int = 16,
    percentile: float = 99.9,
) -> ConvertedSNN:
    """Convert a trained PyTorch ANN to a rate-coded SNN.

    Parameters
    ----------
    model : nn.Module
        Trained PyTorch model (Sequential with Linear + ReLU).
    calibration_data : Tensor, optional
        Sample input batch for threshold calibration. If None, uses
        default threshold of 1.0 per layer.
    T : int
        Number of simulation timesteps (higher = more accurate, slower).
    percentile : float
        Activation percentile for threshold normalization.

    Returns
    -------
    ConvertedSNN
        Converted spiking network ready to run.
    """
    if not HAS_TORCH:
        raise ImportError("PyTorch required for ANN-to-SNN conversion")

    layers = _extract_layers(model)
    if not layers:
        raise ValueError("No Linear/Conv2d layers found in model")

    weights = [w for w, _ in layers]
    biases = [b for _, b in layers]

    if calibration_data is not None:
        max_acts = _compute_max_activations(model, cast(torch.Tensor, calibration_data), percentile)
        # Pad if fewer ReLUs than Linear layers
        while len(max_acts) < len(weights):
            max_acts.append(1.0)
        thresholds = max_acts
    else:
        thresholds = [1.0] * len(weights)

    # Normalize weights: scale so that max activation maps to threshold
    normalized_weights = []
    prev_scale = 1.0
    for i, (w, theta) in enumerate(zip(weights, thresholds)):
        scale = theta / prev_scale if i > 0 else theta
        normalized_weights.append(w / scale)
        prev_scale = theta

    return ConvertedSNN(
        weights=normalized_weights,
        biases=biases,
        thresholds=[1.0] * len(weights),
        T=T,
    )
