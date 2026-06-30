# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — ANN-to-SNN conversion engine

"""Convert trained PyTorch ANNs to rate-coded spiking neural networks.

Two conversion routes are supported, selected automatically from the
activations present in the source model:

Threshold-balancing route (ReLU models, Diehl et al. 2015)
    Replaces ReLU activations with integrate-and-fire (IF) neurons and
    rescales weights so that the calibrated maximum activation maps onto
    the firing threshold. A spike train of rate ``a / theta`` over ``T``
    timesteps approximates the ANN activation ``a``.

QCFS route (QCFS-trained models, Bu et al. 2022)
    When the model carries :class:`~sc_neurocore.conversion.qcfs.QCFSActivation`
    layers, their *learned* per-layer thresholds become the IF thresholds
    directly (no calibration pass) and each IF neuron is initialised to a
    membrane potential of ``theta / 2`` — the optimal shift that cancels
    the quantisation flooring bias. A QCFS-trained ANN then converts to an
    SNN with near-zero accuracy loss at the matching timestep budget.

Pipeline:
    1. Extract weights and biases from the PyTorch model.
    2. Derive per-layer thresholds — from QCFS layers when present,
       otherwise from calibration activation statistics.
    3. Normalise weights so each threshold maps onto unity.
    4. Build an IF-neuron SNN that reproduces the ANN output as spike
       counts over ``T`` timesteps, with the QCFS membrane shift applied
       when converting a QCFS-trained model.

References
----------
Diehl et al. 2015 — "Fast-classifying, high-accuracy spiking deep networks
through weight and threshold balancing".
Bu et al. 2022 — "Optimal ANN-SNN Conversion for High-accuracy and
Ultra-low-latency Spiking Neural Networks" (ICLR).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np

try:
    import torch
    import torch.nn as nn

    from .qcfs import QCFSActivation

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
    initial_membrane_fraction : float
        Fraction of each layer's threshold pre-loaded into the IF membrane
        potential before the first timestep. ``0.0`` reproduces the
        threshold-balancing route; ``0.5`` applies the QCFS optimal shift
        (Bu et al. 2022) that cancels the quantisation flooring bias.
    n_layers : int
        Number of layers.
    """

    weights: list[np.ndarray[Any, Any]]
    biases: list[np.ndarray[Any, Any] | None]
    thresholds: list[float]
    T: int
    initial_membrane_fraction: float = 0.0
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

        # Initialize membrane voltages. The QCFS route pre-loads theta/2 per
        # layer (initial_membrane_fraction == 0.5); the threshold-balancing
        # route starts from rest (0.0).
        voltages = [
            np.full((batch, w.shape[0]), self.initial_membrane_fraction * theta)
            for w, theta in zip(self.weights, self.thresholds)
        ]
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


def _extract_qcfs_layers(model: Any) -> list[tuple[float, int]]:
    """Collect the (theta, T) of each QCFS activation in forward order.

    Parameters
    ----------
    model : nn.Module
        Model possibly containing :class:`QCFSActivation` layers.

    Returns
    -------
    list of (float, int)
        Per-QCFS-layer ``(theta, T)`` pairs in module-traversal order. Empty
        when the model carries no QCFS activations (the ReLU route is used).
    """
    found: list[tuple[float, int]] = []
    for module in model.modules():
        if isinstance(module, QCFSActivation):
            found.append((float(module.theta.item()), int(module.T)))
    return found


def replace_relu_with_qcfs(
    model: Any,
    T: int = 8,
    theta: float = 1.0,
    learn_theta: bool = True,
) -> Any:
    """Swap every ReLU/ReLU6 in a model for a QCFS activation, in place.

    This prepares a trained or fresh ANN for conversion-aware fine-tuning:
    after substitution the network is retrained for a few epochs so the QCFS
    thresholds settle, after which :func:`convert` produces a near-lossless
    SNN (Bu et al. 2022).

    Parameters
    ----------
    model : nn.Module
        Model whose ReLU/ReLU6 activations are replaced. Mutated in place,
        recursing through every submodule.
    T : int
        Quantisation step budget for each inserted QCFS layer.
    theta : float
        Initial firing threshold for each inserted QCFS layer.
    learn_theta : bool
        Whether each inserted threshold is a trainable parameter (the QCFS
        fine-tuning default).

    Returns
    -------
    nn.Module
        The same ``model`` instance, returned for chaining.
    """
    if not HAS_TORCH:
        raise ImportError("PyTorch required for ANN-to-SNN conversion")

    for name, child in model.named_children():
        if isinstance(child, (nn.ReLU, nn.ReLU6)):
            setattr(model, name, QCFSActivation(T=T, theta=theta, learn_theta=learn_theta))
        else:
            replace_relu_with_qcfs(child, T=T, theta=theta, learn_theta=learn_theta)
    return model


def convert(
    model: object,
    calibration_data: object = None,
    T: int | None = None,
    percentile: float = 99.9,
) -> ConvertedSNN:
    """Convert a trained PyTorch ANN to a rate-coded SNN.

    The conversion route is selected from the model's activations: a model
    carrying :class:`QCFSActivation` layers takes the QCFS route (learned
    thresholds, ``theta / 2`` membrane shift, no calibration); any other
    model takes the threshold-balancing route (calibrated or unit
    thresholds, rest-state membrane).

    Parameters
    ----------
    model : nn.Module
        Trained PyTorch model with Linear/Conv2d layers and either ReLU or
        QCFS activations.
    calibration_data : Tensor, optional
        Sample input batch for threshold calibration on the ReLU route. If
        None, the ReLU route uses a default threshold of 1.0 per layer.
        Ignored on the QCFS route, whose thresholds are already learned.
    T : int, optional
        Number of simulation timesteps (higher = more accurate, slower). If
        None, the QCFS route adopts the layers' trained step budget and the
        ReLU route defaults to 16.
    percentile : float
        Activation percentile for threshold normalization on the ReLU route.

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

    qcfs_layers = _extract_qcfs_layers(model)
    if qcfs_layers:
        # QCFS route (Bu et al. 2022): the learned per-layer theta is the
        # threshold and theta/2 is the optimal initial membrane potential.
        thresholds = [theta for theta, _ in qcfs_layers]
        # Pad if fewer QCFS layers than weight layers (e.g. no output QCFS).
        while len(thresholds) < len(weights):
            thresholds.append(1.0)
        initial_membrane_fraction = 0.5
        if T is None:
            T = qcfs_layers[0][1]
    else:
        # Threshold-balancing route (Diehl et al. 2015).
        initial_membrane_fraction = 0.0
        if calibration_data is not None:
            max_acts = _compute_max_activations(
                model, cast(torch.Tensor, calibration_data), percentile
            )
            # Pad if fewer ReLUs than Linear layers
            while len(max_acts) < len(weights):
                max_acts.append(1.0)
            thresholds = max_acts
        else:
            thresholds = [1.0] * len(weights)
        if T is None:
            T = 16

    # Normalize weights: scale so that each threshold maps to unity.
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
        initial_membrane_fraction=initial_membrane_fraction,
    )
