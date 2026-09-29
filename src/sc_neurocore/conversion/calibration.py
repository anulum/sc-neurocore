# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Torch activation calibration and model custody

"""Measure inference activation thresholds without retaining model mutations."""

import numpy as np
import torch
from torch import nn
from torch.utils.hooks import RemovableHandle

from .random_custody import preserved_random_state


def calibrate_activation_thresholds(
    model: nn.Module, calibration_data: torch.Tensor, percentile: float = 99.9
) -> list[float]:
    """Measure each ReLU's inference activation percentile in forward order.

    Parameters
    ----------
    model : nn.Module
        Source ANN. Its original per-module training modes and user hooks are
        retained on success and failure.
    calibration_data : torch.Tensor
        Nonempty finite input tensor passed through the actual source network.
    percentile : float
        Activation percentile in the closed interval ``[0, 100]``.

    Returns
    -------
    list of float
        Per-invocation activation scales, floored at ``1e-6`` for silent ReLUs.

    Raises
    ------
    ValueError
        If the input, percentile or measured activations are invalid.
    """
    if not 0 <= percentile <= 100:
        raise ValueError("calibration percentile must be between zero and 100")
    if calibration_data.numel() == 0 or not bool(torch.isfinite(calibration_data).all()):
        raise ValueError("calibration data must be nonempty and finite")
    modes = [(module, module.training) for module in model.modules()]
    hooks: list[RemovableHandle] = []
    activations: list[torch.Tensor] = []

    def record(module: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        """Retain the actual source ReLU output for its activation statistic."""
        activations.append(output.detach().cpu().clone())

    try:
        for module, _ in modes:
            if isinstance(module, (nn.ReLU, nn.ReLU6)):
                hooks.append(module.register_forward_hook(record))
        model.eval()
        with preserved_random_state(), torch.no_grad():
            model(calibration_data)
    finally:
        for handle in hooks:
            handle.remove()
        for module, training in modes:
            module.training = training

    scales: list[float] = []
    for activation in activations:
        if not bool(torch.isfinite(activation).all()):
            raise ValueError("calibration activation must be finite")
        scale = float(np.percentile(activation.double().numpy(), percentile))
        scales.append(max(scale, 1e-6))
    return scales
