# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — QCFS activation for ANN-to-SNN conversion

"""QCFS (Quantization-Clip-Floor-Shift) activation function.

Replaces ReLU in the ANN during conversion-aware training or post-hoc
conversion. QCFS approximates the rate-coded SNN firing rate as a
quantized step function, minimizing conversion error.

Reference: Bu et al. 2022 — "Optimal ANN-SNN Conversion for
High-accuracy and Ultra-low-latency Spiking Neural Networks"
"""

from __future__ import annotations

import torch
import torch.nn as nn


class QCFSActivation(nn.Module):
    """QCFS activation: quantized clip-floor-shift ReLU replacement.

    For T timesteps and threshold theta:
        QCFS(x) = clip(floor(x * T / theta + 0.5), 0, T) * theta / T

    This quantizes activations to T+1 levels in [0, theta], matching
    the achievable spike rates of an IF neuron over T timesteps.

    Parameters
    ----------
    T : int
        Number of simulation timesteps.
    theta : float
        Firing threshold (default 1.0).
    learn_theta : bool
        Make threshold trainable (default False).
    """

    def __init__(self, T: int = 8, theta: float = 1.0, learn_theta: bool = False) -> None:
        super().__init__()
        self.T = T
        if learn_theta:
            self.theta = nn.Parameter(torch.tensor(theta))
        else:
            self.register_buffer("theta", torch.tensor(theta))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Quantise activations to the spike-rate grid with a straight-through gradient.

        Parameters
        ----------
        x : torch.Tensor
            ANN activation tensor to clip and quantise into ``T + 1`` rate levels.

        Returns
        -------
        torch.Tensor
            Tensor with values clipped to ``[0, theta]`` and quantised to the
            finite-timestep spike-rate lattice.
        """
        scaled = x * self.T / self.theta + 0.5
        # STE: floor in forward, pass gradient straight through
        quantized = scaled.floor() - (scaled.floor() - scaled).detach()
        clipped = quantized.clamp(0, self.T)
        out: torch.Tensor = clipped * self.theta / self.T
        return out

    def extra_repr(self) -> str:
        """Return the compact PyTorch module representation."""
        return f"T={self.T}, theta={self.theta.item():.2f}"
