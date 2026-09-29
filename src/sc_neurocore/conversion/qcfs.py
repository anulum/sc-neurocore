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

import math

import torch
import torch.nn as nn

from .qcfs_kernel import QCFS_STEP_LIMIT


class QCFSActivation(nn.Module):
    """QCFS activation: quantized clip-floor-shift ReLU replacement.

    For T timesteps and threshold theta:
        QCFS(x) = clip(floor(x * T / theta + 0.5), 0, T) * theta / T

    This quantizes activations to T+1 levels in [0, theta], matching
    the achievable spike rates of an IF neuron over T timesteps.

    Parameters
    ----------
    T : int
        Number of simulation timesteps, ``1 <= T <= 2**32 - 1``: the step
        domain every native counterpart shares.
    theta : float
        Firing threshold (default 1.0).
    learn_theta : bool
        Make threshold trainable (default False).
    """

    def __init__(self, T: int = 8, theta: float = 1.0, learn_theta: bool = False) -> None:
        """Create a positive finite threshold and positive integer rate grid."""
        super().__init__()
        if type(T) is not int or not 1 <= T <= QCFS_STEP_LIMIT:
            raise ValueError("QCFS T must be a positive integer no larger than 2**32 - 1")
        if not math.isfinite(theta) or theta <= 0:
            raise ValueError("QCFS theta must be finite and positive")
        self.T = T
        if learn_theta:
            self.theta = nn.Parameter(torch.tensor(float(theta)))
        else:
            self.register_buffer("theta", torch.tensor(float(theta)))

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
            finite-timestep spike-rate lattice. The backward pass uses the
            identity floor surrogate on the continuously clipped shifted input;
            The two clipping endpoints use zero input derivative. Saturated values
            have zero input derivative, and upper saturation differentiates to
            one with respect to the learned threshold.
        """
        if not bool(torch.isfinite(self.theta) & (self.theta > 0)):
            raise ValueError("QCFS theta must be finite and positive")
        scaled = (x * self.T / self.theta + 0.5).detach()
        interior = (scaled > 0) & (scaled < self.T)
        # Only interior elements carry a gradient. Rebuilding their coordinate
        # from the element alone keeps a saturated infinite or NaN neighbour
        # out of the threshold derivative, where a zero upstream times its
        # infinite quotient would otherwise turn the whole batch into NaN.
        carrier = torch.where(interior, x, torch.zeros_like(x)) * self.T / self.theta + 0.5
        clipped = torch.where(interior, carrier, scaled).clamp(0, self.T)
        # Bu et al., Eq. 17: floor has an identity surrogate derivative.
        quantized = clipped + (clipped.floor() - clipped).detach()
        out: torch.Tensor = quantized * self.theta / self.T
        return out

    def extra_repr(self) -> str:
        """Return the compact PyTorch module representation."""
        return f"T={self.T}, theta={self.theta.item():.2f}"
