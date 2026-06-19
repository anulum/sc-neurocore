# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — DelayLinear: trainable per-synapse delays

"""Learnable-delay linear layer for surrogate gradient SNN training.

Implements the DCLS (Dilated Convolutions with Learnable Spacings)
principle applied to fully-connected SNN layers. Each synapse has a
trainable weight AND a trainable delay. During forward pass, the input
spike history is queried at fractional delay positions via linear
interpolation, making delays differentiable.

References:
    Hammouamri et al. 2023 — "Learning Delays in SNNs using DCLS"
    Göltz et al. 2025 — "DelGrad: exact gradients for delays on BrainScaleS-2"
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class DelayLinear(nn.Module):
    """Linear layer with trainable per-synapse delays.

    Parameters
    ----------
    in_features : int
        Number of input neurons.
    out_features : int
        Number of output neurons.
    max_delay : int
        Maximum delay in timesteps. Delay buffer stores this many steps.
    bias : bool
        Include bias term (default False for SNN).
    learn_delay : bool
        Make delays trainable (default True).
    init_delay : float
        Initial delay value for all synapses (default 1.0).

    Forward pass
    -------------
    Call ``step(input_spikes)`` at each timestep. The module maintains
    an internal spike history buffer. For each synapse (i, j):

        delayed_input[j] = sum_i W[j,i] * interp(history, t - D[j,i])

    where interp linearly interpolates between integer delay bins,
    making D differentiable.

    Export
    ------
    ``delays_int`` returns quantized integer delays for hardware deployment.
    ``to_nir_delay_array()`` returns delays in the CSR format expected by
    ``Projection(delay=array)``.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        max_delay: int = 16,
        bias: bool = False,
        learn_delay: bool = True,
        init_delay: float = 1.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.max_delay = max_delay

        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        # Delays stored as continuous values in [0, max_delay)
        delay_init = torch.full((out_features, in_features), init_delay)
        if learn_delay:
            self.delay = nn.Parameter(delay_init)
        else:
            self.register_buffer("delay", delay_init)

        # Spike history buffer: (max_delay + 1, in_features)
        self.register_buffer("_history", torch.zeros(max_delay + 1, in_features))
        self._t = 0

    def reset(self) -> None:
        """Clear spike history. Call between sequences."""
        self._history.zero_()  # type: ignore[operator]
        self._t = 0

    def step(self, x: torch.Tensor) -> torch.Tensor:
        """Process one timestep.

        Parameters
        ----------
        x : Tensor of shape (batch, in_features) or (in_features,)
            Input spikes (binary or continuous).

        Returns
        -------
        Tensor of shape (batch, out_features) or (out_features,)
            Weighted, delayed input current.
        """
        squeeze = x.dim() == 1
        if squeeze:
            x = x.unsqueeze(0)

        batch_size = x.shape[0]
        buf_len = self.max_delay + 1

        # Store current input in history (use first batch element for buffer)
        write_idx = self._t % buf_len
        self._history[write_idx] = x[0].detach()  # type: ignore[operator]

        # Clamp delays to valid range
        d = self.delay.clamp(0, self.max_delay - 1e-6)

        # Integer floor and ceil indices
        d_floor = d.long()
        d_ceil = (d_floor + 1).clamp(max=self.max_delay)
        frac = d - d_floor.float()

        # Read from history at delayed positions
        # idx_floor[j, i] = (current_t - d_floor[j, i]) % buf_len
        idx_floor = (self._t - d_floor) % buf_len
        idx_ceil = (self._t - d_ceil) % buf_len

        # Gather delayed spikes via linear interpolation
        # history shape: (buf_len, in_features)
        # We need history[idx[j,i], i] for each (j, i)
        hist_floor = self._history[idx_floor, torch.arange(self.in_features).unsqueeze(0)]  # type: ignore[index]
        hist_ceil = self._history[idx_ceil, torch.arange(self.in_features).unsqueeze(0)]  # type: ignore[index]
        delayed_x = (1 - frac) * hist_floor + frac * hist_ceil

        # Weighted sum: out[j] = sum_i W[j,i] * delayed_x[j,i]
        output = (self.weight * delayed_x).sum(dim=1)
        if self.bias is not None:
            output = output + self.bias

        self._t += 1

        # Broadcast to batch
        output = output.unsqueeze(0).expand(batch_size, -1)
        if squeeze:
            output = output.squeeze(0)
        result: torch.Tensor = output
        return result

    @property
    def delays_int(self) -> torch.Tensor:
        """Quantized integer delays for hardware export."""
        with torch.no_grad():
            return self.delay.clamp(0, self.max_delay).round().long()

    def to_nir_delay_array(self) -> Any:
        """Export delays as flat array matching CSR data order.

        Returns delays in the same order as weights flattened row-major,
        suitable for ``Projection(delay=array)``.
        """
        import numpy as np

        return self.delays_int.detach().cpu().numpy().flatten().astype(np.float64)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"max_delay={self.max_delay}, learn_delay={isinstance(self.delay, nn.Parameter)}"
        )
