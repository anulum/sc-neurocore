# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — PACT parameterised clipping activation

"""PACT: PArameterized Clipping acTivation for quantisation-aware training.

PACT replaces the unbounded ReLU with a clipping activation whose upper bound
``alpha`` is a trainable parameter, so the network learns the activation range
that minimises quantisation error instead of relying on a fixed clip:

    y = clip(x, 0, alpha)          (parameterised clip)
    y_q = round(y / s) * s,  s = alpha / (2**n_bits - 1)   (uniform quantise)

The clip gradient flows to ``alpha`` only where the input saturates the upper
bound (``x > alpha``), which is the PACT contribution; rounding is handled with
a straight-through estimator. Clipping the activation range is what makes low
bit-width activation quantisation viable, complementing the learned-step weight
quantiser in :mod:`sc_neurocore.qat.lsq`.

Reference: Choi et al. 2018 — "PACT: Parameterized Clipping Activation for
Quantized Neural Networks".
"""

from __future__ import annotations

from typing import Any, cast

import torch
import torch.nn as nn


class _PACTClip(torch.autograd.Function):
    """Clip to ``[0, alpha]`` with the PACT gradient to ``alpha``."""

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        """Return ``clip(x, 0, alpha)``."""
        ctx.save_for_backward(x, alpha)
        return torch.clamp(x, min=0.0).minimum(alpha)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Gradient: pass-through in ``(0, alpha)``; to ``alpha`` where ``x >= alpha``."""
        x, alpha = ctx.saved_tensors
        grad_x = grad_output * ((x >= 0) & (x <= alpha))
        grad_alpha = (grad_output * (x > alpha)).sum().reshape(alpha.shape)
        return grad_x, grad_alpha


def _round_ste(x: torch.Tensor) -> torch.Tensor:
    """Round with a straight-through (identity) gradient."""
    return (torch.round(x) - x).detach() + x


class PACTActivation(nn.Module):
    """Parameterised clipping activation with uniform quantisation.

    Parameters
    ----------
    n_bits : int
        Activation quantiser bit width (``>= 2``). The activation grid has
        ``2**n_bits - 1`` positive levels over ``[0, alpha]``.
    alpha_init : float
        Initial clipping bound.

    Attributes
    ----------
    alpha : torch.nn.Parameter
        The learned clipping bound.
    """

    def __init__(self, n_bits: int = 8, alpha_init: float = 6.0) -> None:
        super().__init__()
        if n_bits < 2:
            raise ValueError(f"n_bits must be >= 2, got {n_bits}")
        self.n_bits = n_bits
        self.n_levels = (1 << n_bits) - 1
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Clip ``x`` to ``[0, alpha]`` and quantise to ``n_bits`` levels.

        Parameters
        ----------
        x : torch.Tensor
            Pre-activation values.

        Returns
        -------
        torch.Tensor
            Clipped, quantised activations, differentiable in ``x`` and
            ``alpha``.
        """
        y = cast(torch.Tensor, _PACTClip.apply(x, self.alpha))
        scale = self.alpha.clamp(min=1e-8) / self.n_levels
        return _round_ste(y / scale).clamp(0, self.n_levels) * scale

    def quantize(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return integer activation codes and the scale for export.

        Parameters
        ----------
        x : torch.Tensor
            Pre-activation values to encode with the learned clip.

        Returns
        -------
        tuple of (torch.Tensor, torch.Tensor)
            Integer codes in ``[0, n_levels]`` and the scalar activation scale.
        """
        with torch.no_grad():
            scale = self.alpha.clamp(min=1e-8) / self.n_levels
            y = torch.clamp(x, min=0.0).minimum(self.alpha)
            codes = torch.round(y / scale).clamp(0, self.n_levels)
            return codes.to(torch.int32), scale.detach()

    def extra_repr(self) -> str:
        """Return the compact module representation."""
        return f"n_bits={self.n_bits}, alpha={self.alpha.item():.3f}"
