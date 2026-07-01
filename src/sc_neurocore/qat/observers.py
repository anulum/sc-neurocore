# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Quantisation observers (per-tensor and per-channel)

"""Range observers that turn weight/activation statistics into quantiser scales.

An observer watches tensors during calibration, tracks their value range, and
converts that range into the ``(scale, zero_point)`` a uniform affine quantiser
needs. Two granularities are provided:

Per-tensor
    One scale for the whole tensor (:class:`MinMaxObserver`).

Per-channel
    One scale per output channel along a chosen axis
    (:class:`PerChannelMinMaxObserver`). Per-channel weight quantisation
    absorbs the wide dynamic-range differences between filters that a single
    per-tensor scale would otherwise clip or under-resolve, which is the
    standard remedy for the accuracy loss of low-bit weight quantisation.

Both support a symmetric scheme (signed weights, ``zero_point == 0``) and an
affine scheme (arbitrary ``[min, max]`` mapped onto the integer grid). The
observed range is a running min/max across every :meth:`observe` call, so a
calibration loop can stream batches through the observer before the scales are
read once via :meth:`calculate_qparams`.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _quant_bounds(n_bits: int, *, unsigned: bool) -> tuple[int, int]:
    """Return the ``(qmin, qmax)`` integer grid bounds for a bit width.

    Parameters
    ----------
    n_bits : int
        Quantiser bit width (``>= 2``).
    unsigned : bool
        When ``True`` the grid is ``[0, 2**n_bits - 1]`` (e.g. post-ReLU
        activations); when ``False`` it is the signed range
        ``[-2**(n_bits-1), 2**(n_bits-1) - 1]`` (e.g. weights).

    Returns
    -------
    tuple of (int, int)
        The inclusive lower and upper integer codes.
    """
    if n_bits < 2:
        raise ValueError(f"n_bits must be >= 2, got {n_bits}")
    if unsigned:
        return 0, (1 << n_bits) - 1
    return -(1 << (n_bits - 1)), (1 << (n_bits - 1)) - 1


def _qparams_from_range(
    min_val: torch.Tensor,
    max_val: torch.Tensor,
    *,
    n_bits: int,
    symmetric: bool,
    unsigned: bool,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Derive ``(scale, zero_point)`` from an observed value range.

    Parameters
    ----------
    min_val, max_val : torch.Tensor
        Observed minimum and maximum. Scalars for the per-tensor case or
        1-D per-channel vectors; the returned tensors match their shape.
    n_bits : int
        Quantiser bit width.
    symmetric : bool
        When ``True`` the range is symmetrised about zero and the zero point
        is pinned to the grid centre (``0`` for signed, the mid-code for
        unsigned); when ``False`` an affine mapping of ``[min, max]`` is used.
    unsigned : bool
        Whether the integer grid is unsigned (see :func:`_quant_bounds`).
    eps : float
        Floor applied to the scale so a degenerate (zero-width) range cannot
        produce a zero or non-finite scale.

    Returns
    -------
    tuple of (torch.Tensor, torch.Tensor)
        The per-element ``scale`` (float) and ``zero_point`` (float-valued but
        integral) broadcastable against the quantised tensor.
    """
    qmin, qmax = _quant_bounds(n_bits, unsigned=unsigned)
    # Always include zero in the observed range so it is exactly representable.
    min_r = torch.minimum(min_val, torch.zeros_like(min_val))
    max_r = torch.maximum(max_val, torch.zeros_like(max_val))

    if symmetric:
        abs_max = torch.maximum(min_r.abs(), max_r.abs()).clamp(min=eps)
        if unsigned:
            scale = abs_max / float(qmax)
            zero_point = torch.zeros_like(scale)
        else:
            scale = abs_max / float(qmax)
            zero_point = torch.zeros_like(scale)
    else:
        scale = ((max_r - min_r) / float(qmax - qmin)).clamp(min=eps)
        zero_point = torch.round(float(qmin) - min_r / scale).clamp(qmin, qmax)
    return scale, zero_point


def fake_quantize(
    x: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    *,
    n_bits: int,
    unsigned: bool,
) -> torch.Tensor:
    """Quantise then de-quantise ``x`` (simulated quantisation, no STE).

    This is the inference-time / calibration-time fake-quant used to evaluate
    an observer's scales; for training use the learned-step quantisers in
    :mod:`sc_neurocore.qat.lsq`. ``scale`` and ``zero_point`` broadcast against
    ``x``, so per-channel parameters must already be reshaped onto the channel
    axis by the caller.

    Parameters
    ----------
    x : torch.Tensor
        Tensor to fake-quantise.
    scale, zero_point : torch.Tensor
        Quantiser parameters, broadcastable to ``x``.
    n_bits : int
        Quantiser bit width.
    unsigned : bool
        Whether the integer grid is unsigned.

    Returns
    -------
    torch.Tensor
        The de-quantised approximation of ``x`` on the integer grid.
    """
    qmin, qmax = _quant_bounds(n_bits, unsigned=unsigned)
    codes = torch.round(x / scale + zero_point).clamp(qmin, qmax)
    return (codes - zero_point) * scale


class MinMaxObserver(nn.Module):
    """Per-tensor running min/max range observer.

    Parameters
    ----------
    n_bits : int
        Quantiser bit width the derived scale targets.
    symmetric : bool
        Use a symmetric (zero-centred) mapping — the default for weights.
    unsigned : bool
        Target an unsigned integer grid (e.g. non-negative activations).
    eps : float
        Scale floor guarding against a zero-width observed range.
    """

    min_val: torch.Tensor
    max_val: torch.Tensor

    def __init__(
        self,
        n_bits: int = 8,
        *,
        symmetric: bool = True,
        unsigned: bool = False,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.n_bits = n_bits
        self.symmetric = symmetric
        self.unsigned = unsigned
        self.eps = eps
        self.register_buffer("min_val", torch.tensor(float("inf")))
        self.register_buffer("max_val", torch.tensor(float("-inf")))

    def observe(self, x: torch.Tensor) -> torch.Tensor:
        """Fold ``x`` into the running range and return it unchanged.

        Parameters
        ----------
        x : torch.Tensor
            Calibration tensor.

        Returns
        -------
        torch.Tensor
            ``x`` unchanged, so the observer can be dropped into a forward pass.
        """
        x = x.detach()
        self.min_val = torch.minimum(self.min_val, x.min())
        self.max_val = torch.maximum(self.max_val, x.max())
        return x

    def calculate_qparams(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the ``(scale, zero_point)`` for the observed range.

        Returns
        -------
        tuple of (torch.Tensor, torch.Tensor)
            Scalar scale and zero point.

        Raises
        ------
        RuntimeError
            If no tensor has been observed yet.
        """
        if not torch.isfinite(self.min_val) or not torch.isfinite(self.max_val):
            raise RuntimeError("MinMaxObserver.calculate_qparams called before any observation")
        return _qparams_from_range(
            self.min_val,
            self.max_val,
            n_bits=self.n_bits,
            symmetric=self.symmetric,
            unsigned=self.unsigned,
            eps=self.eps,
        )

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """Fake-quantise ``x`` with the currently observed scale."""
        scale, zero_point = self.calculate_qparams()
        return fake_quantize(x, scale, zero_point, n_bits=self.n_bits, unsigned=self.unsigned)


class PerChannelMinMaxObserver(nn.Module):
    """Per-channel running min/max range observer.

    Tracks an independent min/max — and therefore an independent scale — for
    every slice along ``ch_axis``. For a weight tensor shaped
    ``(out_features, in_features)`` the default ``ch_axis=0`` yields one scale
    per output neuron.

    Parameters
    ----------
    n_bits : int
        Quantiser bit width the derived scales target.
    ch_axis : int
        Axis whose length is the channel count.
    symmetric : bool
        Use a symmetric (zero-centred) mapping — the default for weights.
    unsigned : bool
        Target an unsigned integer grid.
    eps : float
        Scale floor guarding against a zero-width observed range.
    """

    min_vals: torch.Tensor
    max_vals: torch.Tensor

    def __init__(
        self,
        n_bits: int = 8,
        *,
        ch_axis: int = 0,
        symmetric: bool = True,
        unsigned: bool = False,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.n_bits = n_bits
        self.ch_axis = ch_axis
        self.symmetric = symmetric
        self.unsigned = unsigned
        self.eps = eps
        self.register_buffer("min_vals", torch.empty(0))
        self.register_buffer("max_vals", torch.empty(0))

    def _per_channel_min_max(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Collapse every axis except ``ch_axis`` to per-channel min and max."""
        axis = self.ch_axis % x.dim()
        moved = x.movedim(axis, 0).reshape(x.shape[axis], -1)
        return moved.min(dim=1).values, moved.max(dim=1).values

    def observe(self, x: torch.Tensor) -> torch.Tensor:
        """Fold ``x`` into the running per-channel range and return it unchanged.

        Parameters
        ----------
        x : torch.Tensor
            Calibration tensor whose ``ch_axis`` length is the channel count.

        Returns
        -------
        torch.Tensor
            ``x`` unchanged.
        """
        x = x.detach()
        cur_min, cur_max = self._per_channel_min_max(x)
        if self.min_vals.numel() == 0:
            self.min_vals = cur_min
            self.max_vals = cur_max
        else:
            self.min_vals = torch.minimum(self.min_vals, cur_min)
            self.max_vals = torch.maximum(self.max_vals, cur_max)
        return x

    def calculate_qparams(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the per-channel ``(scale, zero_point)`` vectors.

        Returns
        -------
        tuple of (torch.Tensor, torch.Tensor)
            1-D scale and zero-point tensors, one entry per channel.

        Raises
        ------
        RuntimeError
            If no tensor has been observed yet.
        """
        if self.min_vals.numel() == 0:
            raise RuntimeError(
                "PerChannelMinMaxObserver.calculate_qparams called before any observation"
            )
        return _qparams_from_range(
            self.min_vals,
            self.max_vals,
            n_bits=self.n_bits,
            symmetric=self.symmetric,
            unsigned=self.unsigned,
            eps=self.eps,
        )

    def _broadcast_shape(self, ndim: int) -> list[int]:
        """Shape that reshapes a per-channel vector for broadcasting over ``ndim`` dims."""
        axis = self.ch_axis % ndim
        shape = [1] * ndim
        shape[axis] = -1
        return shape

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """Fake-quantise ``x`` with the observed per-channel scales."""
        scale, zero_point = self.calculate_qparams()
        shape = self._broadcast_shape(x.dim())
        return fake_quantize(
            x,
            scale.reshape(shape),
            zero_point.reshape(shape),
            n_bits=self.n_bits,
            unsigned=self.unsigned,
        )
