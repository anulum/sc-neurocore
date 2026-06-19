# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Surrogate gradient execution paths for backprop through spike

"""Surrogate gradient functions for backprop through spike discontinuities.

Forward pass: Heaviside step function (hard threshold).
Backward pass: smooth surrogate gradient that approximates the Dirac delta.

Two explicit execution paths are available:

- ``custom_op``: modern ``torch.library.custom_op`` registration
- ``legacy_autograd``: classic ``torch.autograd.Function`` path

All functions expect pre-shifted input ``x = v - threshold``.
"""

from __future__ import annotations

from typing import Any, Callable, Literal, cast

import torch
from torch.autograd import Function

SurrogatePath = Literal["custom_op", "legacy_autograd"]
SURROGATE_PATHS: tuple[SurrogatePath, ...] = ("custom_op", "legacy_autograd")


def _heaviside(x: torch.Tensor) -> torch.Tensor:
    return (x > 0).float()


def _fast_sigmoid_grad(x: torch.Tensor, slope: float) -> torch.Tensor:
    grad: torch.Tensor = slope / (1.0 + slope * x.abs()) ** 2
    return grad


def _superspike_grad(x: torch.Tensor, beta: float) -> torch.Tensor:
    grad: torch.Tensor = 1.0 / (1.0 + beta * x.abs()) ** 2
    return grad


def _atan_grad(x: torch.Tensor, alpha: float) -> torch.Tensor:
    grad: torch.Tensor = alpha / (2.0 * (1.0 + (torch.pi * alpha * x / 2.0) ** 2))
    return grad


def _sigmoid_grad(x: torch.Tensor, slope: float) -> torch.Tensor:
    sx = torch.sigmoid(slope * x)
    grad: torch.Tensor = slope * sx * (1.0 - sx)
    return grad


def _triangular_grad(x: torch.Tensor, width: float) -> torch.Tensor:
    return torch.clamp(1.0 - x.abs() / width, min=0.0) / width


class _FastSigmoidLegacy(Function):
    """Zenke & Vogels 2021 (legacy autograd path)."""

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, slope: float) -> torch.Tensor:
        ctx.save_for_backward(x)
        ctx.slope = slope
        return _heaviside(x)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        (x,) = ctx.saved_tensors
        return grad_output * _fast_sigmoid_grad(x, ctx.slope), None


class _SuperSpikeLegacy(Function):
    """Zenke & Ganguli 2018 (legacy autograd path)."""

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, beta: float) -> torch.Tensor:
        ctx.save_for_backward(x)
        ctx.beta = beta
        return _heaviside(x)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        (x,) = ctx.saved_tensors
        return grad_output * _superspike_grad(x, ctx.beta), None


class _ATanLegacy(Function):
    """Fang et al. 2021 (legacy autograd path)."""

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.save_for_backward(x)
        ctx.alpha = alpha
        return _heaviside(x)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        (x,) = ctx.saved_tensors
        return grad_output * _atan_grad(x, ctx.alpha), None


class _SigmoidLegacy(Function):
    """Standard sigmoid surrogate (legacy autograd path)."""

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, slope: float) -> torch.Tensor:
        ctx.save_for_backward(x)
        ctx.slope = slope
        return _heaviside(x)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        (x,) = ctx.saved_tensors
        return grad_output * _sigmoid_grad(x, ctx.slope), None


class _StraightThroughLegacy(Function):
    """Straight-through estimator (legacy autograd path)."""

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
        return _heaviside(x)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
        return grad_output


class _TriangularLegacy(Function):
    """Triangular window surrogate (legacy autograd path)."""

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, width: float) -> torch.Tensor:
        ctx.save_for_backward(x)
        ctx.width = width
        return _heaviside(x)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        (x,) = ctx.saved_tensors
        return grad_output * _triangular_grad(x, ctx.width), None


# Deprecated aliases kept for one release cycle.
_FastSigmoid = _FastSigmoidLegacy
_SuperSpike = _SuperSpikeLegacy
_ATan = _ATanLegacy
_Sigmoid = _SigmoidLegacy
_StraightThrough = _StraightThroughLegacy
_Triangular = _TriangularLegacy


def _fake_like_float(x: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x, dtype=torch.float32)


@torch.library.custom_op("sc_neurocore::fast_sigmoid", mutates_args=())
def _fast_sigmoid_custom_op(x: torch.Tensor, slope: float) -> torch.Tensor:
    return _heaviside(x)


@torch.library.register_fake("sc_neurocore::fast_sigmoid")
def _fast_sigmoid_fake(x: torch.Tensor, slope: float) -> torch.Tensor:
    del slope
    return _fake_like_float(x)


def _fast_sigmoid_setup_context(
    ctx: Any, inputs: tuple[torch.Tensor, float], output: torch.Tensor
) -> None:
    del output
    x, slope = inputs
    ctx.save_for_backward(x)
    ctx.slope = slope


def _fast_sigmoid_backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
    (x,) = ctx.saved_tensors
    return grad_output * _fast_sigmoid_grad(x, float(ctx.slope)), None


torch.library.register_autograd(
    "sc_neurocore::fast_sigmoid",
    _fast_sigmoid_backward,
    setup_context=_fast_sigmoid_setup_context,
)


@torch.library.custom_op("sc_neurocore::superspike", mutates_args=())
def _superspike_custom_op(x: torch.Tensor, beta: float) -> torch.Tensor:
    return _heaviside(x)


@torch.library.register_fake("sc_neurocore::superspike")
def _superspike_fake(x: torch.Tensor, beta: float) -> torch.Tensor:
    del beta
    return _fake_like_float(x)


def _superspike_setup_context(
    ctx: Any, inputs: tuple[torch.Tensor, float], output: torch.Tensor
) -> None:
    del output
    x, beta = inputs
    ctx.save_for_backward(x)
    ctx.beta = beta


def _superspike_backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
    (x,) = ctx.saved_tensors
    return grad_output * _superspike_grad(x, float(ctx.beta)), None


torch.library.register_autograd(
    "sc_neurocore::superspike",
    _superspike_backward,
    setup_context=_superspike_setup_context,
)


@torch.library.custom_op("sc_neurocore::atan_surrogate", mutates_args=())
def _atan_custom_op(x: torch.Tensor, alpha: float) -> torch.Tensor:
    return _heaviside(x)


@torch.library.register_fake("sc_neurocore::atan_surrogate")
def _atan_fake(x: torch.Tensor, alpha: float) -> torch.Tensor:
    del alpha
    return _fake_like_float(x)


def _atan_setup_context(ctx: Any, inputs: tuple[torch.Tensor, float], output: torch.Tensor) -> None:
    del output
    x, alpha = inputs
    ctx.save_for_backward(x)
    ctx.alpha = alpha


def _atan_backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
    (x,) = ctx.saved_tensors
    return grad_output * _atan_grad(x, float(ctx.alpha)), None


torch.library.register_autograd(
    "sc_neurocore::atan_surrogate",
    _atan_backward,
    setup_context=_atan_setup_context,
)


@torch.library.custom_op("sc_neurocore::sigmoid_surrogate", mutates_args=())
def _sigmoid_custom_op(x: torch.Tensor, slope: float) -> torch.Tensor:
    return _heaviside(x)


@torch.library.register_fake("sc_neurocore::sigmoid_surrogate")
def _sigmoid_fake(x: torch.Tensor, slope: float) -> torch.Tensor:
    del slope
    return _fake_like_float(x)


def _sigmoid_setup_context(
    ctx: Any, inputs: tuple[torch.Tensor, float], output: torch.Tensor
) -> None:
    del output
    x, slope = inputs
    ctx.save_for_backward(x)
    ctx.slope = slope


def _sigmoid_backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
    (x,) = ctx.saved_tensors
    return grad_output * _sigmoid_grad(x, float(ctx.slope)), None


torch.library.register_autograd(
    "sc_neurocore::sigmoid_surrogate",
    _sigmoid_backward,
    setup_context=_sigmoid_setup_context,
)


@torch.library.custom_op("sc_neurocore::straight_through", mutates_args=())
def _straight_through_custom_op(x: torch.Tensor) -> torch.Tensor:
    return _heaviside(x)


@torch.library.register_fake("sc_neurocore::straight_through")
def _straight_through_fake(x: torch.Tensor) -> torch.Tensor:
    return _fake_like_float(x)


def _straight_through_backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
    del ctx
    return grad_output


torch.library.register_autograd("sc_neurocore::straight_through", _straight_through_backward)


@torch.library.custom_op("sc_neurocore::triangular", mutates_args=())
def _triangular_custom_op(x: torch.Tensor, width: float) -> torch.Tensor:
    return _heaviside(x)


@torch.library.register_fake("sc_neurocore::triangular")
def _triangular_fake(x: torch.Tensor, width: float) -> torch.Tensor:
    del width
    return _fake_like_float(x)


def _triangular_setup_context(
    ctx: Any, inputs: tuple[torch.Tensor, float], output: torch.Tensor
) -> None:
    del output
    x, width = inputs
    ctx.save_for_backward(x)
    ctx.width = width


def _triangular_backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
    (x,) = ctx.saved_tensors
    return grad_output * _triangular_grad(x, float(ctx.width)), None


torch.library.register_autograd(
    "sc_neurocore::triangular",
    _triangular_backward,
    setup_context=_triangular_setup_context,
)


def _dispatch_path(
    path: SurrogatePath,
    custom_impl: Callable[..., torch.Tensor],
    legacy_impl: Callable[..., torch.Tensor],
    *args: Any,
) -> torch.Tensor:
    if path == "custom_op":
        return custom_impl(*args)
    if path == "legacy_autograd":
        return legacy_impl(*args)
    raise ValueError(f"Unknown surrogate path: {path!r}")


def fast_sigmoid_custom_op(x: torch.Tensor, slope: float = 25.0) -> torch.Tensor:
    """Heaviside forward, fast-sigmoid backward via ``torch.library.custom_op``."""
    return cast(torch.Tensor, _fast_sigmoid_custom_op(x, slope))


def fast_sigmoid_legacy(x: torch.Tensor, slope: float = 25.0) -> torch.Tensor:
    """Heaviside forward, fast-sigmoid backward via legacy autograd."""
    return cast(torch.Tensor, _FastSigmoidLegacy.apply(x, slope))


def fast_sigmoid(
    x: torch.Tensor,
    slope: float = 25.0,
    *,
    path: SurrogatePath = "custom_op",
) -> torch.Tensor:
    """Heaviside forward, fast-sigmoid backward."""
    return _dispatch_path(path, fast_sigmoid_custom_op, fast_sigmoid_legacy, x, slope)


def superspike_custom_op(x: torch.Tensor, beta: float = 10.0) -> torch.Tensor:
    """Heaviside forward, SuperSpike backward via ``torch.library.custom_op``."""
    return cast(torch.Tensor, _superspike_custom_op(x, beta))


def superspike_legacy(x: torch.Tensor, beta: float = 10.0) -> torch.Tensor:
    """Heaviside forward, SuperSpike backward via legacy autograd."""
    return cast(torch.Tensor, _SuperSpikeLegacy.apply(x, beta))


def superspike(
    x: torch.Tensor,
    beta: float = 10.0,
    *,
    path: SurrogatePath = "custom_op",
) -> torch.Tensor:
    """Heaviside forward, SuperSpike backward."""
    return _dispatch_path(path, superspike_custom_op, superspike_legacy, x, beta)


def atan_surrogate_custom_op(x: torch.Tensor, alpha: float = 2.0) -> torch.Tensor:
    """Heaviside forward, arctan backward via ``torch.library.custom_op``."""
    return cast(torch.Tensor, _atan_custom_op(x, alpha))


def atan_surrogate_legacy(x: torch.Tensor, alpha: float = 2.0) -> torch.Tensor:
    """Heaviside forward, arctan backward via legacy autograd."""
    return cast(torch.Tensor, _ATanLegacy.apply(x, alpha))


def atan_surrogate(
    x: torch.Tensor,
    alpha: float = 2.0,
    *,
    path: SurrogatePath = "custom_op",
) -> torch.Tensor:
    """Heaviside forward, arctan backward."""
    return _dispatch_path(path, atan_surrogate_custom_op, atan_surrogate_legacy, x, alpha)


def sigmoid_surrogate_custom_op(x: torch.Tensor, slope: float = 5.0) -> torch.Tensor:
    """Heaviside forward, sigmoid backward via ``torch.library.custom_op``."""
    return cast(torch.Tensor, _sigmoid_custom_op(x, slope))


def sigmoid_surrogate_legacy(x: torch.Tensor, slope: float = 5.0) -> torch.Tensor:
    """Heaviside forward, sigmoid backward via legacy autograd."""
    return cast(torch.Tensor, _SigmoidLegacy.apply(x, slope))


def sigmoid_surrogate(
    x: torch.Tensor,
    slope: float = 5.0,
    *,
    path: SurrogatePath = "custom_op",
) -> torch.Tensor:
    """Heaviside forward, sigmoid backward."""
    return _dispatch_path(path, sigmoid_surrogate_custom_op, sigmoid_surrogate_legacy, x, slope)


def straight_through_custom_op(x: torch.Tensor) -> torch.Tensor:
    """Heaviside forward, identity backward via ``torch.library.custom_op``."""
    return cast(torch.Tensor, _straight_through_custom_op(x))


def straight_through_legacy(x: torch.Tensor) -> torch.Tensor:
    """Heaviside forward, identity backward via legacy autograd."""
    return cast(torch.Tensor, _StraightThroughLegacy.apply(x))


def straight_through(
    x: torch.Tensor,
    *,
    path: SurrogatePath = "custom_op",
) -> torch.Tensor:
    """Heaviside forward, identity backward (STE)."""
    return _dispatch_path(path, straight_through_custom_op, straight_through_legacy, x)


def triangular_custom_op(x: torch.Tensor, width: float = 1.0) -> torch.Tensor:
    """Heaviside forward, triangular backward via ``torch.library.custom_op``."""
    return cast(torch.Tensor, _triangular_custom_op(x, width))


def triangular_legacy(x: torch.Tensor, width: float = 1.0) -> torch.Tensor:
    """Heaviside forward, triangular backward via legacy autograd."""
    return cast(torch.Tensor, _TriangularLegacy.apply(x, width))


def triangular(
    x: torch.Tensor,
    width: float = 1.0,
    *,
    path: SurrogatePath = "custom_op",
) -> torch.Tensor:
    """Heaviside forward, triangular window backward."""
    return _dispatch_path(path, triangular_custom_op, triangular_legacy, x, width)
