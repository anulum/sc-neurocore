# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — backend dispatch for contrastive self-supervised learning

"""Run the contrastive SSL kernels on a selected execution lane.

The kernels are those of :mod:`sc_neurocore.contrastive.ssl`: the InfoNCE loss
of :class:`~sc_neurocore.contrastive.ssl.SpikeContrastiveLoss`, one phase of
:class:`~sc_neurocore.contrastive.ssl.CSDPRule` and its goodness score. The
Python original is the reference and the floor. Julia (``accel/julia/
contrastive/ssl.jl``) and Mojo (``accel/mojo/contrastive_ssl``) are parity
lanes. Every input is validated here, with the original's own validators,
before any lane runs, so all lanes refuse the same inputs with the same errors.

``backend="auto"`` follows the measured order for kernel ``contrastive_ssl``
on this host. Without a committed measurement that order is Python alone, so
``auto`` behaves as the original does.

Parity bounds against Python, stated per lane:

- InfoNCE loss: relative ``LOSS_RTOL``. The original's BLAS matrix product and
  the lanes' sequential sums round differently.
- CSDP phases: ``|lane - python| <= CSDP_TERM_EPS * eps * (|W| + lr*|post (x) pre|
  + decay*|W|)`` element by element. Julia is exact (factor 0). The Mojo
  compiler contracts multiply and add into fused multiply-adds, which round once
  where the original rounds twice, so its factor is 4.
- Goodness: relative ``GOODNESS_RTOL``. NumPy and Julia both sum pairwise but in
  different blocks, and the Mojo lane sums sequentially.
"""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Literal

import numpy as np
import numpy.typing as npt

from sc_neurocore.accel.backend_selection import select_backend_order
from sc_neurocore.contrastive.ssl import (
    CSDPRule,
    SpikeContrastiveLoss,
    _as_float_matrix,
    _as_float_vector,
    _as_weight_matrix,
)

Backend = Literal["auto", "python", "julia", "mojo"]

KERNEL = "contrastive_ssl"
BACKENDS: tuple[str, ...] = ("python", "julia", "mojo")
_STATIC_ORDER: tuple[str, ...] = ("python",)
LOSS_RTOL: dict[str, float] = {"julia": 1e-9, "mojo": 1e-9}
CSDP_TERM_EPS: dict[str, float] = {"julia": 0.0, "mojo": 4.0}
GOODNESS_RTOL: dict[str, float] = {"julia": 1e-12, "mojo": 1e-12}

_LANE_MODULES = {
    "julia": "sc_neurocore.accel.julia.contrastive.ssl",
    "mojo": "sc_neurocore.accel.mojo.contrastive_ssl",
}


def _lane(backend: str) -> ModuleType:
    return importlib.import_module(_LANE_MODULES[backend])


def backend_available(backend: str) -> bool:
    """Return whether one execution lane is ready on this host.

    Parameters
    ----------
    backend : str
        One of ``python``, ``julia`` or ``mojo``.

    Returns
    -------
    bool
        ``True`` when the lane's runtime and its source or library load.
    """
    if backend == "python":
        return True
    if backend == "julia":
        try:
            _lane("julia").ensure_loaded()
        except (ImportError, FileNotFoundError):
            return False
        except Exception as exc:
            if exc.__class__.__name__ == "JuliaError":
                return False
            raise
        return True
    if backend == "mojo":
        try:
            module = _lane("mojo")
        except ImportError:
            return False
        return bool(module._HAS_MOJO_CONTRASTIVE_SSL)
    return False


def auto_backend() -> str:
    """Return the first available lane in this host's measured order.

    Returns
    -------
    str
        A lane name; ``python`` without a committed measurement for this host.
    """
    order = select_backend_order(KERNEL, static=_STATIC_ORDER)
    return next((backend for backend in order if backend_available(backend)), "python")


def _resolve(backend: Backend) -> str:
    if backend == "auto":
        return auto_backend()
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {('auto', *BACKENDS)}, got {backend!r}")
    if not backend_available(backend):
        raise RuntimeError(f"the {backend} lane of {KERNEL} is not available on this host")
    return backend


def infonce_loss(
    view_a: npt.ArrayLike,
    view_b: npt.ArrayLike,
    *,
    temperature: float = 0.5,
    backend: Backend = "auto",
) -> float:
    """Return the mean InfoNCE loss of two spike-rate views.

    Parameters
    ----------
    view_a, view_b : array_like of shape (batch, n_features)
        Spike rates of two augmentations of the same inputs.
    temperature : float, default 0.5
        Contrastive temperature; finite and positive.
    backend : {"auto", "python", "julia", "mojo"}, default "auto"
        The execution lane.

    Returns
    -------
    float
        The loss; ``0.0`` for a batch of fewer than two rows.

    Raises
    ------
    ValueError
        If an input is invalid, as :meth:`SpikeContrastiveLoss.compute` refuses
        it, or ``backend`` is unknown.
    RuntimeError
        If an explicitly named lane is not available.
    """
    loss = SpikeContrastiveLoss(temperature)
    a = _as_float_matrix(np.asarray(view_a), "view_a")
    b = _as_float_matrix(np.asarray(view_b), "view_b")
    if a.shape != b.shape:
        raise ValueError("view_a and view_b must have the same shape")
    lane = _resolve(backend)
    if lane == "python":
        return loss.compute(a, b)
    return float(
        _lane(lane).infonce_loss(np.ascontiguousarray(a), np.ascontiguousarray(b), loss.temperature)
    )


def csdp_update(
    weights: npt.ArrayLike,
    pre_spikes: npt.ArrayLike,
    post_spikes: npt.ArrayLike,
    *,
    positive: bool,
    lr: float = 0.01,
    decay: float = 0.001,
    backend: Backend = "auto",
) -> npt.NDArray[np.float64]:
    """Return one CSDP phase applied to ``weights``.

    Parameters
    ----------
    weights : array_like of shape (n_post, n_pre)
        Current synaptic weights; ``n_pre`` may be zero.
    pre_spikes, post_spikes : array_like of shape (n_pre,) and (n_post,)
        Presynaptic and postsynaptic spike-rate vectors.
    positive : bool
        ``True`` for the Hebbian phase (:meth:`CSDPRule.positive_update`),
        ``False`` for the anti-Hebbian phase (:meth:`CSDPRule.negative_update`).
    lr, decay : float
        Finite, non-negative rule parameters.
    backend : {"auto", "python", "julia", "mojo"}, default "auto"
        The execution lane.

    Returns
    -------
    ndarray of shape (n_post, n_pre)
        The updated weights, in a new array.

    Raises
    ------
    ValueError
        If an input is invalid, as :class:`CSDPRule` refuses it, or ``backend``
        is unknown.
    RuntimeError
        If an explicitly named lane is not available.
    """
    rule = CSDPRule(lr=lr, decay=decay)
    w = _as_weight_matrix(np.asarray(weights))
    pre = _as_float_vector(np.asarray(pre_spikes), "pre_spikes")
    post = _as_float_vector(np.asarray(post_spikes), "post_spikes")
    if w.shape != (post.shape[0], pre.shape[0]):
        raise ValueError("weights must have shape (len(post_spikes), len(pre_spikes))")
    lane = _resolve(backend)
    if lane == "python":
        phase = rule.positive_update if positive else rule.negative_update
        return np.asarray(phase(w, pre, post), dtype=np.float64)
    result: npt.NDArray[np.float64] = _lane(lane).csdp_update(
        np.ascontiguousarray(w),
        np.ascontiguousarray(pre),
        np.ascontiguousarray(post),
        rule.lr,
        rule.decay,
        positive=positive,
    )
    return result


def goodness(activations: npt.ArrayLike, *, backend: Backend = "auto") -> float:
    """Return the goodness score (sum of squared activations).

    Parameters
    ----------
    activations : array_like
        Activations of any shape.
    backend : {"auto", "python", "julia", "mojo"}, default "auto"
        The execution lane.

    Returns
    -------
    float
        ``sum(activations**2)``.

    Raises
    ------
    ValueError
        If an activation is not finite, or ``backend`` is unknown.
    RuntimeError
        If an explicitly named lane is not available.
    """
    values = np.asarray(activations, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("activations must contain only finite values")
    lane = _resolve(backend)
    if lane == "python":
        return CSDPRule().goodness(values)
    return float(_lane(lane).goodness(np.ascontiguousarray(values)))
