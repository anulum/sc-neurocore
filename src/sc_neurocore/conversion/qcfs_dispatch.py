# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Public torch-free QCFS evaluation with runtime selection

"""Quantise with QCFS and evaluate its surrogate derivatives through a chosen runtime.

NumPy is always available. Rust, Go and Mojo run from owner-built libraries
named by ``SC_NEUROCORE_QCFS_RUST_LIB``, ``SC_NEUROCORE_QCFS_GO_LIB`` and
``SC_NEUROCORE_QCFS_MOJO_LIB``; Julia runs in the configured JuliaCall runtime
when ``SC_NEUROCORE_QCFS_JULIA_ENABLED=1``. Every runtime returns the same
float64 bits as NumPy and ``QCFSActivation``.
"""

import os
from typing import Literal, cast

import numpy as np
import numpy.typing as npt

from .if_parameters import FloatArray
from .qcfs_kernel import (
    checked_qcfs_parameters,
    qcfs_values,
    reference_backward,
    reference_forward,
)
from .qcfs_native import QCFSNativeAPI, load_qcfs_julia, load_qcfs_library

QCFSBackend = Literal["auto", "numpy", "rust", "go", "mojo", "julia"]

QCFS_AUTO_ORDER: tuple[Literal["rust", "go", "mojo", "julia"], ...] = (
    "mojo",
    "rust",
    "go",
    "julia",
)
"""Configured providers ``auto`` tries in turn before NumPy: the order of their
equal-weight geometric-mean latency in the local five-runtime comparison
(``benchmarks/bench_qcfs_runtimes.py``), a preference rather than a guarantee."""


def select_qcfs_native(backend: QCFSBackend) -> QCFSNativeAPI | None:
    """Resolve the requested QCFS runtime; None selects NumPy.

    Parameters
    ----------
    backend : {'auto', 'numpy', 'rust', 'go', 'mojo', 'julia'}
        Auto takes the first configured provider of ``QCFS_AUTO_ORDER`` (Mojo,
        Rust, Go, Julia) and otherwise NumPy. An explicit native backend
        requires its configuration; explicit NumPy never reads any.

    Returns
    -------
    QCFSNativeAPI or None
        Loaded provider, or None for NumPy.

    Raises
    ------
    ValueError
        Backend name is unsupported.
    RuntimeError
        Explicit configuration is absent, or a configured provider cannot load.
    """
    if backend not in ("auto", "numpy", "rust", "go", "mojo", "julia"):
        raise ValueError("unsupported QCFS backend")
    if backend == "numpy":
        return None
    enabled = os.environ.get("SC_NEUROCORE_QCFS_JULIA_ENABLED", "0")
    if enabled not in ("0", "1"):
        raise RuntimeError("Julia QCFS opt-in must be 0 or 1")
    if backend == "julia":
        if enabled != "1":
            raise RuntimeError("Julia QCFS backend requires SC_NEUROCORE_QCFS_JULIA_ENABLED=1")
        return load_qcfs_julia()
    if backend != "auto":
        variable = f"SC_NEUROCORE_QCFS_{backend.upper()}_LIB"
        configured = os.environ.get(variable)
        if not configured:
            raise RuntimeError(f"{backend} QCFS backend requires {variable}")
        return load_qcfs_library(configured)
    for name in QCFS_AUTO_ORDER:
        if (
            (enabled == "1")
            if name == "julia"
            else bool(os.environ.get(f"SC_NEUROCORE_QCFS_{name.upper()}_LIB"))
        ):
            return select_qcfs_native(cast(QCFSBackend, name))
    return None


def qcfs_forward(
    x: npt.ArrayLike, steps: int = 8, theta: float = 1.0, *, backend: QCFSBackend = "auto"
) -> FloatArray:
    """Quantise activations onto the QCFS rate lattice without PyTorch.

    Parameters
    ----------
    x : array_like
        Real activations of any shape.
    steps : int
        Simulation steps, ``1 <= steps <= 2**32 - 1``.
    theta : float
        Finite positive firing threshold.
    backend : {'auto', 'numpy', 'rust', 'go', 'mojo', 'julia'}
        Runtime selection; see ``select_qcfs_native``.

    Returns
    -------
    numpy.ndarray
        float64 ``floor(clip(x * T / theta + 0.5, 0, T)) * theta / T`` with the
        shape of ``x``; infinities saturate and NaN stays NaN.

    Raises
    ------
    ValueError
        Step count, threshold or backend name invalid.
    TypeError
        Non-real activations.
    RuntimeError
        Selected runtime unavailable or a provider refused admitted input.
    """
    steps, theta = checked_qcfs_parameters(steps, theta)
    values = qcfs_values(x, "x")
    api = select_qcfs_native(backend)
    if api is None:
        return reference_forward(values, steps, theta)
    output = np.empty_like(values)
    status = api.forward(steps, theta, values.ctypes.data, values.size, output.ctypes.data)
    if status != 0:
        raise RuntimeError("native QCFS provider refused admitted input")
    return output


def qcfs_backward(
    x: npt.ArrayLike,
    upstream: npt.ArrayLike,
    steps: int = 8,
    theta: float = 1.0,
    *,
    backend: QCFSBackend = "auto",
) -> tuple[FloatArray, FloatArray]:
    """Evaluate QCFS straight-through derivatives without PyTorch.

    Parameters
    ----------
    x, upstream : array_like
        Real activations and upstream gradients of one shape.
    steps : int
        Simulation steps, ``1 <= steps <= 2**32 - 1``.
    theta : float
        Finite positive firing threshold.
    backend : {'auto', 'numpy', 'rust', 'go', 'mojo', 'julia'}
        Runtime selection; see ``select_qcfs_native``.

    Returns
    -------
    tuple of numpy.ndarray
        Input derivative and each element's threshold derivative, as
        ``QCFSActivation`` autograd yields for a one-element batch; summing the
        second array gives a shared threshold's gradient up to summation order.

    Raises
    ------
    ValueError
        Step count, threshold, backend name or shape mismatch.
    TypeError
        Non-real activations or gradients.
    RuntimeError
        Selected runtime unavailable or a provider refused admitted input.
    """
    steps, theta = checked_qcfs_parameters(steps, theta)
    values = qcfs_values(x, "x")
    gradients = qcfs_values(upstream, "upstream")
    if gradients.shape != values.shape:
        raise ValueError("QCFS upstream gradients must have the shape of x")
    api = select_qcfs_native(backend)
    if api is None:
        return reference_backward(values, gradients, steps, theta)
    inputs, thresholds = np.empty_like(values), np.empty_like(values)
    status = api.backward(
        steps,
        theta,
        values.ctypes.data,
        gradients.ctypes.data,
        values.size,
        inputs.ctypes.data,
        thresholds.ctypes.data,
    )
    if status != 0:
        raise RuntimeError("native QCFS provider refused admitted input")
    return inputs, thresholds


__all__ = ["QCFS_AUTO_ORDER", "QCFSBackend", "qcfs_backward", "qcfs_forward", "select_qcfs_native"]
