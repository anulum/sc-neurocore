# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Phi estimation for Integrated Information Theory

"""Approximate Phi (integrated information) for spiking networks.

Tononi 2004; Oizumi, Albantakis & Tononi 2014 (IIT 3.0+); the geometric
estimator is Barrett & Seth (2011).

Phi measures how much a system is "more than the sum of its parts" -- the
information a system generates above and beyond its parts. Full Phi is NP-hard
(it ranks every bipartition); this module provides the tractable Gaussian
geometric approximation ``phi_star``.

Under the Gaussian assumption the mutual information between two blocks is a
difference of covariance log-determinants, computed from Cholesky factors so the
estimate stays numerically stable for larger channel counts (the naive product /
ratio of raw determinants underflows). The same computation is available across
the polyglot chain (NumPy / Rust / Julia / Go / Mojo) with parity to
floating-point round-off.

    from sc_neurocore.analysis.phi_estimation import phi_star

    phi = phi_star(spike_trains, tau=1)  # spike_trains: (n_channels, n_timesteps)
"""

from __future__ import annotations

import ctypes as _ctypes
import importlib as _importlib
import importlib.util as _importlib_util
import os as _os
from typing import Any

import numpy as np


def _accel_path(*parts: str) -> str:
    """Absolute path to a backend asset under the ``accel`` tree."""
    root = _os.path.dirname(_os.path.dirname(__file__))
    return _os.path.join(root, "accel", *parts)


def _logdet_spd(a: np.ndarray[Any, Any]) -> float:
    """Natural log-determinant of an SPD matrix via Cholesky (``2 Σ log Lᵢᵢ``)."""
    chol = np.linalg.cholesky(a)
    return 2.0 * float(np.sum(np.log(np.diag(chol))))


def _gaussian_mi(x: np.ndarray[Any, Any], y: np.ndarray[Any, Any]) -> float:
    """Gaussian mutual information ``MI(X;Y)`` between two channel blocks.

    ``MI = 0.5 (log|Cov_X| + log|Cov_Y| - log|Cov_XY|)``. The covariances use the
    unbiased (``ddof=1``) estimator and a ``1e-10`` diagonal jitter, so each is SPD
    and its log-determinant comes from a Cholesky factor. Summing log-determinants
    is the numerically stable form of the determinant ratio.
    """
    eps = 1e-10
    cov_x = np.atleast_2d(np.cov(x))
    cov_y = np.atleast_2d(np.cov(y))
    cov_xy = np.atleast_2d(np.cov(np.vstack([x, y])))
    cov_x = cov_x + eps * np.eye(cov_x.shape[0])
    cov_y = cov_y + eps * np.eye(cov_y.shape[0])
    cov_xy = cov_xy + eps * np.eye(cov_xy.shape[0])

    mi = 0.5 * (_logdet_spd(cov_x) + _logdet_spd(cov_y) - _logdet_spd(cov_xy))
    return max(0.0, float(mi))


def _phi_star_python(data: np.ndarray[Any, Any], tau: int) -> float:
    """NumPy reference: geometric Phi* via Gaussian mutual information.

    ``Phi* = MI(past; future) - min_partition Σ_k MI(past_k; future_k)`` over the
    contiguous bipartitions ``{0..k} | {k..n}``.
    """
    n = data.shape[0]
    past = data[:, :-tau]
    future = data[:, tau:]

    mi_whole = _gaussian_mi(past, future)
    mi_parts_min = np.inf
    for k in range(1, n):
        mi_a = _gaussian_mi(past[:k], future[:k])
        mi_b = _gaussian_mi(past[k:], future[k:])
        mi_parts_min = min(mi_parts_min, mi_a + mi_b)

    return max(0.0, float(mi_whole - mi_parts_min))


def _load_rust_phi() -> Any | None:
    """Return the Rust ``phi_star`` entry point, or ``None`` when absent."""
    try:
        return _importlib.import_module("sc_neurocore_engine").py_phi_star
    except (ImportError, AttributeError):
        return None


# Rust acceleration backend — probed at import (the wheel import is cheap).
_rust_phi: Any | None = _load_rust_phi()

# Julia / Go / Mojo backends — loaded lazily on first explicit request.
_julia_phi: Any | None = None
_go_phi_lib: Any | None = None
_mojo_phi_lib: Any | None = None


def _ensure_julia_phi() -> bool:
    """Lazy-load the Julia Phi module, returning ``True`` when available."""
    global _julia_phi
    if _julia_phi is not None:
        return True
    if _importlib_util.find_spec("juliacall") is None:
        return False
    jl_path = _accel_path("julia", "analysis", "phi_estimation.jl")
    if not _os.path.isfile(jl_path):
        return False
    jl = _importlib.import_module("juliacall").Main
    jl.include(jl_path)
    _julia_phi = jl.PhiAccel
    return True


def _ensure_go_phi() -> bool:
    """Lazy-load the Go Phi c-shared library, returning ``True`` when available."""
    global _go_phi_lib
    if _go_phi_lib is not None:
        return True
    so_path = _accel_path("go", "phi", "libphi.so")
    if not _os.path.isfile(so_path):
        return False
    try:
        lib = _ctypes.CDLL(so_path)
    except OSError:
        return False
    fn = getattr(lib, "phi_star_c", None)
    if fn is None:
        return False
    fn.argtypes = [_ctypes.POINTER(_ctypes.c_double)] + [_ctypes.c_int] * 3
    fn.restype = _ctypes.c_double
    _go_phi_lib = lib
    return True


def _ensure_mojo_phi() -> bool:
    """Lazy-load the Mojo Phi shared library, returning ``True`` when available."""
    global _mojo_phi_lib
    if _mojo_phi_lib is not None:
        return True
    so_path = _accel_path("mojo", "kernels", "libphi.so")
    if not _os.path.isfile(so_path):
        return False
    try:
        lib = _ctypes.CDLL(so_path)
    except OSError:
        return False
    fn = getattr(lib, "phi_star_c", None)
    if fn is None:
        return False
    fn.argtypes = [_ctypes.c_int64] * 5
    fn.restype = None
    _mojo_phi_lib = lib
    return True


def _run_rust_phi(data: np.ndarray[Any, Any], tau: int) -> float:
    """Dispatch Phi* to the Rust engine."""
    assert _rust_phi is not None
    return float(_rust_phi(np.ascontiguousarray(data, dtype=np.float64), int(tau)))


def _run_julia_phi(data: np.ndarray[Any, Any], tau: int) -> float:
    """Dispatch Phi* to the Julia backend."""
    assert _julia_phi is not None
    return float(_julia_phi.phi_star(np.ascontiguousarray(data, dtype=np.float64), int(tau)))


def _run_go_phi(data: np.ndarray[Any, Any], tau: int) -> float:
    """Dispatch Phi* to the Go c-shared backend."""
    assert _go_phi_lib is not None
    n_channels, n_timesteps = data.shape
    buf = np.ascontiguousarray(data, dtype=np.float64).reshape(-1)
    ptr = buf.ctypes.data_as(_ctypes.POINTER(_ctypes.c_double))
    return float(_go_phi_lib.phi_star_c(ptr, n_channels, n_timesteps, int(tau)))


def _run_mojo_phi(data: np.ndarray[Any, Any], tau: int) -> float:
    """Dispatch Phi* to the Mojo c-shared backend (raw ``int64`` addresses)."""
    assert _mojo_phi_lib is not None
    n_channels, n_timesteps = data.shape
    buf = np.ascontiguousarray(data, dtype=np.float64).reshape(-1)
    out = np.zeros(1, dtype=np.float64)
    _mojo_phi_lib.phi_star_c(buf.ctypes.data, n_channels, n_timesteps, int(tau), out.ctypes.data)
    return float(out[0])


def _phi_star_dispatch(data: np.ndarray[Any, Any], tau: int, backend: str) -> float:
    """Run Phi* on the requested backend; ``auto`` selects the fastest available.

    Every backend computes the same Gaussian geometric estimator with the
    Cholesky log-determinant form and agrees with the NumPy reference up to
    floating-point round-off. ``auto`` prefers the Rust engine when present and
    falls back to the NumPy reference otherwise; Julia, Go and Mojo run on request.
    """
    if backend not in ("auto", "python", "rust", "julia", "go", "mojo"):
        raise ValueError(f"Phi backend {backend!r} is not available")
    if backend in ("auto", "rust") and _rust_phi is not None:
        return _run_rust_phi(data, tau)
    if backend == "rust":
        raise RuntimeError("Rust Phi backend is not available in this environment")
    if backend == "julia":
        if not _ensure_julia_phi():
            raise RuntimeError("Julia Phi backend is not available")
        return _run_julia_phi(data, tau)
    if backend == "go":
        if not _ensure_go_phi():
            raise RuntimeError("Go Phi backend is not available")
        return _run_go_phi(data, tau)
    if backend == "mojo":
        if not _ensure_mojo_phi():
            raise RuntimeError("Mojo Phi backend is not available")
        return _run_mojo_phi(data, tau)
    return _phi_star_python(data, tau)


def phi_star(data: np.ndarray[Any, Any], tau: int = 1, backend: str = "auto") -> float:
    """Geometric integrated information Phi* (Barrett & Seth 2011).

    ``Phi* = MI(past; future) - min_partition Σ_k MI(past_k; future_k)`` under the
    Gaussian assumption, with each mutual information a difference of covariance
    log-determinants (Cholesky form).

    Parameters
    ----------
    data : numpy.ndarray
        Shape ``(n_channels, n_timesteps)`` — spike counts or continuous signals.
    tau : int, optional
        Time lag for the past→future mapping.
    backend : str, optional
        ``"auto"`` selects the fastest available backend (Rust engine when present,
        otherwise the NumPy reference); ``"python"``, ``"rust"``, ``"julia"``,
        ``"go"`` and ``"mojo"`` force a specific path.

    Returns
    -------
    float
        Phi* estimate in nats. Non-negative; ``0`` means fully reducible.
    """
    arr = np.ascontiguousarray(data, dtype=np.float64)
    n, n_timesteps = arr.shape
    if n < 2 or 2 * tau >= n_timesteps:
        return 0.0
    return _phi_star_dispatch(arr, int(tau), backend)


def phi_from_spike_trains(
    spikes: np.ndarray[Any, Any], bin_size: int = 10, tau: int = 1, backend: str = "auto"
) -> float:
    """Compute Phi* from binary spike trains by binning into spike counts.

    Parameters
    ----------
    spikes : numpy.ndarray
        Shape ``(n_neurons, n_timesteps)``, binary ``{0, 1}``.
    bin_size : int, optional
        Number of timesteps per bin.
    tau : int, optional
        Time lag in bins.
    backend : str, optional
        Forwarded to :func:`phi_star`.

    Returns
    -------
    float
        Phi* in nats.
    """
    n_neurons, n_steps = spikes.shape
    n_bins = n_steps // bin_size
    if n_bins < 2 * tau + 2:
        return 0.0

    binned = np.zeros((n_neurons, n_bins), dtype=np.float64)
    for b in range(n_bins):
        binned[:, b] = spikes[:, b * bin_size : (b + 1) * bin_size].sum(axis=1)

    return phi_star(binned, tau=tau, backend=backend)
