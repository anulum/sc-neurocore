# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Spike sorting quality metrics

"""Spike sorting quality metrics.

The Mahalanobis cluster-quality metrics ``isolation_distance`` (Harris et al.
2001) and ``l_ratio`` (Schmitzer-Torbert et al. 2005) share one numerically
optimal kernel: the squared Mahalanobis distance ``(x-μ)ᵀ Σ⁻¹ (x-μ)`` is
evaluated through the Cholesky factor of the regularised cluster covariance
``Σ = L Lᵀ`` (solving ``L z = x-μ`` and summing ``z²``) rather than forming and
multiplying by ``Σ⁻¹``. This is the LAPACK-grade route — more accurate for
ill-conditioned cluster covariances and cheaper than an explicit inverse. The
same kernel is available across the polyglot chain (NumPy / Rust / Julia / Go /
Mojo) with parity to floating-point round-off, selected via ``backend=``.
"""

from __future__ import annotations

import ctypes as _ctypes
import importlib as _importlib
import importlib.util as _importlib_util
import os as _os
from typing import Any

import numpy as np
from scipy.linalg import solve_triangular

from .basic import isi, bin_spike_train


def _accel_path(*parts: str) -> str:
    """Absolute path to a backend asset under the ``accel`` tree."""
    root = _os.path.dirname(_os.path.dirname(_os.path.dirname(__file__)))
    return _os.path.join(root, "accel", *parts)


def _cluster_mahalanobis_sq(
    cluster: np.ndarray[Any, Any], noise: np.ndarray[Any, Any]
) -> np.ndarray[Any, Any]:
    """Squared Mahalanobis distances of each ``noise`` row from the cluster mean.

    The cluster covariance ``Σ`` is the unbiased (``ddof=1``) feature covariance
    with a ``1e-8`` diagonal jitter, so it is SPD. The quadratic form
    ``(x-μ)ᵀ Σ⁻¹ (x-μ)`` is computed from the Cholesky factor ``Σ = L Lᵀ`` via a
    triangular solve ``L z = (x-μ)`` followed by ``Σ z²``; ``Σ`` is never
    inverted explicitly.
    """
    mu = cluster.mean(axis=0)
    cov = np.cov(cluster.T)
    if cov.ndim < 2:
        cov = np.atleast_2d(cov)
    cov = cov + 1e-8 * np.eye(cov.shape[0])
    chol = np.linalg.cholesky(cov)  # lower L
    diff = (noise - mu).T  # (d, n_noise)
    z = solve_triangular(chol, diff, lower=True)
    mah_sq: np.ndarray[Any, Any] = np.einsum("ij,ij->j", z, z)
    return mah_sq


def _isolation_distance_python(cluster: np.ndarray[Any, Any], noise: np.ndarray[Any, Any]) -> float:
    """NumPy reference for :func:`isolation_distance` (inputs pre-validated)."""
    n_c = cluster.shape[0]
    mah = np.sort(_cluster_mahalanobis_sq(cluster, noise))
    # Inputs are pre-validated (n_noise >= n_cluster), so the n_cluster-th
    # smallest squared Mahalanobis distance always exists.
    return float(mah[n_c - 1])


def _l_ratio_python(cluster: np.ndarray[Any, Any], noise: np.ndarray[Any, Any]) -> float:
    """NumPy reference for :func:`l_ratio` (inputs pre-validated)."""
    n_c = cluster.shape[0]
    mah = np.clip(_cluster_mahalanobis_sq(cluster, noise), 1e-10, None)
    d = cluster.shape[1]
    l_vals = np.clip(np.exp(-0.5 * (mah - d)), 0, 1)
    return float(l_vals.sum() / n_c)


def _load_rust_metric(name: str) -> Any | None:
    """Return a Rust sorting-quality entry point, or ``None`` when absent."""
    try:
        return getattr(_importlib.import_module("sc_neurocore_engine"), name)
    except (ImportError, AttributeError):
        return None


# Rust acceleration backends — probed at import (the wheel import is cheap).
_rust_isolation: Any | None = _load_rust_metric("py_isolation_distance")
_rust_l_ratio: Any | None = _load_rust_metric("py_l_ratio")

# Julia / Go / Mojo backends — loaded lazily on first explicit request.
_julia_sq: Any | None = None
_go_sq_lib: Any | None = None
_mojo_sq_lib: Any | None = None


def _ensure_julia_sq() -> bool:
    """Lazy-load the Julia sorting-quality module, ``True`` when available."""
    global _julia_sq
    if _julia_sq is not None:
        return True
    if _importlib_util.find_spec("juliacall") is None:
        return False
    jl_path = _accel_path("julia", "analysis", "sorting_quality.jl")
    if not _os.path.isfile(jl_path):
        return False
    jl = _importlib.import_module("juliacall").Main
    jl.include(jl_path)
    _julia_sq = jl.SortingQualityAccel
    return True


def _ensure_go_sq() -> bool:
    """Lazy-load the Go sorting-quality c-shared library, ``True`` when available."""
    global _go_sq_lib
    if _go_sq_lib is not None:
        return True
    so_path = _accel_path("go", "sorting_quality", "libsorting_quality.so")
    if not _os.path.isfile(so_path):
        return False
    try:
        lib = _ctypes.CDLL(so_path)
    except OSError:
        return False
    iso = getattr(lib, "isolation_distance_c", None)
    lr = getattr(lib, "l_ratio_c", None)
    if iso is None or lr is None:
        return False
    sig = [
        _ctypes.POINTER(_ctypes.c_double),
        _ctypes.c_int,
        _ctypes.POINTER(_ctypes.c_double),
        _ctypes.c_int,
        _ctypes.c_int,
    ]
    for fn in (iso, lr):
        fn.argtypes = sig
        fn.restype = _ctypes.c_double
    _go_sq_lib = lib
    return True


def _ensure_mojo_sq() -> bool:
    """Lazy-load the Mojo sorting-quality shared library, ``True`` when available."""
    global _mojo_sq_lib
    if _mojo_sq_lib is not None:
        return True
    so_path = _accel_path("mojo", "kernels", "libsorting_quality.so")
    if not _os.path.isfile(so_path):
        return False
    try:
        lib = _ctypes.CDLL(so_path)
    except OSError:
        return False
    iso = getattr(lib, "isolation_distance_c", None)
    lr = getattr(lib, "l_ratio_c", None)
    if iso is None or lr is None:
        return False
    for fn in (iso, lr):
        fn.argtypes = [_ctypes.c_int64] * 6
        fn.restype = None
    _mojo_sq_lib = lib
    return True


def _run_go_metric(fn: Any, cluster: np.ndarray[Any, Any], noise: np.ndarray[Any, Any]) -> float:
    """Dispatch a sorting-quality metric to the Go c-shared backend."""
    n_c, d = cluster.shape
    n_noise = noise.shape[0]
    cbuf = np.ascontiguousarray(cluster, dtype=np.float64).reshape(-1)
    nbuf = np.ascontiguousarray(noise, dtype=np.float64).reshape(-1)
    cptr = cbuf.ctypes.data_as(_ctypes.POINTER(_ctypes.c_double))
    nptr = nbuf.ctypes.data_as(_ctypes.POINTER(_ctypes.c_double))
    return float(fn(cptr, n_c, nptr, n_noise, d))


def _run_mojo_metric(fn: Any, cluster: np.ndarray[Any, Any], noise: np.ndarray[Any, Any]) -> float:
    """Dispatch a sorting-quality metric to the Mojo backend (raw ``int64`` addresses)."""
    n_c, d = cluster.shape
    n_noise = noise.shape[0]
    cbuf = np.ascontiguousarray(cluster, dtype=np.float64).reshape(-1)
    nbuf = np.ascontiguousarray(noise, dtype=np.float64).reshape(-1)
    out = np.zeros(1, dtype=np.float64)
    fn(cbuf.ctypes.data, n_c, nbuf.ctypes.data, n_noise, d, out.ctypes.data)
    return float(out[0])


_SQ_BACKENDS = ("auto", "python", "rust", "julia", "go", "mojo")


def _sq_dispatch(
    metric: str,
    cluster: np.ndarray[Any, Any],
    noise: np.ndarray[Any, Any],
    backend: str,
) -> float:
    """Run a Mahalanobis sorting-quality metric on the requested backend.

    ``metric`` is ``"isolation_distance"`` or ``"l_ratio"``. Every backend shares
    the Cholesky-solve kernel and agrees with the NumPy reference up to
    floating-point round-off. ``auto`` prefers the Rust engine when present and
    falls back to the NumPy reference; Julia, Go and Mojo run on request.
    """
    if backend not in _SQ_BACKENDS:
        raise ValueError(f"sorting-quality backend {backend!r} is not available")

    rust = _rust_isolation if metric == "isolation_distance" else _rust_l_ratio
    if backend in ("auto", "rust") and rust is not None:
        return float(rust(cluster, noise))
    if backend == "rust":
        raise RuntimeError("Rust sorting-quality backend is not available in this environment")
    if backend == "julia":
        if not _ensure_julia_sq():
            raise RuntimeError("Julia sorting-quality backend is not available")
        return float(getattr(_julia_sq, metric)(cluster, noise))
    if backend == "go":
        if not _ensure_go_sq():
            raise RuntimeError("Go sorting-quality backend is not available")
        return _run_go_metric(getattr(_go_sq_lib, f"{metric}_c"), cluster, noise)
    if backend == "mojo":
        if not _ensure_mojo_sq():
            raise RuntimeError("Mojo sorting-quality backend is not available")
        return _run_mojo_metric(getattr(_mojo_sq_lib, f"{metric}_c"), cluster, noise)
    if metric == "isolation_distance":
        return _isolation_distance_python(cluster, noise)
    return _l_ratio_python(cluster, noise)


def isolation_distance(
    cluster: np.ndarray[Any, Any], noise: np.ndarray[Any, Any], backend: str = "auto"
) -> float:
    """Isolation distance (Harris et al. 2001).

    The Mahalanobis distance at which the number of noise points reaching the
    cluster equals the cluster size — the squared Mahalanobis radius of the
    ``n_cluster``-th nearest noise point.

    Parameters
    ----------
    cluster : numpy.ndarray
        Shape ``(n_cluster, n_features)`` — the cluster's feature vectors.
    noise : numpy.ndarray
        Shape ``(n_noise, n_features)`` — competing (noise) feature vectors.
    backend : str, optional
        ``"auto"`` selects the fastest available backend (Rust engine when
        present, otherwise the NumPy reference); ``"python"``, ``"rust"``,
        ``"julia"``, ``"go"`` and ``"mojo"`` force a specific path.

    Returns
    -------
    float
        Isolation distance; ``nan`` when ``n_cluster < 2`` or fewer noise points
        than cluster points are supplied.
    """
    cluster = np.ascontiguousarray(cluster, dtype=np.float64)
    noise = np.ascontiguousarray(noise, dtype=np.float64)
    n_c = cluster.shape[0]
    if n_c < 2 or noise.shape[0] < n_c:
        return float("nan")
    return _sq_dispatch("isolation_distance", cluster, noise, backend)


def l_ratio(
    cluster: np.ndarray[Any, Any], noise: np.ndarray[Any, Any], backend: str = "auto"
) -> float:
    """L-ratio cluster quality (Schmitzer-Torbert et al. 2005).

    The mean over noise points of the chi-squared survival weight
    ``exp(-½ (d²_Mahalanobis - n_features))`` (clamped to ``[0, 1]``), normalised
    by the cluster size — small for well-isolated clusters.

    Parameters
    ----------
    cluster : numpy.ndarray
        Shape ``(n_cluster, n_features)`` — the cluster's feature vectors.
    noise : numpy.ndarray
        Shape ``(n_noise, n_features)`` — competing (noise) feature vectors.
    backend : str, optional
        Forwarded to the polyglot dispatch (see :func:`isolation_distance`).

    Returns
    -------
    float
        L-ratio; ``nan`` when ``n_cluster < 2`` or no noise points are supplied.
    """
    cluster = np.ascontiguousarray(cluster, dtype=np.float64)
    noise = np.ascontiguousarray(noise, dtype=np.float64)
    n_c = cluster.shape[0]
    if n_c < 2 or noise.shape[0] == 0:
        return float("nan")
    return _sq_dispatch("l_ratio", cluster, noise, backend)


def silhouette_score(features: np.ndarray[Any, Any], labels: np.ndarray[Any, Any]) -> float:
    """Mean silhouette score. Rousseeuw 1987.

    Measures cluster separation: s_i = (b_i - a_i) / max(a_i, b_i).
    """
    n = features.shape[0]
    if n < 2:
        return 0.0
    classes = np.unique(labels)
    if len(classes) < 2:
        return 0.0
    scores = np.zeros(n)
    for i in range(n):
        own_class = labels[i]
        own_mask = labels == own_class
        other_classes = classes[classes != own_class]
        own_dists = np.sqrt(np.sum((features[own_mask] - features[i]) ** 2, axis=1))
        a_i = own_dists.sum() / max(own_mask.sum() - 1, 1)
        b_i = np.inf
        for c in other_classes:
            c_mask = labels == c
            c_dists = np.sqrt(np.sum((features[c_mask] - features[i]) ** 2, axis=1))
            b_i = min(b_i, c_dists.mean())
        scores[i] = (b_i - a_i) / max(a_i, b_i, 1e-30)
    return float(scores.mean())


def d_prime(cluster_a: np.ndarray[Any, Any], cluster_b: np.ndarray[Any, Any]) -> float:
    """d-prime (sensitivity index) between two clusters. Green & Swets 1966.

    Uses first principal axis for projection.
    """
    mu_a = cluster_a.mean(axis=0)
    mu_b = cluster_b.mean(axis=0)
    direction = mu_b - mu_a
    norm = np.linalg.norm(direction)
    if norm < 1e-30:
        return 0.0
    direction /= norm
    proj_a = cluster_a @ direction
    proj_b = cluster_b @ direction
    var_a = proj_a.var()
    var_b = proj_b.var()
    pooled_std = np.sqrt(0.5 * (var_a + var_b))
    if pooled_std < 1e-30:
        return 0.0
    return float(abs(proj_a.mean() - proj_b.mean()) / pooled_std)


def isi_violation_rate(
    binary_train: np.ndarray[Any, Any], dt: float = 0.001, refractory_ms: float = 1.5
) -> float:
    """ISI violation rate: fraction of ISIs below refractory period. Hill et al. 2011."""
    intervals = isi(binary_train, dt)
    if intervals.size == 0:
        return 0.0
    ref = refractory_ms / 1000.0
    return float(np.sum(intervals < ref) / intervals.size)


def presence_ratio(binary_train: np.ndarray[Any, Any], n_bins: int = 100) -> float:
    """Presence ratio: fraction of time bins containing at least one spike. IBL 2019."""
    bin_size = max(1, binary_train.size // n_bins)
    counts = bin_spike_train(binary_train, bin_size)
    return float(np.sum(counts > 0) / max(counts.size, 1))


def amplitude_cutoff(amplitudes: np.ndarray[Any, Any], bins: int = 100) -> float:
    """Amplitude cutoff estimate. Hill et al. 2011.

    Fraction of spikes estimated to be missing below the amplitude histogram peak.
    """
    if amplitudes.size < 10:
        return float("nan")
    hist, edges = np.histogram(amplitudes, bins=bins)
    peak_idx = np.argmax(hist)
    if peak_idx == 0:
        return 0.5
    left_count = hist[:peak_idx].sum()
    right_count = hist[peak_idx:].sum()
    # Every amplitude lands in exactly one bin, so total == len(amplitudes) >= 10
    # after the early-return guard above — it is always positive.
    total = left_count + right_count
    estimated_missing = max(0, right_count - left_count)
    return float(estimated_missing / (total + estimated_missing))


def snr(waveforms: np.ndarray[Any, Any]) -> float:
    """Signal-to-noise ratio of spike waveforms. Suner et al. 2005.

    waveforms: (n_spikes, n_samples). SNR = peak_amplitude / noise_std.
    """
    if waveforms.ndim < 2 or waveforms.shape[0] < 2:
        return float("nan")
    mean_wf = waveforms.mean(axis=0)
    peak = np.max(np.abs(mean_wf))
    noise_std = waveforms.std(axis=0).mean()
    if noise_std < 1e-30:
        return float("inf")
    return float(peak / noise_std)


def nn_hit_rate(cluster: np.ndarray[Any, Any], noise: np.ndarray[Any, Any], k: int = 4) -> float:
    """Nearest-neighbor hit rate. Chung et al. 2017.

    Fraction of cluster points whose k nearest neighbors are also in the cluster.
    """
    n_c = cluster.shape[0]
    if n_c < k + 1:
        return float("nan")
    all_points = np.vstack([cluster, noise])
    all_labels = np.concatenate([np.ones(n_c), np.zeros(noise.shape[0])])
    hits = 0
    for i in range(n_c):
        dists = np.sqrt(np.sum((all_points - cluster[i]) ** 2, axis=1))
        dists[i] = np.inf
        nn_idx = np.argpartition(dists, k)[:k]
        if np.all(all_labels[nn_idx] == 1):
            hits += 1
    return float(hits / n_c)


def drift_metric(
    waveforms: np.ndarray[Any, Any], timestamps: np.ndarray[Any, Any], n_bins: int = 10
) -> float:
    """Waveform drift metric. IBL 2019.

    Measures change in mean waveform amplitude over time.
    """
    if waveforms.ndim < 2 or waveforms.shape[0] < n_bins:
        return float("nan")
    amplitudes = np.max(np.abs(waveforms), axis=1)
    sorted_idx = np.argsort(timestamps)
    amplitudes = amplitudes[sorted_idx]
    bin_size = len(amplitudes) // n_bins
    means_list: list[Any] = []
    for i in range(n_bins):
        chunk = amplitudes[i * bin_size : (i + 1) * bin_size]
        means_list.append(chunk.mean())
    means = np.array(means_list)
    if means.std() < 1e-30:
        return 0.0
    return float((means.max() - means.min()) / means.mean())
