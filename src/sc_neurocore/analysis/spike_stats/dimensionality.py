# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Dimensionality reduction for spike train populations

"""Dimensionality reduction for spike train populations.

PCA, demixed PCA (Kobak et al. 2016) and factor analysis (Rubin & Thayer 1982)
on binned spike-count matrices. The covariance eigendecomposition is taken from
LAPACK (``numpy.linalg.eigh``) with eigenvalues in descending order and
**sign-canonicalised** eigenvectors — each component's largest-magnitude entry is
made positive — so the projections are deterministic and reproducible across the
polyglot chain. Factor analysis uses a deterministic PCA-based initialisation
(replacing a random one) and Cholesky solves for its symmetric positive-definite
``M`` and ``E[zzᵀ]`` systems instead of explicit inverses.

The covariance eigendecomposition and the factor-analysis EM run across five
backends (NumPy / Rust / Julia / Go / Mojo) — LAPACK in NumPy / Julia, the
``nalgebra`` symmetric solver in Rust, an accurate cyclic-Jacobi solver where no
LAPACK is linked (Go / Mojo) — agreeing to floating-point round-off, selected via
``backend=``. ``backend="auto"`` resolves to the NumPy/LAPACK reference: dense
symmetric eigendecomposition is LAPACK's strength, and the compiled backends
(provided for cross-language parity and portability) do not beat it here.
"""

from __future__ import annotations

import ctypes as _ctypes
import importlib as _importlib
import importlib.util as _importlib_util
import os as _os
from typing import Any

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from .basic import bin_spike_train

_DimResult = tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]


def _accel_path(*parts: str) -> str:
    """Absolute path to a backend asset under the ``accel`` tree."""
    root = _os.path.dirname(_os.path.dirname(_os.path.dirname(__file__)))
    return _os.path.join(root, "accel", *parts)


def _canonical_sign(components: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Flip each column so its largest-magnitude entry is positive.

    Eigenvectors are defined only up to sign; fixing the sign by the dominant
    entry makes the projections identical across every backend.
    """
    if components.size == 0:
        return components
    pivot = np.argmax(np.abs(components), axis=0)
    signs = np.sign(components[pivot, np.arange(components.shape[1])])
    signs[signs == 0] = 1.0
    fixed: np.ndarray[Any, Any] = components * signs
    return fixed


# ── binning helpers (shared by the reference and the matrix backends) ──


def _pca_matrix(
    trains: list[np.ndarray[Any, Any]], bin_size: int
) -> tuple[np.ndarray[Any, Any], int, int]:
    """Mean-centred binned count matrix ``(n_neurons, min_bins)`` for PCA / FA.

    Callers guarantee a non-empty train list, and ``bin_spike_train`` always
    yields at least one bin, so ``min_bins`` is always positive here.
    """
    binned = [bin_spike_train(t, bin_size).astype(np.float64) for t in trains]
    min_bins = min(b.size for b in binned)
    mat = np.array([b[:min_bins] for b in binned])
    mat = mat - mat.mean(axis=1, keepdims=True)
    return np.ascontiguousarray(mat), mat.shape[0], min_bins


def _demixed_matrix(
    trains_by_condition: dict[int, list[np.ndarray[Any, Any]]], bin_size: int
) -> tuple[np.ndarray[Any, Any], int, int] | None:
    """Grand-mean-centred condition-mean matrix ``(n_conditions, min_bins)``."""
    all_means: list[np.ndarray[Any, Any]] = []
    for _cond, trains in sorted(trains_by_condition.items()):
        binned = [bin_spike_train(t, bin_size).astype(np.float64) for t in trains]
        min_bins = min((b.size for b in binned), default=0)
        if min_bins == 0:
            continue
        mat = np.array([b[:min_bins] for b in binned])
        all_means.append(mat.mean(axis=0))
    if len(all_means) < 2:
        return None
    min_bins = min(m.size for m in all_means)
    mean_mat = np.array([m[:min_bins] for m in all_means])
    mean_mat = mean_mat - mean_mat.mean(axis=0, keepdims=True)
    return np.ascontiguousarray(mean_mat), mean_mat.shape[0], min_bins


# ── NumPy reference cores (operate on a pre-centred matrix) ────────────


def _pca_from_matrix(mat: np.ndarray[Any, Any], n_components: int) -> _DimResult:
    """PCA of a centred ``(n_neurons, n_bins)`` matrix → ``(projected, explained)``.

    The covariance is the unbiased (``ddof=1``) sample covariance of the already
    mean-centred rows; callers handle the single-neuron case before dispatch.
    """
    _d, t = mat.shape
    cov = mat @ mat.T / max(t - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1][:n_components]
    components = _canonical_sign(eigvecs[:, order])
    projected = components.T @ mat
    total = eigvals.sum()
    explained = eigvals[order] / total if total > 0 else eigvals[order]
    return projected, explained


def _demixed_from_matrix(mean_mat: np.ndarray[Any, Any], n_components: int) -> _DimResult:
    """Demixed PCA of a centred ``(n_conditions, n_bins)`` matrix."""
    n_cond = mean_mat.shape[0]
    cov = mean_mat.T @ mean_mat / n_cond
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1][:n_components]
    components = _canonical_sign(eigvecs[:, order])
    projected = mean_mat @ components
    total = eigvals.sum()
    explained = eigvals[order] / total if total > 0 else eigvals[order]
    return projected, explained


def _fa_from_matrix(mat: np.ndarray[Any, Any], n_factors: int, n_iter: int) -> _DimResult:
    """Factor analysis EM of a centred ``(n_neurons, n_bins)`` matrix.

    The loadings start from a deterministic PCA initialisation (top eigenvectors
    of the sample covariance scaled by ``sqrt`` of the eigenvalues, sign-fixed),
    and each EM step solves its symmetric positive-definite ``M`` and ``E[zzᵀ]``
    systems through Cholesky factorisations rather than explicit inverses.
    """
    _d, t = mat.shape
    cov = mat @ mat.T / t
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1][:n_factors]
    loadings = _canonical_sign(eigvecs[:, order] * np.sqrt(np.clip(eigvals[order], 0.0, None)))
    nf = loadings.shape[1]
    eye_nf = np.eye(nf)
    psi = np.diag(cov).copy()
    for _ in range(n_iter):
        psi_inv = 1.0 / (psi + 1e-10)
        m = loadings.T @ (psi_inv[:, None] * loadings) + eye_nf
        cf = cho_factor(m, lower=True)
        m_inv = cho_solve(cf, eye_nf)
        beta = cho_solve(cf, (loadings * psi_inv[:, None]).T)
        ez = beta @ mat
        ezzt = nf * m_inv + ez @ ez.T / t
        mat_ez_t = mat @ ez.T / t
        cf2 = cho_factor(ezzt, lower=True)
        loadings = cho_solve(cf2, mat_ez_t.T).T
        psi = np.clip(np.diag(cov - loadings @ ez @ mat.T / t), 1e-6, None)
    return loadings, psi


# ── backend loaders ───────────────────────────────────────────────────


def _load_rust_dim(name: str) -> Any | None:
    """Return a Rust dimensionality entry point, or ``None`` when absent."""
    try:
        return getattr(_importlib.import_module("sc_neurocore_engine"), name)
    except (ImportError, AttributeError):
        return None


_rust_pca: Any | None = _load_rust_dim("py_pca_components")
_rust_demixed: Any | None = _load_rust_dim("py_demixed_components")
_rust_fa: Any | None = _load_rust_dim("py_factor_loadings")

_julia_dim: Any | None = None
_go_dim_lib: Any | None = None
_mojo_dim_lib: Any | None = None


def _ensure_julia_dim() -> bool:
    """Lazy-load the Julia dimensionality module, ``True`` when available."""
    global _julia_dim
    if _julia_dim is not None:
        return True
    if _importlib_util.find_spec("juliacall") is None:
        return False
    jl_path = _accel_path("julia", "analysis", "dimensionality.jl")
    if not _os.path.isfile(jl_path):
        return False
    jl = _importlib.import_module("juliacall").Main
    jl.include(jl_path)
    _julia_dim = jl.DimensionalityAccel
    return True


_GO_SIG = [
    _ctypes.POINTER(_ctypes.c_double),
    _ctypes.c_int,
    _ctypes.c_int,
    _ctypes.c_int,
    _ctypes.POINTER(_ctypes.c_double),
    _ctypes.POINTER(_ctypes.c_double),
]


def _ensure_go_dim() -> bool:
    """Lazy-load the Go dimensionality c-shared library, ``True`` when available."""
    global _go_dim_lib
    if _go_dim_lib is not None:
        return True
    so_path = _accel_path("go", "dimensionality", "libdimensionality.so")
    if not _os.path.isfile(so_path):
        return False
    try:
        lib: Any = _ctypes.CDLL(so_path)
    except OSError:
        return False
    names = ("pca_from_matrix_c", "demixed_from_matrix_c", "factor_analysis_c")
    if any(getattr(lib, n, None) is None for n in names):
        return False
    lib.pca_from_matrix_c.argtypes = _GO_SIG
    lib.pca_from_matrix_c.restype = None
    lib.demixed_from_matrix_c.argtypes = _GO_SIG
    lib.demixed_from_matrix_c.restype = None
    lib.factor_analysis_c.argtypes = _GO_SIG + [_ctypes.c_int]
    lib.factor_analysis_c.restype = None
    _go_dim_lib = lib
    return True


def _ensure_mojo_dim() -> bool:
    """Lazy-load the Mojo dimensionality shared library, ``True`` when available."""
    global _mojo_dim_lib
    if _mojo_dim_lib is not None:
        return True
    so_path = _accel_path("mojo", "kernels", "libdimensionality.so")
    if not _os.path.isfile(so_path):
        return False
    try:
        lib: Any = _ctypes.CDLL(so_path)
    except OSError:
        return False
    names = ("pca_from_matrix_c", "demixed_from_matrix_c", "factor_analysis_c")
    if any(getattr(lib, n, None) is None for n in names):
        return False
    lib.pca_from_matrix_c.argtypes = [_ctypes.c_int64] * 6
    lib.pca_from_matrix_c.restype = None
    lib.demixed_from_matrix_c.argtypes = [_ctypes.c_int64] * 6
    lib.demixed_from_matrix_c.restype = None
    lib.factor_analysis_c.argtypes = [_ctypes.c_int64] * 7
    lib.factor_analysis_c.restype = None
    _mojo_dim_lib = lib
    return True


# ── dispatch ───────────────────────────────────────────────────────────

_DIM_BACKENDS = ("auto", "python", "rust", "julia", "go", "mojo")


def _check_backend(backend: str) -> None:
    if backend not in _DIM_BACKENDS:
        raise ValueError(f"dimensionality backend {backend!r} is not available")


def _pca_dispatch(mat: np.ndarray[Any, Any], n_components: int, backend: str) -> _DimResult:
    _check_backend(backend)
    d, t = mat.shape
    nc = min(n_components, d)
    if backend == "rust":
        if _rust_pca is None:
            raise RuntimeError("Rust dimensionality backend is not available in this environment")
        proj, expl = _rust_pca(mat, n_components)
        return np.asarray(proj).reshape(nc, t), np.asarray(expl)
    if backend == "julia":
        if not _ensure_julia_dim():
            raise RuntimeError("Julia dimensionality backend is not available")
        mod: Any = _julia_dim
        proj, expl = mod.pca_from_matrix(mat, n_components)
        return np.asarray(proj), np.asarray(expl)
    if backend == "go":
        if not _ensure_go_dim():
            raise RuntimeError("Go dimensionality backend is not available")
        lib: Any = _go_dim_lib
        buf = np.ascontiguousarray(mat, dtype=np.float64).reshape(-1)
        proj = np.zeros(nc * t, dtype=np.float64)
        expl = np.zeros(nc, dtype=np.float64)
        lib.pca_from_matrix_c(
            buf.ctypes.data_as(_ctypes.POINTER(_ctypes.c_double)), d, t, nc,
            proj.ctypes.data_as(_ctypes.POINTER(_ctypes.c_double)),
            expl.ctypes.data_as(_ctypes.POINTER(_ctypes.c_double)),
        )
        return proj.reshape(nc, t), expl
    if backend == "mojo":
        if not _ensure_mojo_dim():
            raise RuntimeError("Mojo dimensionality backend is not available")
        lib2: Any = _mojo_dim_lib
        buf = np.ascontiguousarray(mat, dtype=np.float64).reshape(-1)
        proj = np.zeros(nc * t, dtype=np.float64)
        expl = np.zeros(nc, dtype=np.float64)
        lib2.pca_from_matrix_c(buf.ctypes.data, d, t, nc, proj.ctypes.data, expl.ctypes.data)
        return proj.reshape(nc, t), expl
    return _pca_from_matrix(mat, n_components)


def _demixed_dispatch(mean_mat: np.ndarray[Any, Any], n_components: int, backend: str) -> _DimResult:
    _check_backend(backend)
    n_cond, t = mean_mat.shape
    nc = min(n_components, t)
    if backend == "rust":
        if _rust_demixed is None:
            raise RuntimeError("Rust dimensionality backend is not available in this environment")
        proj, expl = _rust_demixed(mean_mat, n_components)
        return np.asarray(proj).reshape(n_cond, nc), np.asarray(expl)
    if backend == "julia":
        if not _ensure_julia_dim():
            raise RuntimeError("Julia dimensionality backend is not available")
        mod: Any = _julia_dim
        proj, expl = mod.demixed_from_matrix(mean_mat, n_components)
        return np.asarray(proj), np.asarray(expl)
    if backend == "go":
        if not _ensure_go_dim():
            raise RuntimeError("Go dimensionality backend is not available")
        lib: Any = _go_dim_lib
        buf = np.ascontiguousarray(mean_mat, dtype=np.float64).reshape(-1)
        proj = np.zeros(n_cond * nc, dtype=np.float64)
        expl = np.zeros(nc, dtype=np.float64)
        lib.demixed_from_matrix_c(
            buf.ctypes.data_as(_ctypes.POINTER(_ctypes.c_double)), n_cond, t, nc,
            proj.ctypes.data_as(_ctypes.POINTER(_ctypes.c_double)),
            expl.ctypes.data_as(_ctypes.POINTER(_ctypes.c_double)),
        )
        return proj.reshape(n_cond, nc), expl
    if backend == "mojo":
        if not _ensure_mojo_dim():
            raise RuntimeError("Mojo dimensionality backend is not available")
        lib2: Any = _mojo_dim_lib
        buf = np.ascontiguousarray(mean_mat, dtype=np.float64).reshape(-1)
        proj = np.zeros(n_cond * nc, dtype=np.float64)
        expl = np.zeros(nc, dtype=np.float64)
        lib2.demixed_from_matrix_c(
            buf.ctypes.data, n_cond, t, nc, proj.ctypes.data, expl.ctypes.data
        )
        return proj.reshape(n_cond, nc), expl
    return _demixed_from_matrix(mean_mat, n_components)


def _fa_dispatch(mat: np.ndarray[Any, Any], n_factors: int, n_iter: int, backend: str) -> _DimResult:
    _check_backend(backend)
    d, _t = mat.shape
    nf = min(n_factors, d)
    if backend == "rust":
        if _rust_fa is None:
            raise RuntimeError("Rust dimensionality backend is not available in this environment")
        loadings, psi = _rust_fa(mat, n_factors, n_iter)
        return np.asarray(loadings).reshape(d, nf), np.asarray(psi)
    if backend == "julia":
        if not _ensure_julia_dim():
            raise RuntimeError("Julia dimensionality backend is not available")
        mod: Any = _julia_dim
        loadings, psi = mod.factor_analysis(mat, n_factors, n_iter)
        return np.asarray(loadings), np.asarray(psi)
    if backend == "go":
        if not _ensure_go_dim():
            raise RuntimeError("Go dimensionality backend is not available")
        lib: Any = _go_dim_lib
        buf = np.ascontiguousarray(mat, dtype=np.float64).reshape(-1)
        loadings = np.zeros(d * nf, dtype=np.float64)
        psi = np.zeros(d, dtype=np.float64)
        lib.factor_analysis_c(
            buf.ctypes.data_as(_ctypes.POINTER(_ctypes.c_double)), d, _t, nf,
            loadings.ctypes.data_as(_ctypes.POINTER(_ctypes.c_double)),
            psi.ctypes.data_as(_ctypes.POINTER(_ctypes.c_double)), n_iter,
        )
        return loadings.reshape(d, nf), psi
    if backend == "mojo":
        if not _ensure_mojo_dim():
            raise RuntimeError("Mojo dimensionality backend is not available")
        lib2: Any = _mojo_dim_lib
        buf = np.ascontiguousarray(mat, dtype=np.float64).reshape(-1)
        loadings = np.zeros(d * nf, dtype=np.float64)
        psi = np.zeros(d, dtype=np.float64)
        lib2.factor_analysis_c(
            buf.ctypes.data, d, _t, nf, loadings.ctypes.data, psi.ctypes.data, n_iter
        )
        return loadings.reshape(d, nf), psi
    return _fa_from_matrix(mat, n_factors, n_iter)


# ── public API ─────────────────────────────────────────────────────────


def spike_train_pca(
    trains: list[np.ndarray[Any, Any]],
    n_components: int = 3,
    bin_size: int = 10,
    backend: str = "auto",
) -> _DimResult:
    """PCA on a binned spike-count matrix (neurons × time bins).

    Parameters
    ----------
    trains : list of numpy.ndarray
        Binary spike trains, one per neuron.
    n_components : int, optional
        Number of principal components to keep.
    bin_size : int, optional
        Number of timesteps per bin.
    backend : str, optional
        ``"auto"`` selects the fastest measured path — the NumPy/LAPACK reference,
        which outperforms the compiled dense-eigendecomposition backends;
        ``"python"``, ``"rust"``, ``"julia"``, ``"go"`` and ``"mojo"`` force a
        specific path (the compiled backends are kept for cross-language parity
        and portability).

    Returns
    -------
    tuple of numpy.ndarray
        ``(projected, explained_variance_ratio)``; ``projected`` is
        ``(n_components, n_bins)``. Empty arrays when no trains are supplied.
    """
    _check_backend(backend)
    if not trains:
        return np.array([[]]), np.array([])
    mat, d, _t = _pca_matrix(trains, bin_size)
    if d < 2:
        return mat[:1], np.array([1.0])
    return _pca_dispatch(mat, n_components, backend)


def demixed_pca(
    trains_by_condition: dict[int, list[np.ndarray[Any, Any]]],
    n_components: int = 3,
    bin_size: int = 10,
    backend: str = "auto",
) -> _DimResult:
    """Demixed PCA (Kobak et al. 2016) on condition-mean activity.

    Separates condition-dependent variance by projecting the grand-mean-centred
    condition means onto the leading eigenvectors of their covariance.

    Parameters
    ----------
    trains_by_condition : dict
        ``{condition_id: [binary trains per neuron]}``.
    n_components : int, optional
        Number of components to keep.
    bin_size : int, optional
        Number of timesteps per bin.
    backend : str, optional
        See :func:`spike_train_pca`.

    Returns
    -------
    tuple of numpy.ndarray
        ``(projected, explained_variance_ratio)``; empty arrays when fewer than
        two conditions carry data.
    """
    _check_backend(backend)
    prep = _demixed_matrix(trains_by_condition, bin_size)
    if prep is None:
        return np.array([[]]), np.array([])
    mean_mat, _n_cond, _t = prep
    return _demixed_dispatch(mean_mat, n_components, backend)


def factor_analysis(
    trains: list[np.ndarray[Any, Any]],
    n_factors: int = 3,
    bin_size: int = 10,
    n_iter: int = 50,
    backend: str = "auto",
) -> _DimResult:
    """Factor analysis via EM (Rubin & Thayer 1982) on binned activity.

    The loadings start from a deterministic PCA initialisation (so the result is
    reproducible and seed-independent) and each EM step solves its symmetric
    positive-definite systems by Cholesky factorisation.

    Parameters
    ----------
    trains : list of numpy.ndarray
        Binary spike trains, one per neuron.
    n_factors : int, optional
        Number of latent factors.
    bin_size : int, optional
        Number of timesteps per bin.
    n_iter : int, optional
        Number of EM iterations.
    backend : str, optional
        See :func:`spike_train_pca`.

    Returns
    -------
    tuple of numpy.ndarray
        ``(loadings, uniquenesses)`` of shapes ``(n_neurons, n_factors)`` and
        ``(n_neurons,)``.
    """
    _check_backend(backend)
    if not trains:
        return np.array([]), np.array([])
    mat, _d, _t = _pca_matrix(trains, bin_size)
    return _fa_dispatch(mat, n_factors, n_iter, backend)
