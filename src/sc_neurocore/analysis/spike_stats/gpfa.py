# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — GPFA -- Gaussian Process Factor Analysis

"""GPFA -- Gaussian Process Factor Analysis.

Yu, Cunningham, Santhanam, Ryu, Shenoy, Sahani (2009)
"Gaussian-process factor analysis for low-dimensional single-trial
analysis of neural population activity." J. Neurophysiol. 102:614-635.

Extracts smooth low-dimensional latent trajectories from binned spike
counts via EM with squared-exponential GP priors on latent dimensions.
"""

from __future__ import annotations

import ctypes as _ctypes
import importlib as _importlib
import importlib.util as _importlib_util
import os as _os
from typing import Any

import numpy as np

from .basic import bin_spike_train


def _accel_path(*parts: str) -> str:
    """Absolute path to a backend asset under the ``accel`` tree."""
    root = _os.path.dirname(_os.path.dirname(_os.path.dirname(__file__)))
    return _os.path.join(root, "accel", *parts)


def _as_double_ptr(arr: np.ndarray[Any, Any]) -> Any:
    """Return a ``ctypes`` double pointer to a contiguous float64 array."""
    return arr.ctypes.data_as(_ctypes.POINTER(_ctypes.c_double))


def _load_rust_gpfa_em() -> Any | None:
    """Return the Rust GPFA EM entry point, or ``None`` when the engine is absent."""
    try:
        return _importlib.import_module("sc_neurocore_engine").py_gpfa_em
    except (ImportError, AttributeError):
        return None


# Rust acceleration backend — probed at import (the wheel import is cheap).
_rust_gpfa_em: Any | None = _load_rust_gpfa_em()

# Julia / Go / Mojo backends — loaded lazily on first explicit request.
_julia_gpfa: Any | None = None
_go_gpfa_lib: Any | None = None
_mojo_gpfa_lib: Any | None = None


def _ensure_julia_gpfa() -> bool:
    """Lazy-load the Julia GPFA module, returning ``True`` when available."""
    global _julia_gpfa
    if _julia_gpfa is not None:
        return True
    if _importlib_util.find_spec("juliacall") is None:
        return False
    jl_path = _accel_path("julia", "analysis", "gpfa.jl")
    if not _os.path.isfile(jl_path):
        return False
    jl = _importlib.import_module("juliacall").Main
    jl.include(jl_path)
    _julia_gpfa = jl.GpfaAccel
    return True


def _ensure_go_gpfa() -> bool:
    """Lazy-load the Go GPFA c-shared library, returning ``True`` when available."""
    global _go_gpfa_lib
    if _go_gpfa_lib is not None:
        return True
    so_path = _accel_path("go", "gpfa", "libgpfa.so")
    if not _os.path.isfile(so_path):
        return False
    try:
        lib = _ctypes.CDLL(so_path)
    except OSError:
        return False
    fn = getattr(lib, "gpfa_em_c", None)
    if fn is None:
        return False
    dp = _ctypes.POINTER(_ctypes.c_double)
    fn.argtypes = [dp] * 5 + [_ctypes.c_int] * 4 + [_ctypes.c_double] + [dp] * 3
    fn.restype = None
    _go_gpfa_lib = lib
    return True


def _ensure_mojo_gpfa() -> bool:
    """Lazy-load the Mojo GPFA shared library, returning ``True`` when available.

    Built once via::

        cd src/sc_neurocore/accel/mojo/kernels
        mojo build --emit shared-lib -o libgpfa.so gpfa.mojo

    Per ``feedback_mojo_026_ffi_pattern``, the ``@export`` signature accepts no
    parametric pointer types, so every array is passed as a raw ``int64`` address
    (``numpy.ndarray.ctypes.data``) and the lone ``tol`` scalar as a ``c_double``.
    """
    global _mojo_gpfa_lib
    if _mojo_gpfa_lib is not None:
        return True
    so_path = _accel_path("mojo", "kernels", "libgpfa.so")
    if not _os.path.isfile(so_path):
        return False
    try:
        lib = _ctypes.CDLL(so_path)
    except OSError:
        return False
    fn = getattr(lib, "gpfa_em_c", None)
    if fn is None:
        return False
    # 13 args: 5 input addresses + 4 size scalars + tol + 3 output addresses.
    fn.argtypes = [_ctypes.c_int64] * 5 + [_ctypes.c_int64] * 4 + [_ctypes.c_double] + [_ctypes.c_int64] * 3
    fn.restype = None
    _mojo_gpfa_lib = lib
    return True


def _gp_kernel(n_bins: int, tau: float, sigma: float = 1.0) -> np.ndarray[Any, Any]:
    """Squared-exponential kernel matrix for *n_bins* time points."""
    t = np.arange(n_bins, dtype=np.float64)
    diff = t[:, None] - t[None, :]
    return sigma**2 * np.exp(-0.5 * diff**2 / (tau**2 + 1e-12))


def _gpfa_e_step(
    Y: np.ndarray[Any, Any],
    C: np.ndarray[Any, Any],
    d: np.ndarray[Any, Any],
    R: np.ndarray[Any, Any],
    K_all: list[np.ndarray[Any, Any]],
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Posterior p(x|y) for each latent dimension jointly."""
    n_neurons, n_bins = Y.shape
    n_latents = C.shape[1]

    # Build block-diagonal K (n_latents*n_bins x n_latents*n_bins)
    KT = n_latents * n_bins
    K_big = np.zeros((KT, KT))
    for j in range(n_latents):
        sl = slice(j * n_bins, (j + 1) * n_bins)
        K_big[sl, sl] = K_all[j]

    # Observation model: Y_centered = C x + noise
    Y_centered = Y - d[:, None]  # (n_neurons, n_bins)

    # Kronecker structure: C_big = I_T kron C, R_big = I_T kron R
    # Posterior: Sigma_post = (K^-1 + C_big^T R_big^-1 C_big)^-1
    # Mean: mu_post = Sigma_post C_big^T R_big^-1 y_vec
    R_inv = np.diag(1.0 / (np.diag(R) + 1e-10))  # (n_neurons, n_neurons)

    # Exploit temporal structure: work per-timepoint then combine
    # C^T R^{-1} C is (n_latents x n_latents), same every timepoint
    CtRinvC = C.T @ R_inv @ C  # (n_latents, n_latents)
    CtRinv = C.T @ R_inv  # (n_latents, n_neurons)

    # Build the precision of the posterior in block form
    # For efficiency: Sigma_post^{-1}[j,k block] = K_j^{-1} delta_{jk} + CtRinvC[j,k] I_T
    # This is block-structured: n_latents blocks of (n_bins x n_bins)
    # Off-diagonal blocks are CtRinvC[j,k] * I_T
    # Diagonal blocks are K_j^{-1} + CtRinvC[j,j] * I_T

    prec = np.zeros((KT, KT))
    for j in range(n_latents):
        slj = slice(j * n_bins, (j + 1) * n_bins)
        K_j_inv = np.linalg.solve(K_all[j] + 1e-6 * np.eye(n_bins), np.eye(n_bins))
        prec[slj, slj] = K_j_inv + CtRinvC[j, j] * np.eye(n_bins)
        for k in range(n_latents):
            if k != j:
                slk = slice(k * n_bins, (k + 1) * n_bins)
                prec[slj, slk] = CtRinvC[j, k] * np.eye(n_bins)

    # y_vec -> (n_neurons * n_bins,) but we compute C^T R^{-1} y per timepoint
    rhs = np.zeros(KT)
    for t_idx in range(n_bins):
        v = CtRinv @ Y_centered[:, t_idx]  # (n_latents,)
        for j in range(n_latents):
            rhs[j * n_bins + t_idx] = v[j]

    # Solve for posterior mean
    x_vec = np.linalg.solve(prec + 1e-8 * np.eye(KT), rhs)
    x_post = x_vec.reshape(n_latents, n_bins)

    # Posterior covariance (for M-step sufficient statistics)
    Sigma_post = np.linalg.solve(prec + 1e-8 * np.eye(KT), np.eye(KT))

    # E[x x^T] per timepoint: sum over timepoints
    xx_post = np.zeros((n_latents, n_latents))
    for t_idx in range(n_bins):
        xt = x_post[:, t_idx]
        xx_post += np.outer(xt, xt)
        for j in range(n_latents):
            for k in range(n_latents):
                xx_post[j, k] += Sigma_post[j * n_bins + t_idx, k * n_bins + t_idx]

    return x_post, xx_post


def _gpfa_m_step(
    Y: np.ndarray[Any, Any], x_post: np.ndarray[Any, Any], xx_post: np.ndarray[Any, Any]
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Update C, d, R from sufficient statistics."""
    n_neurons, n_bins = Y.shape

    d_new = Y.mean(axis=1)
    Y_centered = Y - d_new[:, None]

    # C_new = (sum_t y_t x_t^T) (sum_t x_t x_t^T + Sigma)^{-1}
    Yx = Y_centered @ x_post.T  # (n_neurons, n_latents)
    C_new = np.linalg.solve(xx_post.T + 1e-8 * np.eye(xx_post.shape[0]), Yx.T).T

    # R_new = diag(1/T sum_t (y_t - d)(y_t - d)^T - C E[x y^T])
    YYt = Y_centered @ Y_centered.T / n_bins
    CxYt = C_new @ x_post @ Y_centered.T / n_bins
    R_diag = np.diag(YYt - CxYt)
    R_diag = np.clip(R_diag, 1e-6, None)
    R_new = np.diag(R_diag)

    return C_new, d_new, R_new


def _gpfa_log_likelihood(
    Y: np.ndarray[Any, Any],
    C: np.ndarray[Any, Any],
    d: np.ndarray[Any, Any],
    R: np.ndarray[Any, Any],
    K_all: list[np.ndarray[Any, Any]],
) -> np.float64:
    """Exact marginal Gaussian log likelihood for the GPFA observation model."""
    n_neurons, n_bins = Y.shape
    n_latents = C.shape[1]
    n_obs = n_neurons * n_bins
    n_state = n_latents * n_bins

    A = np.zeros((n_obs, n_state), dtype=np.float64)
    for t_idx in range(n_bins):
        row_start = t_idx * n_neurons
        for j in range(n_latents):
            col = j * n_bins + t_idx
            A[row_start : row_start + n_neurons, col] = C[:, j]

    K_big = np.zeros((n_state, n_state), dtype=np.float64)
    for j, kernel in enumerate(K_all):
        sl = slice(j * n_bins, (j + 1) * n_bins)
        K_big[sl, sl] = kernel

    R_big = np.kron(np.eye(n_bins, dtype=np.float64), R)
    cov = A @ K_big @ A.T + R_big
    cov = cov + 1e-8 * np.eye(n_obs, dtype=np.float64)

    y_centered = (Y - d[:, None]).T.reshape(n_obs)
    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0:
        raise np.linalg.LinAlgError("GPFA marginal covariance is not positive definite")
    quad = y_centered @ np.linalg.solve(cov, y_centered)
    return np.float64(-0.5 * (quad + logdet + n_obs * np.log(2.0 * np.pi)))


def gpfa_pca_init(
    Y: np.ndarray[Any, Any], n_latents: int, bin_ms: float
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Deterministic PCA initialisation of the GPFA parameters.

    The loading matrix ``C`` is the top ``n_latents`` left singular vectors of the
    centred data, scaled by their singular values; a fixed sign convention (each
    column's largest-magnitude entry made positive) makes the result reproducible
    across runs and BLAS/LAPACK implementations. This replaces the former random
    initialisation, so every backend can start the EM from an identical ``C``.

    Parameters
    ----------
    Y : numpy.ndarray
        Binned spike counts, shape ``(n_neurons, n_bins)``.
    n_latents : int
        Number of latent dimensions.
    bin_ms : float
        Bin width in milliseconds, used to set the GP timescales ``tau``.

    Returns
    -------
    tuple
        ``(C, d, R, tau)`` — loading matrix ``(n_neurons, n_latents)``, offset
        ``(n_neurons,)``, observation-noise covariance ``(n_neurons, n_neurons)``
        and GP timescales ``(n_latents,)``.
    """
    n_neurons, n_bins = Y.shape
    d = Y.mean(axis=1)
    y_centered = Y - d[:, None]
    u, s, _ = np.linalg.svd(y_centered, full_matrices=False)
    u = u[:, :n_latents]
    s = s[:n_latents]
    max_abs_row = np.argmax(np.abs(u), axis=0)
    signs = np.sign(u[max_abs_row, np.arange(u.shape[1])])
    signs[signs == 0] = 1.0
    c = u * signs * (s / np.sqrt(max(n_bins, 1)))
    r = np.diag(Y.var(axis=1) + 1e-4)
    tau = np.full(n_latents, bin_ms * 2.0)
    return c, d, r, tau


def gpfa_em(
    Y: np.ndarray[Any, Any],
    C0: np.ndarray[Any, Any],
    d0: np.ndarray[Any, Any],
    R0: np.ndarray[Any, Any],
    tau: np.ndarray[Any, Any],
    max_iter: int,
    tol: float,
) -> tuple[
    np.ndarray[Any, Any],
    np.ndarray[Any, Any],
    np.ndarray[Any, Any],
    np.ndarray[Any, Any],
    list[float],
]:
    """Run the GPFA EM loop from a fixed initialisation (NumPy reference floor).

    The GP timescales ``tau`` are held fixed, so the kernel matrices are constant
    across iterations. Returns the filtered trajectories together with the final
    parameters and the per-iteration marginal log-likelihoods.

    Parameters
    ----------
    Y : numpy.ndarray
        Binned spike counts, shape ``(n_neurons, n_bins)``.
    C0, d0, R0 : numpy.ndarray
        Initial loading matrix, offset and observation-noise covariance.
    tau : numpy.ndarray
        GP timescales, shape ``(n_latents,)``.
    max_iter : int
        Maximum EM iterations.
    tol : float
        Convergence tolerance on the log-likelihood increment.

    Returns
    -------
    tuple
        ``(trajectories, C, d, R, log_likelihoods)``.
    """
    n_latents = int(C0.shape[1])
    n_bins = int(Y.shape[1])
    k_all = [_gp_kernel(n_bins, float(tau[j])) for j in range(n_latents)]

    C = C0.astype(np.float64, copy=True)
    d = d0.astype(np.float64, copy=True)
    R = R0.astype(np.float64, copy=True)

    log_liks: list[float] = []
    x_post = np.zeros((n_latents, n_bins), dtype=np.float64)
    for _ in range(max_iter):
        x_post, xx_post = _gpfa_e_step(Y, C, d, R, k_all)
        C, d, R = _gpfa_m_step(Y, x_post, xx_post)
        log_liks.append(float(_gpfa_log_likelihood(Y, C, d, R, k_all)))
        if len(log_liks) > 1 and abs(log_liks[-1] - log_liks[-2]) < tol:
            break

    return x_post, C, d, R, log_liks


def _run_rust_gpfa_em(
    Y: np.ndarray[Any, Any],
    C0: np.ndarray[Any, Any],
    d0: np.ndarray[Any, Any],
    R0: np.ndarray[Any, Any],
    tau: np.ndarray[Any, Any],
    max_iter: int,
    tol: float,
) -> tuple[
    np.ndarray[Any, Any],
    np.ndarray[Any, Any],
    np.ndarray[Any, Any],
    np.ndarray[Any, Any],
    list[float],
]:
    """Dispatch the EM loop to the Rust engine and rebuild NumPy outputs."""
    assert _rust_gpfa_em is not None
    n_neurons, n_bins = Y.shape
    n_latents = int(C0.shape[1])
    x_flat, c_flat, d_out, r_diag, log_liks = _rust_gpfa_em(
        np.ascontiguousarray(Y, dtype=np.float64).reshape(-1),
        n_neurons,
        n_bins,
        np.ascontiguousarray(C0, dtype=np.float64).reshape(-1),
        np.ascontiguousarray(d0, dtype=np.float64),
        np.ascontiguousarray(np.diag(R0), dtype=np.float64),
        np.ascontiguousarray(tau, dtype=np.float64),
        n_latents,
        int(max_iter),
        float(tol),
    )
    return (
        np.asarray(x_flat, dtype=np.float64).reshape(n_latents, n_bins),
        np.asarray(c_flat, dtype=np.float64).reshape(n_neurons, n_latents),
        np.asarray(d_out, dtype=np.float64),
        np.diag(np.asarray(r_diag, dtype=np.float64)),
        [float(v) for v in log_liks],
    )


def _run_julia_gpfa_em(
    Y: np.ndarray[Any, Any],
    C0: np.ndarray[Any, Any],
    d0: np.ndarray[Any, Any],
    R0: np.ndarray[Any, Any],
    tau: np.ndarray[Any, Any],
    max_iter: int,
    tol: float,
) -> tuple[
    np.ndarray[Any, Any],
    np.ndarray[Any, Any],
    np.ndarray[Any, Any],
    np.ndarray[Any, Any],
    list[float],
]:
    """Dispatch the EM loop to the Julia backend and rebuild NumPy outputs."""
    assert _julia_gpfa is not None
    result = _julia_gpfa.gpfa_em(
        np.ascontiguousarray(Y, dtype=np.float64),
        np.ascontiguousarray(C0, dtype=np.float64),
        np.ascontiguousarray(d0, dtype=np.float64),
        np.ascontiguousarray(np.diag(R0), dtype=np.float64),
        np.ascontiguousarray(tau, dtype=np.float64),
        int(max_iter),
        float(tol),
    )
    return (
        np.asarray(result.trajectories, dtype=np.float64),
        np.asarray(result.C, dtype=np.float64),
        np.asarray(result.d, dtype=np.float64),
        np.diag(np.asarray(result.R_diag, dtype=np.float64)),
        [float(v) for v in result.log_liks],
    )


def _run_go_gpfa_em(
    Y: np.ndarray[Any, Any],
    C0: np.ndarray[Any, Any],
    d0: np.ndarray[Any, Any],
    R0: np.ndarray[Any, Any],
    tau: np.ndarray[Any, Any],
    max_iter: int,
    tol: float,
) -> tuple[
    np.ndarray[Any, Any],
    np.ndarray[Any, Any],
    np.ndarray[Any, Any],
    np.ndarray[Any, Any],
    list[float],
]:
    """Dispatch the EM loop to the Go c-shared backend and rebuild NumPy outputs."""
    assert _go_gpfa_lib is not None
    n_neurons, n_bins = Y.shape
    n_latents = int(C0.shape[1])
    y_buf = np.ascontiguousarray(Y, dtype=np.float64).reshape(-1)
    c0_buf = np.ascontiguousarray(C0, dtype=np.float64).reshape(-1)
    d0_buf = np.ascontiguousarray(d0, dtype=np.float64)
    r0_buf = np.ascontiguousarray(np.diag(R0), dtype=np.float64)
    tau_buf = np.ascontiguousarray(tau, dtype=np.float64)
    x_out = np.zeros(n_latents * n_bins, dtype=np.float64)
    params_out = np.zeros(n_neurons * n_latents + 2 * n_neurons, dtype=np.float64)
    loglik_out = np.zeros(max_iter + 1, dtype=np.float64)
    _go_gpfa_lib.gpfa_em_c(
        _as_double_ptr(y_buf),
        _as_double_ptr(c0_buf),
        _as_double_ptr(d0_buf),
        _as_double_ptr(r0_buf),
        _as_double_ptr(tau_buf),
        n_neurons,
        n_bins,
        n_latents,
        int(max_iter),
        _ctypes.c_double(float(tol)),
        _as_double_ptr(x_out),
        _as_double_ptr(params_out),
        _as_double_ptr(loglik_out),
    )
    c_end = n_neurons * n_latents
    n_iter = int(loglik_out[0])
    return (
        x_out.reshape(n_latents, n_bins),
        params_out[:c_end].reshape(n_neurons, n_latents).copy(),
        params_out[c_end : c_end + n_neurons].copy(),
        np.diag(params_out[c_end + n_neurons : c_end + 2 * n_neurons].copy()),
        [float(v) for v in loglik_out[1 : 1 + n_iter]],
    )


def _run_mojo_gpfa_em(
    Y: np.ndarray[Any, Any],
    C0: np.ndarray[Any, Any],
    d0: np.ndarray[Any, Any],
    R0: np.ndarray[Any, Any],
    tau: np.ndarray[Any, Any],
    max_iter: int,
    tol: float,
) -> tuple[
    np.ndarray[Any, Any],
    np.ndarray[Any, Any],
    np.ndarray[Any, Any],
    np.ndarray[Any, Any],
    list[float],
]:
    """Dispatch the EM loop to the Mojo c-shared backend and rebuild NumPy outputs.

    Mirrors :func:`_run_go_gpfa_em` but, per the Mojo 0.26 FFI rules, hands every
    buffer to the kernel as a raw ``int64`` address rather than a typed pointer.
    """
    assert _mojo_gpfa_lib is not None
    n_neurons, n_bins = Y.shape
    n_latents = int(C0.shape[1])
    y_buf = np.ascontiguousarray(Y, dtype=np.float64).reshape(-1)
    c0_buf = np.ascontiguousarray(C0, dtype=np.float64).reshape(-1)
    d0_buf = np.ascontiguousarray(d0, dtype=np.float64)
    r0_buf = np.ascontiguousarray(np.diag(R0), dtype=np.float64)
    tau_buf = np.ascontiguousarray(tau, dtype=np.float64)
    x_out = np.zeros(n_latents * n_bins, dtype=np.float64)
    params_out = np.zeros(n_neurons * n_latents + 2 * n_neurons, dtype=np.float64)
    loglik_out = np.zeros(max_iter + 1, dtype=np.float64)
    _mojo_gpfa_lib.gpfa_em_c(
        y_buf.ctypes.data,
        c0_buf.ctypes.data,
        d0_buf.ctypes.data,
        r0_buf.ctypes.data,
        tau_buf.ctypes.data,
        n_neurons,
        n_bins,
        n_latents,
        int(max_iter),
        float(tol),
        x_out.ctypes.data,
        params_out.ctypes.data,
        loglik_out.ctypes.data,
    )
    c_end = n_neurons * n_latents
    n_iter = int(loglik_out[0])
    return (
        x_out.reshape(n_latents, n_bins),
        params_out[:c_end].reshape(n_neurons, n_latents).copy(),
        params_out[c_end : c_end + n_neurons].copy(),
        np.diag(params_out[c_end + n_neurons : c_end + 2 * n_neurons].copy()),
        [float(v) for v in loglik_out[1 : 1 + n_iter]],
    )


def _gpfa_em_dispatch(
    Y: np.ndarray[Any, Any],
    C0: np.ndarray[Any, Any],
    d0: np.ndarray[Any, Any],
    R0: np.ndarray[Any, Any],
    tau: np.ndarray[Any, Any],
    max_iter: int,
    tol: float,
    backend: str,
) -> tuple[
    np.ndarray[Any, Any],
    np.ndarray[Any, Any],
    np.ndarray[Any, Any],
    np.ndarray[Any, Any],
    list[float],
]:
    """Run the GPFA EM loop on the requested backend; ``auto`` selects the fastest.

    The deterministic initialisation (see :func:`gpfa_pca_init`) lets every backend
    share an identical starting point. The Rust, Julia, Go and Mojo backends bind the
    same ``gpfa_em`` contract and agree with the NumPy reference up to floating-point
    round-off.

    Backend selection under ``auto`` is data-driven (see
    ``benchmarks/results/bench_gpfa.json``): GPFA's inner loop is dense linear
    algebra — Gaussian elimination, marginal-covariance ``slogdet`` and solves — and
    the NumPy reference dispatches those to LAPACK, which outpaces the hand-written
    Gauss-Jordan kernels of the compiled backends (measured ~6× faster than Rust/Mojo,
    ~11× faster than Go). ``auto`` therefore runs the NumPy path; the Rust, Julia, Go
    and Mojo backends remain available by name for cross-language parity checking and
    for embedding in environments without a tuned NumPy/LAPACK stack.
    """
    if backend not in ("auto", "python", "rust", "julia", "go", "mojo"):
        raise ValueError(f"GPFA backend {backend!r} is not available")
    if backend in ("auto", "python"):
        return gpfa_em(Y, C0, d0, R0, tau, max_iter, tol)
    if backend == "rust":
        if _rust_gpfa_em is None:
            raise RuntimeError("Rust GPFA backend is not available in this environment")
        return _run_rust_gpfa_em(Y, C0, d0, R0, tau, max_iter, tol)
    if backend == "julia":
        if not _ensure_julia_gpfa():
            raise RuntimeError("Julia GPFA backend is not available")
        return _run_julia_gpfa_em(Y, C0, d0, R0, tau, max_iter, tol)
    if backend == "go":
        if not _ensure_go_gpfa():
            raise RuntimeError("Go GPFA backend is not available")
        return _run_go_gpfa_em(Y, C0, d0, R0, tau, max_iter, tol)
    if not _ensure_mojo_gpfa():
        raise RuntimeError("Mojo GPFA backend is not available")
    return _run_mojo_gpfa_em(Y, C0, d0, R0, tau, max_iter, tol)


def _bin_trains(
    trains: list[np.ndarray[Any, Any]], bin_ms: float, dt: float
) -> np.ndarray[Any, Any]:
    """Bin parallel spike trains into an aligned ``(n_neurons, n_bins)`` matrix."""
    bin_steps = max(1, int(bin_ms / (dt * 1000)))
    binned = [bin_spike_train(t, bin_steps).astype(np.float64) for t in trains]
    min_bins = min(b.size for b in binned)
    return np.array([b[:min_bins] for b in binned])


def gpfa(
    trains: list[np.ndarray[Any, Any]],
    n_latents: int = 3,
    bin_ms: float = 20.0,
    dt: float = 0.001,
    max_iter: int = 50,
    tol: float = 1e-4,
    seed: int = 42,
    backend: str = "auto",
) -> dict[str, Any]:
    """Extract smooth latent trajectories from parallel spike trains via EM.

    The initialisation is deterministic (PCA, see :func:`gpfa_pca_init`), so the
    result is reproducible and identical across acceleration backends up to
    floating-point round-off. ``seed`` is retained for API compatibility but no
    longer affects the result.

    Parameters
    ----------
    trains : list of numpy.ndarray
        Parallel binary/integer spike trains.
    n_latents : int, optional
        Number of latent dimensions (clamped to ``min(n_neurons, n_bins)``).
    bin_ms, dt : float, optional
        Bin width (ms) and simulation timestep (s).
    max_iter, tol : int, float, optional
        EM iteration cap and log-likelihood convergence tolerance.
    seed : int, optional
        Retained for API compatibility; initialisation is deterministic.
    backend : str, optional
        ``"auto"`` selects the fastest measured backend, which for GPFA is the
        LAPACK-backed NumPy reference (``"python"``); ``"rust"``, ``"julia"``,
        ``"go"`` and ``"mojo"`` run the parity-verified compiled paths on request.

    Returns
    -------
    dict
        Keys ``trajectories``, ``C``, ``d``, ``R``, ``log_likelihoods``, ``tau``.
    """
    del seed  # initialisation is deterministic; retained only for API compatibility
    n_neurons = len(trains)
    if n_neurons == 0:
        return {
            "trajectories": np.array([]),
            "C": np.array([]),
            "d": np.array([]),
            "R": np.array([]),
            "log_likelihoods": [],
            "tau": np.array([]),
        }

    Y = _bin_trains(trains, bin_ms, dt)
    n_bins = Y.shape[1]
    n_latents = min(n_latents, n_neurons, n_bins)

    C0, d0, R0, tau = gpfa_pca_init(Y, n_latents, bin_ms)
    x_post, C, d, R, log_liks = _gpfa_em_dispatch(Y, C0, d0, R0, tau, max_iter, tol, backend)

    return {
        "trajectories": x_post,
        "C": C,
        "d": d,
        "R": R,
        "log_likelihoods": log_liks,
        "tau": tau,
    }


def gpfa_transform(
    new_trains: list[np.ndarray[Any, Any]],
    params: dict[str, Any],
    bin_ms: float = 20.0,
    dt: float = 0.001,
) -> np.ndarray[Any, Any]:
    """Project new spike trains using learned GPFA parameters."""
    C = params["C"]
    d = params["d"]
    R = params["R"]
    tau = params["tau"]

    n_neurons = len(new_trains)
    if n_neurons == 0 or C.size == 0:
        return np.array([])

    bin_steps = max(1, int(bin_ms / (dt * 1000)))
    binned = [bin_spike_train(t, bin_steps).astype(np.float64) for t in new_trains]
    min_bins = min(b.size for b in binned)
    Y = np.array([b[:min_bins] for b in binned])
    n_bins = Y.shape[1]
    n_latents = C.shape[1]

    K_all = [_gp_kernel(n_bins, tau[j]) for j in range(n_latents)]
    x_post, _ = _gpfa_e_step(Y, C, d, R, K_all)
    return x_post
