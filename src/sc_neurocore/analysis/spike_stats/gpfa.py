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

from typing import Any

import numpy as np

from .basic import bin_spike_train


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


def gpfa(
    trains: list[np.ndarray[Any, Any]],
    n_latents: int = 3,
    bin_ms: float = 20.0,
    dt: float = 0.001,
    max_iter: int = 50,
    tol: float = 1e-4,
    seed: int = 42,
) -> dict[str, Any]:
    """Extract smooth latent trajectories from parallel spike trains via EM."""
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

    bin_steps = max(1, int(bin_ms / (dt * 1000)))
    binned = [bin_spike_train(t, bin_steps).astype(np.float64) for t in trains]
    min_bins = min(b.size for b in binned)
    Y = np.array([b[:min_bins] for b in binned])  # (n_neurons, n_bins)
    n_bins = Y.shape[1]

    n_latents = min(n_latents, n_neurons, n_bins)
    rng = np.random.default_rng(seed)

    # Initialize
    C = rng.normal(0, 0.1, (n_neurons, n_latents))
    d = Y.mean(axis=1)
    R = np.diag(Y.var(axis=1) + 1e-4)
    tau = np.full(n_latents, bin_ms * 2.0)  # GP timescales in bin units

    log_liks = []
    for iteration in range(max_iter):
        K_all = [_gp_kernel(n_bins, tau[j]) for j in range(n_latents)]

        x_post, xx_post = _gpfa_e_step(Y, C, d, R, K_all)
        C, d, R = _gpfa_m_step(Y, x_post, xx_post)

        ll = _gpfa_log_likelihood(Y, C, d, R, K_all)
        log_liks.append(float(ll))

        if len(log_liks) > 1 and abs(log_liks[-1] - log_liks[-2]) < tol:
            break

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
