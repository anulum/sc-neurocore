# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

"""Granger causality and directed connectivity measures."""

from __future__ import annotations

import numpy as np

from .basic import bin_spike_train


def _var_coefficients(trains_binned: np.ndarray, order: int) -> tuple[np.ndarray, np.ndarray]:
    """Fit VAR(order) model. Returns (coefficients [order*d x d], residual covariance)."""
    d, t = trains_binned.shape
    if t <= order + 1:
        return np.zeros((order * d, d)), np.eye(d)
    y = trains_binned[:, order:].T  # (T-order) x d
    x_parts = [trains_binned[:, order - k - 1 : t - k - 1].T for k in range(order)]
    x = np.hstack(x_parts)  # (T-order) x (order*d)
    reg = 1e-8 * np.eye(x.shape[1])
    beta = np.linalg.solve(x.T @ x + reg, x.T @ y)
    residuals = y - x @ beta
    cov = residuals.T @ residuals / max(residuals.shape[0], 1)
    return beta, cov


def pairwise_granger_causality(
    source: np.ndarray, target: np.ndarray, bin_size: int = 10, order: int = 5
) -> float:
    """Pairwise Granger causality. Granger 1969.

    Tests if past source spike counts reduce prediction error for target.
    Returns log-likelihood ratio. Positive = source Granger-causes target.
    """
    cs = bin_spike_train(source, bin_size).astype(np.float64)
    ct = bin_spike_train(target, bin_size).astype(np.float64)
    n = min(cs.size, ct.size)
    if n <= 2 * order:
        return 0.0
    cs, ct = cs[:n], ct[:n]
    y = ct[order:]
    n_pts = y.size
    x_r = np.column_stack([ct[order - k - 1 : n - k - 1] for k in range(order)])
    x_f = np.column_stack([x_r] + [cs[order - k - 1 : n - k - 1] for k in range(order)])

    def _sse(x, yy):
        xtx = x.T @ x
        reg = 1e-8 * np.eye(xtx.shape[0])
        beta = np.linalg.solve(xtx + reg, x.T @ yy)
        residuals = yy - x @ beta
        return np.sum(residuals**2)

    sse_r = _sse(x_r, y)
    sse_f = _sse(x_f, y)
    if sse_f <= 0:
        return 0.0
    return float(np.log(max(sse_r, 1e-30) / max(sse_f, 1e-30)))


def conditional_granger_causality(
    source: np.ndarray,
    target: np.ndarray,
    condition: np.ndarray,
    bin_size: int = 10,
    order: int = 5,
) -> float:
    """Conditional Granger causality. Geweke 1984.

    Tests if source Granger-causes target controlling for condition.
    """
    cs = bin_spike_train(source, bin_size).astype(np.float64)
    ct = bin_spike_train(target, bin_size).astype(np.float64)
    cc = bin_spike_train(condition, bin_size).astype(np.float64)
    n = min(cs.size, ct.size, cc.size)
    if n <= 2 * order:
        return 0.0
    cs, ct, cc = cs[:n], ct[:n], cc[:n]
    y = ct[order:]
    x_cond = np.column_stack(
        [ct[order - k - 1 : n - k - 1] for k in range(order)]
        + [cc[order - k - 1 : n - k - 1] for k in range(order)]
    )
    x_full = np.column_stack([x_cond] + [cs[order - k - 1 : n - k - 1] for k in range(order)])

    def _sse(x, yy):
        reg = 1e-8 * np.eye(x.shape[1])
        beta = np.linalg.solve(x.T @ x + reg, x.T @ yy)
        return np.sum((yy - x @ beta) ** 2)

    sse_c = _sse(x_cond, y)
    sse_f = _sse(x_full, y)
    if sse_f <= 0:
        return 0.0
    return float(np.log(max(sse_c, 1e-30) / max(sse_f, 1e-30)))


def spectral_granger_causality(
    trains: list[np.ndarray], bin_size: int = 10, order: int = 5, n_freqs: int = 64
) -> np.ndarray:
    """Spectral Granger causality. Geweke 1982.

    Returns (n_neurons x n_neurons x n_freqs) array of frequency-domain GC values.
    """
    binned = np.array([bin_spike_train(t, bin_size).astype(np.float64) for t in trains])
    d = binned.shape[0]
    beta, sigma = _var_coefficients(binned, order)
    freqs = np.linspace(0, 0.5, n_freqs)
    gc = np.zeros((d, d, n_freqs))
    for fi, f in enumerate(freqs):
        a_f = np.eye(d, dtype=complex)
        for k in range(order):
            coeff_block = beta[k * d : (k + 1) * d, :].T
            a_f -= coeff_block * np.exp(-2j * np.pi * f * (k + 1))
        det_a = np.linalg.det(a_f)
        if abs(det_a) < 1e-30:
            continue
        h = np.linalg.inv(a_f)
        s = h @ sigma @ h.conj().T
        for i in range(d):
            for j in range(d):
                if i == j:
                    continue
                if abs(s[i, i]) > 1e-30:
                    gc[i, j, fi] = max(
                        0.0,
                        np.log(
                            abs(s[i, i]) / abs(s[i, i] - sigma[j, j] * abs(h[i, j]) ** 2 + 1e-30)
                        ).real,
                    )
    return gc


def partial_directed_coherence(
    trains: list[np.ndarray], bin_size: int = 10, order: int = 5, n_freqs: int = 64
) -> np.ndarray:
    """Partial directed coherence (PDC). Baccala & Sameshima 2001.

    Returns (n_neurons x n_neurons x n_freqs) normalized PDC values.
    """
    binned = np.array([bin_spike_train(t, bin_size).astype(np.float64) for t in trains])
    d = binned.shape[0]
    beta, _ = _var_coefficients(binned, order)
    freqs = np.linspace(0, 0.5, n_freqs)
    pdc = np.zeros((d, d, n_freqs))
    for fi, f in enumerate(freqs):
        a_f = np.eye(d, dtype=complex)
        for k in range(order):
            coeff_block = beta[k * d : (k + 1) * d, :].T
            a_f -= coeff_block * np.exp(-2j * np.pi * f * (k + 1))
        for j in range(d):
            norm = np.sqrt(np.sum(np.abs(a_f[:, j]) ** 2))
            if norm > 0:
                for i in range(d):
                    pdc[i, j, fi] = np.abs(a_f[i, j]) / norm
    return pdc


def directed_transfer_function(
    trains: list[np.ndarray], bin_size: int = 10, order: int = 5, n_freqs: int = 64
) -> np.ndarray:
    """Directed transfer function (DTF). Kaminski & Blinowska 1991.

    Returns (n_neurons x n_neurons x n_freqs) normalized DTF values.
    """
    binned = np.array([bin_spike_train(t, bin_size).astype(np.float64) for t in trains])
    d = binned.shape[0]
    beta, sigma = _var_coefficients(binned, order)
    freqs = np.linspace(0, 0.5, n_freqs)
    dtf = np.zeros((d, d, n_freqs))
    for fi, f in enumerate(freqs):
        a_f = np.eye(d, dtype=complex)
        for k in range(order):
            coeff_block = beta[k * d : (k + 1) * d, :].T
            a_f -= coeff_block * np.exp(-2j * np.pi * f * (k + 1))
        det_a = np.linalg.det(a_f)
        if abs(det_a) < 1e-30:
            continue
        h = np.linalg.inv(a_f)
        for i in range(d):
            norm = np.sqrt(np.sum(np.abs(h[i, :]) ** 2))
            if norm > 0:
                for j in range(d):
                    dtf[i, j, fi] = np.abs(h[i, j]) / norm
    return dtf
