# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Granger causality and directed connectivity measures

"""Granger causality and directed connectivity measures for binned spike trains."""

from __future__ import annotations

from typing import Any

import numpy as np

from .basic import bin_spike_train


def _require_positive_int(name: str, value: int) -> int:
    """Return ``value`` after enforcing a positive integer public contract."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _as_finite_train(train: np.ndarray[Any, Any], *, name: str) -> np.ndarray[Any, Any]:
    """Coerce a spike train to a one-dimensional finite ``float64`` array."""
    try:
        values = np.asarray(train, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric spike values") from exc
    if values.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not bool(np.all(np.isfinite(values))):
        raise ValueError(f"{name} must contain only finite values")
    return values


def _binned_train(train: np.ndarray[Any, Any], bin_size: int, *, name: str) -> np.ndarray[Any, Any]:
    """Validate and bin a spike train into finite ``float64`` spike counts."""
    values = _as_finite_train(train, name=name)
    return bin_spike_train(values, bin_size).astype(np.float64)


def _binned_population(trains: list[np.ndarray[Any, Any]], bin_size: int) -> np.ndarray[Any, Any]:
    """Validate and stack a population of binned spike trains."""
    if not trains:
        raise ValueError("trains must contain at least one spike train")
    binned = [
        _binned_train(train, bin_size, name=f"trains[{idx}]") for idx, train in enumerate(trains)
    ]
    n_bins = binned[0].size
    if any(train.size != n_bins for train in binned):
        raise ValueError("all trains must have the same number of bins")
    return np.vstack(binned)


def _var_coefficients(
    trains_binned: np.ndarray[Any, Any], order: int
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Fit a regularised VAR model.

    Parameters
    ----------
    trains_binned:
        Population spike-count matrix with shape ``(n_neurons, n_bins)``.
    order:
        Positive autoregressive order.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Coefficient matrix with shape ``(order * n_neurons, n_neurons)`` and
        residual covariance with shape ``(n_neurons, n_neurons)``. Too-short
        histories return a zero-coefficient identity-covariance fallback.
    """
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
    source: np.ndarray[Any, Any], target: np.ndarray[Any, Any], bin_size: int = 10, order: int = 5
) -> float:
    """Return pairwise Granger causality from ``source`` to ``target``.

    Parameters
    ----------
    source:
        One-dimensional binary or count-valued source spike train.
    target:
        One-dimensional binary or count-valued target spike train.
    bin_size:
        Positive number of samples per spike-count bin.
    order:
        Positive autoregressive model order.

    Returns
    -------
    float
        Log-likelihood ratio. Positive values indicate that past source counts
        reduce target prediction error under the regularised Granger model.

    Raises
    ------
    ValueError
        If ``bin_size`` or ``order`` is not positive, or if either train is not
        one-dimensional and finite.
    """
    bin_size = _require_positive_int("bin_size", bin_size)
    order = _require_positive_int("order", order)
    cs = _binned_train(source, bin_size, name="source")
    ct = _binned_train(target, bin_size, name="target")
    n = min(cs.size, ct.size)
    if n <= 2 * order:
        return 0.0
    cs, ct = cs[:n], ct[:n]
    y = ct[order:]
    x_r = np.column_stack([ct[order - k - 1 : n - k - 1] for k in range(order)])
    x_f = np.column_stack([x_r] + [cs[order - k - 1 : n - k - 1] for k in range(order)])

    def _sse(x: np.ndarray[Any, Any], yy: np.ndarray[Any, Any]) -> float:
        xtx = x.T @ x
        reg = 1e-8 * np.eye(xtx.shape[0])
        beta = np.linalg.solve(xtx + reg, x.T @ yy)
        residuals = yy - x @ beta
        return float(np.sum(residuals**2))

    sse_r = _sse(x_r, y)
    sse_f = _sse(x_f, y)
    if sse_f <= 0:
        return 0.0
    return float(np.log(max(sse_r, 1e-30) / max(sse_f, 1e-30)))


def conditional_granger_causality(
    source: np.ndarray[Any, Any],
    target: np.ndarray[Any, Any],
    condition: np.ndarray[Any, Any],
    bin_size: int = 10,
    order: int = 5,
) -> float:
    """Return conditional Granger causality from ``source`` to ``target``.

    The reduced model predicts ``target`` from its own history and the
    ``condition`` history. The full model adds ``source`` history, following the
    Geweke conditional Granger construction.

    Parameters
    ----------
    source:
        One-dimensional binary or count-valued source spike train.
    target:
        One-dimensional binary or count-valued target spike train.
    condition:
        One-dimensional binary or count-valued conditioning spike train.
    bin_size:
        Positive number of samples per spike-count bin.
    order:
        Positive autoregressive model order.

    Returns
    -------
    float
        Log-likelihood ratio after controlling for ``condition``. Positive
        values indicate source-specific predictive information.

    Raises
    ------
    ValueError
        If ``bin_size`` or ``order`` is not positive, or if any train is not
        one-dimensional and finite.
    """
    bin_size = _require_positive_int("bin_size", bin_size)
    order = _require_positive_int("order", order)
    cs = _binned_train(source, bin_size, name="source")
    ct = _binned_train(target, bin_size, name="target")
    cc = _binned_train(condition, bin_size, name="condition")
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

    def _sse(x: np.ndarray[Any, Any], yy: np.ndarray[Any, Any]) -> float:
        reg = 1e-8 * np.eye(x.shape[1])
        beta = np.linalg.solve(x.T @ x + reg, x.T @ yy)
        return float(np.sum((yy - x @ beta) ** 2))

    sse_c = _sse(x_cond, y)
    sse_f = _sse(x_full, y)
    if sse_f <= 0:
        return 0.0
    return float(np.log(max(sse_c, 1e-30) / max(sse_f, 1e-30)))


def spectral_granger_causality(
    trains: list[np.ndarray[Any, Any]], bin_size: int = 10, order: int = 5, n_freqs: int = 64
) -> np.ndarray[Any, Any]:
    """Return frequency-domain Granger causality for a spike-train population.

    Parameters
    ----------
    trains:
        Non-empty list of one-dimensional spike trains. All trains must produce
        the same number of bins.
    bin_size:
        Positive number of samples per spike-count bin.
    order:
        Positive autoregressive model order.
    n_freqs:
        Positive number of frequencies in the closed interval ``[0, 0.5]``.

    Returns
    -------
    np.ndarray
        Array with shape ``(n_neurons, n_neurons, n_freqs)``. Singular transfer
        matrices are skipped and leave the corresponding frequency slice at
        zero rather than raising during the inverse.

    Raises
    ------
    ValueError
        If any domain parameter is not positive, if the population is empty, if
        any train is not one-dimensional and finite, or if binned lengths differ.
    """
    bin_size = _require_positive_int("bin_size", bin_size)
    order = _require_positive_int("order", order)
    n_freqs = _require_positive_int("n_freqs", n_freqs)
    binned = _binned_population(trains, bin_size)
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
    trains: list[np.ndarray[Any, Any]], bin_size: int = 10, order: int = 5, n_freqs: int = 64
) -> np.ndarray[Any, Any]:
    """Return partial directed coherence for a spike-train population.

    Parameters
    ----------
    trains:
        Non-empty list of one-dimensional spike trains. All trains must produce
        the same number of bins.
    bin_size:
        Positive number of samples per spike-count bin.
    order:
        Positive autoregressive model order.
    n_freqs:
        Positive number of frequencies in the closed interval ``[0, 0.5]``.

    Returns
    -------
    np.ndarray
        Normalised PDC tensor with shape ``(n_neurons, n_neurons, n_freqs)``.

    Raises
    ------
    ValueError
        If any domain parameter is not positive, if the population is empty, if
        any train is not one-dimensional and finite, or if binned lengths differ.
    """
    bin_size = _require_positive_int("bin_size", bin_size)
    order = _require_positive_int("order", order)
    n_freqs = _require_positive_int("n_freqs", n_freqs)
    binned = _binned_population(trains, bin_size)
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
    trains: list[np.ndarray[Any, Any]], bin_size: int = 10, order: int = 5, n_freqs: int = 64
) -> np.ndarray[Any, Any]:
    """Return the directed transfer function for a spike-train population.

    Parameters
    ----------
    trains:
        Non-empty list of one-dimensional spike trains. All trains must produce
        the same number of bins.
    bin_size:
        Positive number of samples per spike-count bin.
    order:
        Positive autoregressive model order.
    n_freqs:
        Positive number of frequencies in the closed interval ``[0, 0.5]``.

    Returns
    -------
    np.ndarray
        Normalised DTF tensor with shape ``(n_neurons, n_neurons, n_freqs)``.
        Singular transfer matrices are skipped and leave the corresponding
        frequency slice at zero.

    Raises
    ------
    ValueError
        If any domain parameter is not positive, if the population is empty, if
        any train is not one-dimensional and finite, or if binned lengths differ.
    """
    bin_size = _require_positive_int("bin_size", bin_size)
    order = _require_positive_int("order", order)
    n_freqs = _require_positive_int("n_freqs", n_freqs)
    binned = _binned_population(trains, bin_size)
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
