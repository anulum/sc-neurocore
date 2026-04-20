# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo SIMD acceleration for causality

fn _var_coefficients(trains_binned: Int, order: Int) -> Int:
    var __var_coefficients_line = 'trains_binned: ndarray[Any, Any], order: int'
    var __var_coefficients_line = ') -> tuple[ndarray[Any, Any], ndarray[Any, Any]]:'
    var __var_coefficients_line = 'd, t = trains_binned.shape'
    var __var_coefficients_line = 'if t <= order + 1:'
    return 0  # return zeros((order * d, d)), eye(d)
    var __var_coefficients_line = 'y = trains_binned[:, order:].T  # (T-order) x d'
    var __var_coefficients_line = 'x_parts = [trains_binned[:, order - k - 1 : t - k - 1].T for'
    var __var_coefficients_line = 'x = hstack(x_parts)  # (T-order) x (order*d)'
    var __var_coefficients_line = 'reg = 1e-8 * eye(x.shape[1])'
    var __var_coefficients_line = 'beta = linalg.solve(x.T @ x + reg, x.T @ y)'
    var __var_coefficients_line = 'residuals = y - x @ beta'
    var __var_coefficients_line = 'cov = residuals.T @ residuals / max(residuals.shape[0], 1)'
    return 0  # return beta, cov

fn pairwise_granger_causality(source: Int, target: Int, bin_size: Int, order: Int) -> Int:
    var _pairwise_granger_causality_line = 'source: ndarray[Any, Any], target: ndarray[Any, Any], bin_si'
    var _pairwise_granger_causality_line = ') -> float:'
    var _pairwise_granger_causality_line = 'cs = bin_spike_train(source, bin_size).astype(float64)'
    var _pairwise_granger_causality_line = 'ct = bin_spike_train(target, bin_size).astype(float64)'
    var _pairwise_granger_causality_line = 'n = min(cs.size, ct.size)'
    var _pairwise_granger_causality_line = 'if n <= 2 * order:'
    return 0  # return 0.0
    var _pairwise_granger_causality_line = 'cs, ct = cs[:n], ct[:n]'
    var _pairwise_granger_causality_line = 'y = ct[order:]'
    var _pairwise_granger_causality_line = 'n_pts = y.size'
    var _pairwise_granger_causality_line = 'x_r = column_stack([ct[order - k - 1 : n - k - 1] for k in r'
    var _pairwise_granger_causality_line = 'x_f = column_stack([x_r] + [cs[order - k - 1 : n - k - 1] fo'
    var _pairwise_granger_causality_line = 'xtx = x.T @ x'
    var _pairwise_granger_causality_line = 'reg = 1e-8 * eye(xtx.shape[0])'
    var _pairwise_granger_causality_line = 'beta = linalg.solve(xtx + reg, x.T @ yy)'
    var _pairwise_granger_causality_line = 'residuals = yy - x @ beta'
    return 0  # return float(sum(residuals**2))
    var _pairwise_granger_causality_line = 'sse_r = _sse(x_r, y)'
    var _pairwise_granger_causality_line = 'sse_f = _sse(x_f, y)'
    var _pairwise_granger_causality_line = 'if sse_f <= 0:'
    return 0  # return 0.0
    return 0  # return float(log(max(sse_r, 1e-30) / max(sse_f, 1e

fn conditional_granger_causality(source: Int, target: Int, condition: Int, bin_size: Int, order: Int) -> Int:
    var _conditional_granger_causality_line = 'source: ndarray[Any, Any],'
    var _conditional_granger_causality_line = 'target: ndarray[Any, Any],'
    var _conditional_granger_causality_line = 'condition: ndarray[Any, Any],'
    var _conditional_granger_causality_line = 'bin_size: int = 10,'
    var _conditional_granger_causality_line = 'order: int = 5,'
    var _conditional_granger_causality_line = ') -> float:'
    var _conditional_granger_causality_line = 'cs = bin_spike_train(source, bin_size).astype(float64)'
    var _conditional_granger_causality_line = 'ct = bin_spike_train(target, bin_size).astype(float64)'
    var _conditional_granger_causality_line = 'cc = bin_spike_train(condition, bin_size).astype(float64)'
    var _conditional_granger_causality_line = 'n = min(cs.size, ct.size, cc.size)'
    var _conditional_granger_causality_line = 'if n <= 2 * order:'
    return 0  # return 0.0
    var _conditional_granger_causality_line = 'cs, ct, cc = cs[:n], ct[:n], cc[:n]'
    var _conditional_granger_causality_line = 'y = ct[order:]'
    var _conditional_granger_causality_line = 'x_cond = column_stack('
    var _conditional_granger_causality_line = '[ct[order - k - 1 : n - k - 1] for k in range(order)]'
    var _conditional_granger_causality_line = '+ [cc[order - k - 1 : n - k - 1] for k in range(order)]'
    var _conditional_granger_causality_line = ')'
    var _conditional_granger_causality_line = 'x_full = column_stack([x_cond] + [cs[order - k - 1 : n - k -'
    var _conditional_granger_causality_line = 'reg = 1e-8 * eye(x.shape[1])'
    var _conditional_granger_causality_line = 'beta = linalg.solve(x.T @ x + reg, x.T @ yy)'
    return 0  # return float(sum((yy - x @ beta) ** 2))
    var _conditional_granger_causality_line = 'sse_c = _sse(x_cond, y)'
    var _conditional_granger_causality_line = 'sse_f = _sse(x_full, y)'
    var _conditional_granger_causality_line = 'if sse_f <= 0:'
    return 0  # return 0.0
    return 0  # return float(log(max(sse_c, 1e-30) / max(sse_f, 1e

fn spectral_granger_causality(trains: Int, bin_size: Int, order: Int, n_freqs: Int) -> Int:
    var _spectral_granger_causality_line = 'trains: list[ndarray[Any, Any]], bin_size: int = 10, order: '
    var _spectral_granger_causality_line = ') -> ndarray[Any, Any]:'
    var _spectral_granger_causality_line = 'binned = array([bin_spike_train(t, bin_size).astype(float64)'
    var _spectral_granger_causality_line = 'd = binned.shape[0]'
    var _spectral_granger_causality_line = 'beta, sigma = _var_coefficients(binned, order)'
    var _spectral_granger_causality_line = 'freqs = linspace(0, 0.5, n_freqs)'
    var _spectral_granger_causality_line = 'gc = zeros((d, d, n_freqs))'
    var _spectral_granger_causality_line = 'for fi, f in enumerate(freqs):'
    var _spectral_granger_causality_line = 'a_f = eye(d, dtype=complex)'
    var _spectral_granger_causality_line = 'for k in range(order):'
    var _spectral_granger_causality_line = 'coeff_block = beta[k * d : (k + 1) * d, :].T'
    var _spectral_granger_causality_line = 'a_f -= coeff_block * exp(-2j * pi * f * (k + 1))'
    var _spectral_granger_causality_line = 'det_a = linalg.det(a_f)'
    var _spectral_granger_causality_line = 'if abs(det_a) < 1e-30:'
    var _spectral_granger_causality_line = 'continue'
    var _spectral_granger_causality_line = 'h = linalg.inv(a_f)'
    var _spectral_granger_causality_line = 's = h @ sigma @ h.conj().T'
    var _spectral_granger_causality_line = 'for i in range(d):'
    var _spectral_granger_causality_line = 'for j in range(d):'
    var _spectral_granger_causality_line = 'if i == j:'
    var _spectral_granger_causality_line = 'continue'
    var _spectral_granger_causality_line = 'if abs(s[i, i]) > 1e-30:'
    var _spectral_granger_causality_line = 'gc[i, j, fi] = max('
    var _spectral_granger_causality_line = '0.0,'
    var _spectral_granger_causality_line = 'log('
    var _spectral_granger_causality_line = 'abs(s[i, i]) / abs(s[i, i] - sigma[j, j] * abs(h[i, j]) ** 2'
    var _spectral_granger_causality_line = ').real,'
    var _spectral_granger_causality_line = ')'
    return 0  # return gc

fn partial_directed_coherence(trains: Int, bin_size: Int, order: Int, n_freqs: Int) -> Int:
    var _partial_directed_coherence_line = 'trains: list[ndarray[Any, Any]], bin_size: int = 10, order: '
    var _partial_directed_coherence_line = ') -> ndarray[Any, Any]:'
    var _partial_directed_coherence_line = 'binned = array([bin_spike_train(t, bin_size).astype(float64)'
    var _partial_directed_coherence_line = 'd = binned.shape[0]'
    var _partial_directed_coherence_line = 'beta, _ = _var_coefficients(binned, order)'
    var _partial_directed_coherence_line = 'freqs = linspace(0, 0.5, n_freqs)'
    var _partial_directed_coherence_line = 'pdc = zeros((d, d, n_freqs))'
    var _partial_directed_coherence_line = 'for fi, f in enumerate(freqs):'
    var _partial_directed_coherence_line = 'a_f = eye(d, dtype=complex)'
    var _partial_directed_coherence_line = 'for k in range(order):'
    var _partial_directed_coherence_line = 'coeff_block = beta[k * d : (k + 1) * d, :].T'
    var _partial_directed_coherence_line = 'a_f -= coeff_block * exp(-2j * pi * f * (k + 1))'
    var _partial_directed_coherence_line = 'for j in range(d):'
    var _partial_directed_coherence_line = 'norm = sqrt(sum(abs(a_f[:, j]) ** 2))'
    var _partial_directed_coherence_line = 'if norm > 0:'
    var _partial_directed_coherence_line = 'for i in range(d):'
    var _partial_directed_coherence_line = 'pdc[i, j, fi] = abs(a_f[i, j]) / norm'
    return 0  # return pdc

fn directed_transfer_function(trains: Int, bin_size: Int, order: Int, n_freqs: Int) -> Int:
    var _directed_transfer_function_line = 'trains: list[ndarray[Any, Any]], bin_size: int = 10, order: '
    var _directed_transfer_function_line = ') -> ndarray[Any, Any]:'
    var _directed_transfer_function_line = 'binned = array([bin_spike_train(t, bin_size).astype(float64)'
    var _directed_transfer_function_line = 'd = binned.shape[0]'
    var _directed_transfer_function_line = 'beta, sigma = _var_coefficients(binned, order)'
    var _directed_transfer_function_line = 'freqs = linspace(0, 0.5, n_freqs)'
    var _directed_transfer_function_line = 'dtf = zeros((d, d, n_freqs))'
    var _directed_transfer_function_line = 'for fi, f in enumerate(freqs):'
    var _directed_transfer_function_line = 'a_f = eye(d, dtype=complex)'
    var _directed_transfer_function_line = 'for k in range(order):'
    var _directed_transfer_function_line = 'coeff_block = beta[k * d : (k + 1) * d, :].T'
    var _directed_transfer_function_line = 'a_f -= coeff_block * exp(-2j * pi * f * (k + 1))'
    var _directed_transfer_function_line = 'det_a = linalg.det(a_f)'
    var _directed_transfer_function_line = 'if abs(det_a) < 1e-30:'
    var _directed_transfer_function_line = 'continue'
    var _directed_transfer_function_line = 'h = linalg.inv(a_f)'
    var _directed_transfer_function_line = 'for i in range(d):'
    var _directed_transfer_function_line = 'norm = sqrt(sum(abs(h[i, :]) ** 2))'
    var _directed_transfer_function_line = 'if norm > 0:'
    var _directed_transfer_function_line = 'for j in range(d):'
    var _directed_transfer_function_line = 'dtf[i, j, fi] = abs(h[i, j]) / norm'
    return 0  # return dtf

fn _sse(x: Int, yy: Int) -> Int:
    var __sse_line = 'xtx = x.T @ x'
    var __sse_line = 'reg = 1e-8 * eye(xtx.shape[0])'
    var __sse_line = 'beta = linalg.solve(xtx + reg, x.T @ yy)'
    var __sse_line = 'residuals = yy - x @ beta'
    return 0  # return float(sum(residuals**2))

fn _sse(x: Int, yy: Int) -> Int:
    var __sse_line = 'reg = 1e-8 * eye(x.shape[1])'
    var __sse_line = 'beta = linalg.solve(x.T @ x + reg, x.T @ yy)'
    return 0  # return float(sum((yy - x @ beta) ** 2))
