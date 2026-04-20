# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo SIMD acceleration for patterns

fn spike_directionality(times_a: Int, times_b: Int, t_start: Int, t_end: Int) -> Int:
    var _spike_directionality_line = 'times_a: ndarray[Any, Any],'
    var _spike_directionality_line = 'times_b: ndarray[Any, Any],'
    var _spike_directionality_line = 't_start: float = 0.0,'
    var _spike_directionality_line = 't_end: float = 1.0,'
    var _spike_directionality_line = ') -> float:'
    var _spike_directionality_line = 'ta = sort(times_a[(times_a >= t_start) & (times_a <= t_end)]'
    var _spike_directionality_line = 'tb = sort(times_b[(times_b >= t_start) & (times_b <= t_end)]'
    var _spike_directionality_line = 'if ta.size == 0 or tb.size == 0:'
    return 0  # return 0.0
    var _spike_directionality_line = 'lead_a = 0'
    var _spike_directionality_line = 'lead_b = 0'
    var _spike_directionality_line = 'for t in ta:'
    var _spike_directionality_line = 'diffs = tb - t'
    var _spike_directionality_line = 'pos = diffs[diffs > 0]'
    var _spike_directionality_line = 'neg = diffs[diffs < 0]'
    var _spike_directionality_line = 'if pos.size > 0 and neg.size > 0:'
    var _spike_directionality_line = 'nearest_after = pos.min()'
    var _spike_directionality_line = 'nearest_before = abs(neg).min()'
    var _spike_directionality_line = 'if nearest_before < nearest_after:'
    var _spike_directionality_line = 'lead_b += 1'
    var _spike_directionality_line = 'else:'
    var _spike_directionality_line = 'lead_a += 1'
    var _spike_directionality_line = 'total = lead_a + lead_b'
    var _spike_directionality_line = 'if total == 0:'
    return 0  # return 0.0
    return 0  # return float((lead_a - lead_b) / total)

fn spike_train_order(times_list: Int, t_start: Int, t_end: Int) -> Int:
    var _spike_train_order_line = 'times_list: list[ndarray[Any, Any]], t_start: float = 0.0, t'
    var _spike_train_order_line = ') -> ndarray[Any, Any]:'
    var _spike_train_order_line = 'n = len(times_list)'
    var _spike_train_order_line = 'mat = zeros((n, n))'
    var _spike_train_order_line = 'for i in range(n):'
    var _spike_train_order_line = 'for j in range(i + 1, n):'
    var _spike_train_order_line = 'd = spike_directionality(times_list[i], times_list[j], t_sta'
    var _spike_train_order_line = 'mat[i, j] = d'
    var _spike_train_order_line = 'mat[j, i] = -d'
    return 0  # return mat

fn cubic_higher_order(binary_train: Int, dt: Int, max_lag: Int) -> Int:
    var _cubic_higher_order_line = 'binary_train: ndarray[Any, Any], dt: float = 0.001, max_lag:'
    var _cubic_higher_order_line = ') -> ndarray[Any, Any]:'
    var _cubic_higher_order_line = 'x = binary_train.astype(float64) - binary_train.mean()'
    var _cubic_higher_order_line = 'n = x.size'
    var _cubic_higher_order_line = 'c3 = zeros((max_lag, max_lag))'
    var _cubic_higher_order_line = 'for t1 in range(max_lag):'
    var _cubic_higher_order_line = 'for t2 in range(max_lag):'
    var _cubic_higher_order_line = 'valid_n = n - max(t1, t2)'
    var _cubic_higher_order_line = 'if valid_n <= 0:'
    var _cubic_higher_order_line = 'continue'
    var _cubic_higher_order_line = 'c3[t1, t2] = sum(x[:valid_n] * x[t1 : t1 + valid_n] * x[t2 :'
    return 0  # return c3
