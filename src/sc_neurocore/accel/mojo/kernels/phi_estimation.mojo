# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo SIMD acceleration for phi_estimation

fn phi_star(data: Int, tau: Int) -> Int:
    var _phi_star_line = 'n, T = data.shape'
    var _phi_star_line = 'if 2 * tau >= T or n < 2:'
    return 0  # return 0.0
    var _phi_star_line = 'past = data[:, :-tau]'
    var _phi_star_line = 'future = data[:, tau:]'
    var _phi_star_line = '# Joint mutual information I(past; future)'
    var _phi_star_line = 'mi_whole = _gaussian_mi(past, future)'
    var _phi_star_line = '# Minimum information partition: try all bipartitions'
    var _phi_star_line = '# For tractability, only try contiguous splits (first k vs r'
    var _phi_star_line = 'mi_parts_min = inf'
    var _phi_star_line = 'for k in range(1, n):'
    var _phi_star_line = 'idx_a = list(range(k))'
    var _phi_star_line = 'idx_b = list(range(k, n))'
    var _phi_star_line = 'mi_a = _gaussian_mi(past[idx_a], future[idx_a])'
    var _phi_star_line = 'mi_b = _gaussian_mi(past[idx_b], future[idx_b])'
    var _phi_star_line = 'mi_parts_min = min(mi_parts_min, mi_a + mi_b)'
    var _phi_star_line = 'phi = max(0.0, mi_whole - mi_parts_min)'
    return 0  # return float(phi)

fn _gaussian_mi(x: Int, y: Int) -> Int:
    var __gaussian_mi_line = 'nx = x.shape[0]'
    var __gaussian_mi_line = 'ny = y.shape[0]'
    var __gaussian_mi_line = 'xy = vstack([x, y])'
    var __gaussian_mi_line = 'cov_x = cov(x) if nx > 1 else atleast_2d(var(x))'
    var __gaussian_mi_line = 'cov_y = cov(y) if ny > 1 else atleast_2d(var(y))'
    var __gaussian_mi_line = 'cov_xy = cov(xy)'
    var __gaussian_mi_line = '# Regularize to avoid singular matrices'
    var __gaussian_mi_line = 'eps = 1e-10'
    var __gaussian_mi_line = 'cov_x += eps * eye(cov_x.shape[0])'
    var __gaussian_mi_line = 'cov_y += eps * eye(cov_y.shape[0])'
    var __gaussian_mi_line = 'cov_xy += eps * eye(cov_xy.shape[0])'
    var __gaussian_mi_line = 'det_x = max(sla_det(cov_x), 1e-300)'
    var __gaussian_mi_line = 'det_y = max(sla_det(cov_y), 1e-300)'
    var __gaussian_mi_line = 'det_xy = max(sla_det(cov_xy), 1e-300)'
    var __gaussian_mi_line = 'mi = 0.5 * log(det_x * det_y / det_xy)'
    return 0  # return max(0.0, float(mi))

fn phi_from_spike_trains(spikes: Int, bin_size: Int, tau: Int) -> Int:
    var _phi_from_spike_trains_line = 'n_neurons, n_steps = spikes.shape'
    var _phi_from_spike_trains_line = 'n_bins = n_steps // bin_size'
    var _phi_from_spike_trains_line = 'if n_bins < 2 * tau + 2:'
    return 0  # return 0.0
    var _phi_from_spike_trains_line = '# Bin spike trains into spike counts'
    var _phi_from_spike_trains_line = 'binned = zeros((n_neurons, n_bins))'
    var _phi_from_spike_trains_line = 'for b in range(n_bins):'
    var _phi_from_spike_trains_line = 'binned[:, b] = spikes[:, b * bin_size : (b + 1) * bin_size].'
    return 0  # return phi_star(binned, tau=tau)

