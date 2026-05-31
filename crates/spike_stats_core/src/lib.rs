// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Spike Stats Core (Rust)
// Authored by Anulum Fortis & Arcane Sapience (protoscience@anulum.li)

//! High-performance spike train distance and correlation metrics.
//!
//! Provides Rust implementations of the computationally intensive
//! spike train analysis functions from `sc_neurocore.analysis.spike_stats`.
//!
//! ## References
//! - Van Rossum (2001) — exponential kernel distance
//! - Victor & Purpura (1996) — edit distance
//! - Kreuz et al. (2007, 2013, 2015) — ISI, SPIKE, SPIKE-sync
//! - Schreiber et al. (2003) — Gaussian-smoothed correlation
//! - Hunter & Milton (2003) — coincidence similarity

// ---------------------------------------------------------------------------
// Distance metrics
// ---------------------------------------------------------------------------

/// Van Rossum (2001) exponential-kernel spike train distance.
///
/// Convolves each binary spike train with a decaying exponential kernel
/// and computes the L2 distance between the filtered signals.
pub fn van_rossum_distance(train_a: &[f64], train_b: &[f64], dt: f64, tau_ms: f64) -> f64 {
    let tau = tau_ms / 1000.0;
    let n = train_a.len().min(train_b.len());
    if n == 0 || tau <= 0.0 {
        return 0.0;
    }

    let mut decay = vec![0.0f64; n];
    for i in 0..n {
        decay[i] = (-(i as f64) * dt / tau).exp();
    }

    let mut fa = vec![0.0f64; n];
    let mut fb = vec![0.0f64; n];

    for i in 0..n {
        let mut sum_a = 0.0;
        let mut sum_b = 0.0;
        let limit = i + 1;
        for k in 0..limit {
            sum_a += train_a[k] * decay[i - k];
            sum_b += train_b[k] * decay[i - k];
        }
        fa[i] = sum_a;
        fb[i] = sum_b;
    }

    let mut acc = 0.0;
    for i in 0..n {
        let d = fa[i] - fb[i];
        acc += d * d;
    }
    (acc * dt / tau).sqrt()
}

/// Victor-Purpura (1996) edit distance between spike time arrays.
///
/// O(na * nb) dynamic programming. `cost_per_s` is the q parameter
/// (cost of shifting one spike by 1 second).
pub fn victor_purpura_distance(times_a: &[f64], times_b: &[f64], cost_per_s: f64) -> f64 {
    let na = times_a.len();
    let nb = times_b.len();
    if na == 0 {
        return nb as f64;
    }
    if nb == 0 {
        return na as f64;
    }

    let cols = nb + 1;
    let mut d = vec![0.0f64; (na + 1) * cols];

    for i in 0..=na {
        d[i * cols] = i as f64;
    }
    for j in 0..=nb {
        d[j] = j as f64;
    }

    for i in 1..=na {
        for j in 1..=nb {
            let shift_cost = cost_per_s * (times_a[i - 1] - times_b[j - 1]).abs();
            let del = d[(i - 1) * cols + j] + 1.0;
            let ins = d[i * cols + (j - 1)] + 1.0;
            let sub = d[(i - 1) * cols + (j - 1)] + shift_cost;
            d[i * cols + j] = del.min(ins).min(sub);
        }
    }

    d[na * cols + nb]
}

/// All-pairs Victor-Purpura distance matrix for N spike trains.
///
/// Returns flat N×N row-major matrix.
pub fn multi_neuron_victor_purpura(
    spike_times_list: &[Vec<f64>],
    cost_per_s: f64,
) -> Vec<f64> {
    let n = spike_times_list.len();
    let mut mat = vec![0.0f64; n * n];
    for i in 0..n {
        for j in (i + 1)..n {
            let d = victor_purpura_distance(
                &spike_times_list[i],
                &spike_times_list[j],
                cost_per_s,
            );
            mat[i * n + j] = d;
            mat[j * n + i] = d;
        }
    }
    mat
}

/// ISI-distance (Kreuz et al. 2007) from binary spike trains.
///
/// Computes element-wise inter-spike-interval ratios.
pub fn isi_distance(train_a: &[f64], train_b: &[f64], dt: f64) -> f64 {
    let n = train_a.len().min(train_b.len());
    if n < 2 {
        return f64::NAN;
    }

    let isi_a = compute_isi(train_a, dt);
    let isi_b = compute_isi(train_b, dt);
    let m = isi_a.len().min(isi_b.len());
    if m == 0 {
        return f64::NAN;
    }

    let mut sum = 0.0;
    for i in 0..m {
        let a = isi_a[i];
        let b = isi_b[i];
        let ratio = if a == 0.0 && b == 0.0 {
            0.0
        } else if a <= b {
            if b > 0.0 { a / b - 1.0 } else { 0.0 }
        } else if a > 0.0 {
            -(b / a - 1.0)
        } else {
            0.0
        };
        sum += ratio.abs();
    }
    sum / m as f64
}

fn compute_isi(train: &[f64], dt: f64) -> Vec<f64> {
    let mut intervals = Vec::new();
    let mut last_spike: Option<usize> = None;
    for (i, &v) in train.iter().enumerate() {
        if v > 0.5 {
            if let Some(prev) = last_spike {
                intervals.push((i - prev) as f64 * dt);
            }
            last_spike = Some(i);
        }
    }
    intervals
}

/// SPIKE-distance (Kreuz et al. 2013) between spike time arrays.
pub fn spike_distance(
    times_a: &[f64],
    times_b: &[f64],
    t_start: f64,
    t_end: f64,
) -> f64 {
    let mut ta: Vec<f64> = times_a
        .iter()
        .copied()
        .filter(|&t| t >= t_start && t <= t_end)
        .collect();
    let mut tb: Vec<f64> = times_b
        .iter()
        .copied()
        .filter(|&t| t >= t_start && t <= t_end)
        .collect();
    ta.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    tb.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

    if ta.is_empty() && tb.is_empty() {
        return 0.0;
    }
    if ta.is_empty() || tb.is_empty() {
        return 1.0;
    }

    let n_eval = 100;
    let step = (t_end - t_start) / (n_eval - 1) as f64;
    let mut sum = 0.0;

    for k in 0..n_eval {
        let t = t_start + k as f64 * step;
        let idx_a = ta.partition_point(|&x| x < t);
        let idx_b = tb.partition_point(|&x| x < t);

        let prev_a = if idx_a > 0 { ta[idx_a - 1] } else { t_start };
        let next_a = if idx_a < ta.len() { ta[idx_a] } else { t_end };
        let prev_b = if idx_b > 0 { tb[idx_b - 1] } else { t_start };
        let next_b = if idx_b < tb.len() { tb[idx_b] } else { t_end };

        let isi_a = (next_a - prev_a).max(1e-30);
        let isi_b = (next_b - prev_b).max(1e-30);

        let da = (t - prev_a).abs().min((t - next_a).abs());
        let db = (t - prev_b).abs().min((t - next_b).abs());

        sum += (da / isi_a - db / isi_b).abs();
    }
    sum / n_eval as f64
}

fn local_isi(times: &[f64], idx: usize) -> f64 {
    if times.len() < 2 {
        return 1.0;
    }
    if idx == 0 {
        return times[1] - times[0];
    }
    if idx >= times.len() - 1 {
        return times[times.len() - 1] - times[times.len() - 2];
    }
    (times[idx] - times[idx - 1]).min(times[idx + 1] - times[idx])
}

/// SPIKE-synchronization (Kreuz et al. 2015).
pub fn spike_sync(
    times_a: &[f64],
    times_b: &[f64],
    t_start: f64,
    t_end: f64,
) -> f64 {
    let mut ta: Vec<f64> = times_a
        .iter()
        .copied()
        .filter(|&t| t >= t_start && t <= t_end)
        .collect();
    let mut tb: Vec<f64> = times_b
        .iter()
        .copied()
        .filter(|&t| t >= t_start && t <= t_end)
        .collect();
    ta.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    tb.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

    if ta.is_empty() || tb.is_empty() {
        return 0.0;
    }

    let total_possible = ta.len() + tb.len();
    let mut coincidences = 0usize;

    for i in 0..ta.len() {
        let mut min_diff = f64::MAX;
        let mut best_j = 0;
        for (j, &t) in tb.iter().enumerate() {
            let d = (t - ta[i]).abs();
            if d < min_diff {
                min_diff = d;
                best_j = j;
            }
        }
        let isi_a = local_isi(&ta, i);
        let isi_b = local_isi(&tb, best_j);
        let tau = isi_a.min(isi_b) / 2.0;
        if tau > 0.0 && min_diff < tau {
            coincidences += 1;
        }
    }

    for j in 0..tb.len() {
        let mut min_diff = f64::MAX;
        let mut best_i = 0;
        for (i, &t) in ta.iter().enumerate() {
            let d = (t - tb[j]).abs();
            if d < min_diff {
                min_diff = d;
                best_i = i;
            }
        }
        let isi_a = local_isi(&ta, best_i);
        let isi_b = local_isi(&tb, j);
        let tau = isi_a.min(isi_b) / 2.0;
        if tau > 0.0 && min_diff < tau {
            coincidences += 1;
        }
    }

    if total_possible == 0 {
        0.0
    } else {
        coincidences as f64 / total_possible as f64
    }
}

/// Hunter-Milton (2003) coincidence similarity.
pub fn hunter_milton_similarity(
    times_a: &[f64],
    times_b: &[f64],
    dt_max: f64,
) -> f64 {
    if times_a.is_empty() || times_b.is_empty() {
        return 0.0;
    }
    let total = times_a.len() + times_b.len();
    let mut count = 0usize;

    for &t in times_a {
        let mut min_d = f64::MAX;
        for &s in times_b {
            let d = (t - s).abs();
            if d < min_d {
                min_d = d;
            }
        }
        if min_d < dt_max {
            count += 1;
        }
    }
    for &t in times_b {
        let mut min_d = f64::MAX;
        for &s in times_a {
            let d = (t - s).abs();
            if d < min_d {
                min_d = d;
            }
        }
        if min_d < dt_max {
            count += 1;
        }
    }

    count as f64 / total as f64
}

/// Earth mover's distance between spike time distributions.
pub fn earth_movers_distance(
    times_a: &[f64],
    times_b: &[f64],
    t_start: f64,
    t_end: f64,
    n_bins: usize,
) -> f64 {
    if n_bins == 0 {
        return 0.0;
    }
    let bin_width = (t_end - t_start) / n_bins as f64;

    let mut ha = vec![0.0f64; n_bins];
    let mut hb = vec![0.0f64; n_bins];

    for &t in times_a {
        let idx = ((t - t_start) / bin_width) as usize;
        if idx < n_bins {
            ha[idx] += 1.0;
        }
    }
    for &t in times_b {
        let idx = ((t - t_start) / bin_width) as usize;
        if idx < n_bins {
            hb[idx] += 1.0;
        }
    }

    let sa: f64 = ha.iter().sum();
    let sb: f64 = hb.iter().sum();
    if sa > 0.0 {
        for x in ha.iter_mut() {
            *x /= sa;
        }
    }
    if sb > 0.0 {
        for x in hb.iter_mut() {
            *x /= sb;
        }
    }

    let mut cum_a = 0.0;
    let mut cum_b = 0.0;
    let mut sum = 0.0;
    for i in 0..n_bins {
        cum_a += ha[i];
        cum_b += hb[i];
        sum += (cum_a - cum_b).abs();
    }
    sum * bin_width
}

// ---------------------------------------------------------------------------
// Correlation metrics
// ---------------------------------------------------------------------------

/// Cross-correlation histogram between two spike time arrays.
pub fn cross_correlation(
    times_a: &[f64],
    times_b: &[f64],
    bin_size: f64,
    max_lag: f64,
) -> Vec<f64> {
    let n_bins = (2.0 * max_lag / bin_size).ceil() as usize + 1;
    let mut hist = vec![0.0f64; n_bins];

    for &ta in times_a {
        for &tb in times_b {
            let lag = ta - tb;
            if lag.abs() <= max_lag {
                let idx = ((lag + max_lag) / bin_size) as usize;
                if idx < n_bins {
                    hist[idx] += 1.0;
                }
            }
        }
    }
    hist
}

/// Event synchronization (Quiroga et al. 2002).
pub fn event_synchronization(
    times_a: &[f64],
    times_b: &[f64],
    tau: f64,
) -> f64 {
    if times_a.is_empty() || times_b.is_empty() {
        return 0.0;
    }
    let mut count = 0usize;
    for &ta in times_a {
        for &tb in times_b {
            if (ta - tb).abs() < tau {
                count += 1;
            }
        }
    }
    let norm = ((times_a.len() * times_b.len()) as f64).sqrt();
    if norm == 0.0 {
        0.0
    } else {
        count as f64 / norm
    }
}

/// Spike-Time Tiling Coefficient (Cutts & Eglen, 2014).
pub fn spike_time_tiling_coefficient(
    times_a: &[f64],
    times_b: &[f64],
    dt: f64,
    t_start: f64,
    t_end: f64,
) -> f64 {
    let duration = t_end - t_start;
    if duration <= 0.0 || times_a.is_empty() || times_b.is_empty() {
        return 0.0;
    }

    let ta = 2.0 * dt * times_a.len() as f64 / duration;
    let tb = 2.0 * dt * times_b.len() as f64 / duration;

    let ta = ta.min(1.0);
    let tb = tb.min(1.0);

    let mut p = 0usize;

    for &a in times_a {
        for &b in times_b {
            if (a - b).abs() <= dt {
                p += 1;
                break;
            }
        }
    }

    let pa = p as f64 / times_a.len() as f64;
    let denom = 1.0 - ta * tb;
    if denom.abs() < 1e-30 {
        0.0
    } else {
        0.5 * (pa - tb) / denom + 0.5
    }
}

/// Coincidence index: fraction of spikes in A that have a nearest
/// neighbour in B within `window`.
pub fn coincidence_index(
    times_a: &[f64],
    times_b: &[f64],
    window: f64,
) -> f64 {
    if times_a.is_empty() {
        return 0.0;
    }
    let mut count = 0usize;
    for &a in times_a {
        for &b in times_b {
            if (a - b).abs() <= window {
                count += 1;
                break;
            }
        }
    }
    count as f64 / times_a.len() as f64
}

// ---------------------------------------------------------------------------
// Variability metrics
// ---------------------------------------------------------------------------

/// Lempel-Ziv 1976 complexity. Normalized by N/log2(N).
pub fn lempel_ziv_complexity(binary_train: &[u8]) -> f64 {
    let n = binary_train.len();
    if n == 0 {
        return 0.0;
    }
    let s: Vec<u8> = binary_train.iter().map(|&x| if x > 0 { 1 } else { 0 }).collect();
    let mut complexity: usize = 1;
    let mut l: usize = 1;
    let mut k: usize = 1;
    let mut k_max: usize = 1;

    while l + k <= n {
        if s[l + k - 1] == s[k - 1] {
            k += 1;
        } else {
            k_max = k_max.max(k);
            k = 1;
            if k_max > k {
                k_max = k;
            }
            complexity += 1;
            l += k_max;
            k = 1;
            k_max = 1;
        }
    }
    complexity += 1;
    let norm = n as f64 / (n as f64).max(2.0).log2();
    complexity as f64 / norm
}

/// Approximate entropy (ApEn). Pincus 1991.
/// O(N²) template matching.
pub fn approximate_entropy(data: &[f64], m: usize, r: f64) -> f64 {
    let n = data.len();
    if n < m + 2 {
        return f64::NAN;
    }

    fn phi(data: &[f64], dim: usize, r: f64) -> f64 {
        let n = data.len();
        if n < dim {
            return 0.0;
        }
        let n_templates = n - dim + 1;
        let mut log_sum = 0.0f64;
        for i in 0..n_templates {
            let mut count = 0usize;
            for j in 0..n_templates {
                let mut max_diff = 0.0f64;
                for k in 0..dim {
                    let d = (data[i + k] - data[j + k]).abs();
                    if d > max_diff {
                        max_diff = d;
                    }
                }
                if max_diff <= r {
                    count += 1;
                }
            }
            log_sum += (count as f64 / n_templates as f64 + 1e-30).ln();
        }
        log_sum / n_templates as f64
    }

    phi(data, m, r) - phi(data, m + 1, r)
}

/// Sample entropy (SampEn). Richman & Moorman 2000.
/// O(N²) template matching, no self-matches.
pub fn sample_entropy(data: &[f64], m: usize, r: f64) -> f64 {
    let n = data.len();
    if n < m + 2 {
        return f64::NAN;
    }

    fn count_matches(data: &[f64], dim: usize, r: f64) -> usize {
        let n = data.len();
        let n_templates = n - dim;
        let mut total = 0usize;
        for i in 0..n_templates {
            for j in (i + 1)..n_templates {
                let mut max_diff = 0.0f64;
                for k in 0..dim {
                    let d = (data[i + k] - data[j + k]).abs();
                    if d > max_diff {
                        max_diff = d;
                    }
                }
                if max_diff <= r {
                    total += 1;
                }
            }
        }
        total
    }

    let a = count_matches(data, m + 1, r);
    let b = count_matches(data, m, r);
    if b == 0 {
        return f64::NAN;
    }
    -((a as f64 + 1e-30) / (b as f64 + 1e-30)).ln()
}

/// Permutation entropy (Bandt & Pompe 2002). Normalized to [0, 1].
pub fn permutation_entropy(data: &[f64], order: usize, delay: usize) -> f64 {
    let n = data.len();
    if n < order * delay {
        return f64::NAN;
    }
    let n_patterns = n - (order - 1) * delay;
    if n_patterns < 1 {
        return f64::NAN;
    }

    // Encode each ordinal pattern as a unique integer
    use std::collections::HashMap;
    let mut pattern_counts: HashMap<Vec<usize>, usize> = HashMap::new();

    for i in 0..n_patterns {
        let mut window: Vec<(f64, usize)> = (0..order)
            .map(|j| (data[i + j * delay], j))
            .collect();
        window.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let rank: Vec<usize> = {
            let mut r = vec![0usize; order];
            for (pos, &(_, orig_idx)) in window.iter().enumerate() {
                r[orig_idx] = pos;
            }
            r
        };
        *pattern_counts.entry(rank).or_insert(0) += 1;
    }

    let total = n_patterns as f64;
    let h: f64 = pattern_counts
        .values()
        .map(|&c| {
            let p = c as f64 / total;
            -p * p.log2()
        })
        .sum();

    // factorial(order)
    let h_max = (1..=order).fold(1usize, |acc, x| acc * x) as f64;
    let h_max = h_max.log2();
    if h_max > 0.0 { h / h_max } else { 0.0 }
}

// ---------------------------------------------------------------------------
// Information-theoretic metrics
// ---------------------------------------------------------------------------

/// Kozachenko-Leonenko k-NN mutual information estimator (Kraskov et al. 2004).
///
/// O(N² × k) due to kth-neighbour distance search. Returns MI in nats.
pub fn kozachenko_leonenko_mi(x: &[f64], y: &[f64], k: usize) -> f64 {
    let n = x.len().min(y.len());
    if n < k + 1 {
        return 0.0;
    }

    fn digamma(mut z: f64) -> f64 {
        if z < 1e-6 {
            return -0.5772156649 - 1.0 / z;
        }
        let mut result = 0.0;
        while z < 6.0 {
            result -= 1.0 / z;
            z += 1.0;
        }
        result += z.ln() - 0.5 / z;
        let z2 = 1.0 / (z * z);
        result -= z2 * (1.0 / 12.0 - z2 * (1.0 / 120.0 - z2 / 252.0));
        result
    }

    let psi_k = digamma(k as f64);
    let psi_n = digamma(n as f64);

    let mut nx_sum = 0.0;
    let mut ny_sum = 0.0;

    for i in 0..n {
        let mut dists: Vec<f64> = (0..n)
            .filter(|&j| j != i)
            .map(|j| {
                let dx = (x[i] - x[j]).abs();
                let dy = (y[i] - y[j]).abs();
                dx.max(dy)
            })
            .collect();

        dists.select_nth_unstable_by(k - 1, |a, b| a.partial_cmp(b).unwrap());
        let eps = dists[k - 1];

        let mut nx = 0usize;
        let mut ny = 0usize;
        for j in 0..n {
            if j == i {
                continue;
            }
            if (x[i] - x[j]).abs() < eps {
                nx += 1;
            }
            if (y[i] - y[j]).abs() < eps {
                ny += 1;
            }
        }
        nx_sum += digamma((nx + 1) as f64);
        ny_sum += digamma((ny + 1) as f64);
    }

    let mi = psi_k + psi_n - nx_sum / n as f64 - ny_sum / n as f64;
    mi.max(0.0)
}

/// Spike train entropy via binary word analysis (Strong et al. 1998).
///
/// Computes Shannon entropy of binary word distribution.
pub fn spike_train_entropy(binned: &[u8], word_length: usize) -> f64 {
    let n = binned.len();
    if n < word_length {
        return f64::NAN;
    }
    let n_words = n - word_length + 1;
    use std::collections::HashMap;
    let mut counts: HashMap<u64, usize> = HashMap::new();

    for i in 0..n_words {
        let mut w: u64 = 0;
        for j in 0..word_length {
            w = w * 2 + if binned[i + j] > 0 { 1 } else { 0 };
        }
        *counts.entry(w).or_insert(0) += 1;
    }

    let total = n_words as f64;
    let mut h = 0.0f64;
    for &c in counts.values() {
        let p = c as f64 / total;
        h -= p * p.log2();
    }
    h
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-6;

    // ── Victor-Purpura ──────────────────────────────────────────────

    #[test]
    fn vp_identical() {
        let times = vec![0.1, 0.3, 0.5, 0.7, 0.9];
        let d = victor_purpura_distance(&times, &times, 1000.0);
        assert!(d < EPS, "identical trains should have VP=0, got {d}");
    }

    #[test]
    fn vp_empty_a() {
        let d = victor_purpura_distance(&[], &[0.1, 0.2, 0.3], 1000.0);
        assert!((d - 3.0).abs() < EPS);
    }

    #[test]
    fn vp_empty_b() {
        let d = victor_purpura_distance(&[0.1, 0.2], &[], 1000.0);
        assert!((d - 2.0).abs() < EPS);
    }

    #[test]
    fn vp_both_empty() {
        let d = victor_purpura_distance(&[], &[], 1000.0);
        assert!(d < EPS);
    }

    #[test]
    fn vp_symmetric() {
        let a = vec![0.1, 0.3, 0.5];
        let b = vec![0.15, 0.35, 0.55, 0.75];
        let d1 = victor_purpura_distance(&a, &b, 500.0);
        let d2 = victor_purpura_distance(&b, &a, 500.0);
        assert!((d1 - d2).abs() < EPS, "VP should be symmetric: {d1} vs {d2}");
    }

    #[test]
    fn vp_single_spikes_close() {
        let d = victor_purpura_distance(&[0.1], &[0.1001], 1000.0);
        assert!(d < 1.0, "close spikes should cost less than insert+delete, got {d}");
    }

    // ── Spike distance ──────────────────────────────────────────────

    #[test]
    fn spike_dist_identical() {
        let times = vec![0.2, 0.5, 0.8];
        let d = spike_distance(&times, &times, 0.0, 1.0);
        assert!(d < 0.1, "identical trains: SPIKE-dist should be ~0, got {d}");
    }

    #[test]
    fn spike_dist_empty_both() {
        let d = spike_distance(&[], &[], 0.0, 1.0);
        assert!(d < EPS);
    }

    #[test]
    fn spike_dist_one_empty() {
        let d = spike_distance(&[0.5], &[], 0.0, 1.0);
        assert!((d - 1.0).abs() < EPS);
    }

    // ── Spike sync ──────────────────────────────────────────────────

    #[test]
    fn spike_sync_identical() {
        let times = vec![0.2, 0.4, 0.6, 0.8];
        let s = spike_sync(&times, &times, 0.0, 1.0);
        assert!((s - 1.0).abs() < 0.01, "identical trains sync should be ~1.0, got {s}");
    }

    #[test]
    fn spike_sync_empty() {
        let s = spike_sync(&[], &[0.5], 0.0, 1.0);
        assert!(s < EPS);
    }

    // ── Hunter-Milton ───────────────────────────────────────────────

    #[test]
    fn hunter_milton_identical() {
        let times = vec![0.1, 0.3, 0.5, 0.7];
        let s = hunter_milton_similarity(&times, &times, 0.01);
        assert!((s - 1.0).abs() < EPS, "identical: HM should be 1.0, got {s}");
    }

    #[test]
    fn hunter_milton_empty() {
        let s = hunter_milton_similarity(&[], &[0.5], 0.01);
        assert!(s < EPS);
    }

    // ── Earth mover's ───────────────────────────────────────────────

    #[test]
    fn emd_identical() {
        let times = vec![0.1, 0.3, 0.5, 0.7];
        let d = earth_movers_distance(&times, &times, 0.0, 1.0, 100);
        assert!(d < EPS, "identical distributions: EMD should be 0, got {d}");
    }

    // ── Cross-correlation ───────────────────────────────────────────

    #[test]
    fn xcorr_self() {
        let times = vec![0.1, 0.3, 0.5, 0.7, 0.9];
        let hist = cross_correlation(&times, &times, 0.01, 0.05);
        let peak = hist.iter().cloned().fold(0.0f64, f64::max);
        assert!(peak >= 5.0, "self-correlation should have peak=N, got {peak}");
    }

    // ── Event synchronization ───────────────────────────────────────

    #[test]
    fn event_sync_identical() {
        let times = vec![0.1, 0.3, 0.5];
        let s = event_synchronization(&times, &times, 0.01);
        assert!(s > 0.0, "identical: event sync should be > 0");
    }

    // ── STTC ────────────────────────────────────────────────────────

    #[test]
    fn sttc_identical() {
        let times = vec![0.1, 0.3, 0.5, 0.7, 0.9];
        let s = spike_time_tiling_coefficient(&times, &times, 0.01, 0.0, 1.0);
        assert!(s > 0.5, "identical: STTC should be high, got {s}");
    }

    // ── Coincidence index ───────────────────────────────────────────

    #[test]
    fn coincidence_identical() {
        let times = vec![0.1, 0.3, 0.5];
        let c = coincidence_index(&times, &times, 0.01);
        assert!((c - 1.0).abs() < EPS, "identical: CI should be 1.0, got {c}");
    }

    // ── Multi-neuron VP ─────────────────────────────────────────────

    #[test]
    fn multi_vp_diagonal_zero() {
        let trains = vec![
            vec![0.1, 0.3, 0.5],
            vec![0.15, 0.35, 0.55],
            vec![0.2, 0.4, 0.6],
        ];
        let mat = multi_neuron_victor_purpura(&trains, 1000.0);
        for i in 0..3 {
            assert!(mat[i * 3 + i] < EPS, "diagonal should be 0");
        }
    }

    #[test]
    fn multi_vp_symmetric() {
        let trains = vec![
            vec![0.1, 0.3, 0.5],
            vec![0.15, 0.35, 0.55],
        ];
        let mat = multi_neuron_victor_purpura(&trains, 1000.0);
        assert!((mat[0 * 2 + 1] - mat[1 * 2 + 0]).abs() < EPS);
    }

    // ── Lempel-Ziv ──────────────────────────────────────────────────

    #[test]
    fn lz_constant() {
        let train = vec![1u8; 100];
        let c = lempel_ziv_complexity(&train);
        assert!(c < 1.0, "constant train should have low LZ, got {c}");
    }

    #[test]
    fn lz_alternating() {
        let train: Vec<u8> = (0..100).map(|i| (i % 2) as u8).collect();
        let c = lempel_ziv_complexity(&train);
        assert!(c > 0.0, "alternating should have non-zero LZ");
    }

    #[test]
    fn lz_empty() {
        assert!(lempel_ziv_complexity(&[]) == 0.0);
    }

    // ── Approximate entropy ─────────────────────────────────────────

    #[test]
    fn apen_regular() {
        let data: Vec<f64> = (0..200).map(|i| (i as f64 * 0.1).sin()).collect();
        let ae = approximate_entropy(&data, 2, 0.2);
        assert!(!ae.is_nan(), "ApEn should be computable");
        assert!(ae >= 0.0, "ApEn should be non-negative, got {ae}");
    }

    #[test]
    fn apen_short() {
        assert!(approximate_entropy(&[1.0, 2.0], 2, 0.2).is_nan());
    }

    // ── Sample entropy ──────────────────────────────────────────────

    #[test]
    fn sampen_regular() {
        let data: Vec<f64> = (0..200).map(|i| (i as f64 * 0.1).sin()).collect();
        let se = sample_entropy(&data, 2, 0.2);
        assert!(!se.is_nan(), "SampEn should be computable");
    }

    // ── Permutation entropy ─────────────────────────────────────────

    #[test]
    fn perm_ent_monotonic() {
        let data: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let pe = permutation_entropy(&data, 3, 1);
        assert!(pe < 0.1, "monotonic should have low perm entropy, got {pe}");
    }

    #[test]
    fn perm_ent_short() {
        assert!(permutation_entropy(&[1.0, 2.0], 3, 1).is_nan());
    }

    // ── KL Mutual Information ───────────────────────────────────────

    #[test]
    fn kl_mi_identical() {
        let x: Vec<f64> = (0..100).map(|i| i as f64 * 0.01).collect();
        let mi = kozachenko_leonenko_mi(&x, &x, 3);
        assert!(mi > 0.0, "identical vars should have MI > 0, got {mi}");
    }

    #[test]
    fn kl_mi_short() {
        let mi = kozachenko_leonenko_mi(&[1.0, 2.0], &[3.0, 4.0], 3);
        assert!(mi == 0.0, "too few points should return 0");
    }

    // ── Spike train entropy ─────────────────────────────────────────

    #[test]
    fn ste_constant() {
        let train = vec![0u8; 100];
        let h = spike_train_entropy(&train, 4);
        assert!(h < 1e-10, "constant train should have 0 entropy, got {h}");
    }

    #[test]
    fn ste_alternating() {
        let train: Vec<u8> = (0..100).map(|i| (i % 2) as u8).collect();
        let h = spike_train_entropy(&train, 4);
        assert!(h > 0.0, "alternating should have positive entropy");
    }

    #[test]
    fn ste_short() {
        assert!(spike_train_entropy(&[0, 1], 4).is_nan());
    }
}

// ---------------------------------------------------------------------------
// PyO3 bindings
// ---------------------------------------------------------------------------

#[cfg(feature = "pyo3_bindings")]
mod python {
    use super::*;
    use numpy::{PyArray1, PyReadonlyArray1, PyUntypedArrayMethods};
    use pyo3::prelude::*;

    #[pyfunction]
    fn py_van_rossum_distance<'py>(
        train_a: PyReadonlyArray1<'py, f64>,
        train_b: PyReadonlyArray1<'py, f64>,
        dt: f64,
        tau_ms: f64,
    ) -> PyResult<f64> {
        Ok(van_rossum_distance(
            train_a.as_slice()?,
            train_b.as_slice()?,
            dt,
            tau_ms,
        ))
    }

    #[pyfunction]
    fn py_victor_purpura_distance<'py>(
        times_a: PyReadonlyArray1<'py, f64>,
        times_b: PyReadonlyArray1<'py, f64>,
        cost_per_s: f64,
    ) -> PyResult<f64> {
        Ok(victor_purpura_distance(
            times_a.as_slice()?,
            times_b.as_slice()?,
            cost_per_s,
        ))
    }

    #[pyfunction]
    fn py_multi_neuron_vp<'py>(
        py: Python<'py>,
        spike_times_list: Vec<PyReadonlyArray1<'py, f64>>,
        cost_per_s: f64,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let trains: Vec<Vec<f64>> = spike_times_list
            .iter()
            .map(|a| a.as_slice().unwrap().to_vec())
            .collect();
        let mat = multi_neuron_victor_purpura(&trains, cost_per_s);
        Ok(PyArray1::from_vec(py, mat))
    }

    #[pyfunction]
    fn py_isi_distance<'py>(
        train_a: PyReadonlyArray1<'py, f64>,
        train_b: PyReadonlyArray1<'py, f64>,
        dt: f64,
    ) -> PyResult<f64> {
        Ok(isi_distance(
            train_a.as_slice()?,
            train_b.as_slice()?,
            dt,
        ))
    }

    #[pyfunction]
    fn py_spike_distance<'py>(
        times_a: PyReadonlyArray1<'py, f64>,
        times_b: PyReadonlyArray1<'py, f64>,
        t_start: f64,
        t_end: f64,
    ) -> PyResult<f64> {
        Ok(spike_distance(
            times_a.as_slice()?,
            times_b.as_slice()?,
            t_start,
            t_end,
        ))
    }

    #[pyfunction]
    fn py_spike_sync<'py>(
        times_a: PyReadonlyArray1<'py, f64>,
        times_b: PyReadonlyArray1<'py, f64>,
        t_start: f64,
        t_end: f64,
    ) -> PyResult<f64> {
        Ok(spike_sync(
            times_a.as_slice()?,
            times_b.as_slice()?,
            t_start,
            t_end,
        ))
    }

    #[pyfunction]
    fn py_hunter_milton<'py>(
        times_a: PyReadonlyArray1<'py, f64>,
        times_b: PyReadonlyArray1<'py, f64>,
        dt_max: f64,
    ) -> PyResult<f64> {
        Ok(hunter_milton_similarity(
            times_a.as_slice()?,
            times_b.as_slice()?,
            dt_max,
        ))
    }

    #[pyfunction]
    fn py_earth_movers_distance<'py>(
        times_a: PyReadonlyArray1<'py, f64>,
        times_b: PyReadonlyArray1<'py, f64>,
        t_start: f64,
        t_end: f64,
        n_bins: usize,
    ) -> PyResult<f64> {
        Ok(earth_movers_distance(
            times_a.as_slice()?,
            times_b.as_slice()?,
            t_start,
            t_end,
            n_bins,
        ))
    }

    #[pyfunction]
    fn py_cross_correlation<'py>(
        py: Python<'py>,
        times_a: PyReadonlyArray1<'py, f64>,
        times_b: PyReadonlyArray1<'py, f64>,
        bin_size: f64,
        max_lag: f64,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let hist = cross_correlation(
            times_a.as_slice()?,
            times_b.as_slice()?,
            bin_size,
            max_lag,
        );
        Ok(PyArray1::from_vec(py, hist))
    }

    #[pyfunction]
    fn py_event_synchronization<'py>(
        times_a: PyReadonlyArray1<'py, f64>,
        times_b: PyReadonlyArray1<'py, f64>,
        tau: f64,
    ) -> PyResult<f64> {
        Ok(event_synchronization(
            times_a.as_slice()?,
            times_b.as_slice()?,
            tau,
        ))
    }

    #[pyfunction]
    fn py_sttc<'py>(
        times_a: PyReadonlyArray1<'py, f64>,
        times_b: PyReadonlyArray1<'py, f64>,
        dt: f64,
        t_start: f64,
        t_end: f64,
    ) -> PyResult<f64> {
        Ok(spike_time_tiling_coefficient(
            times_a.as_slice()?,
            times_b.as_slice()?,
            dt,
            t_start,
            t_end,
        ))
    }

    #[pyfunction]
    fn py_coincidence_index<'py>(
        times_a: PyReadonlyArray1<'py, f64>,
        times_b: PyReadonlyArray1<'py, f64>,
        window: f64,
    ) -> PyResult<f64> {
        Ok(coincidence_index(
            times_a.as_slice()?,
            times_b.as_slice()?,
            window,
        ))
    }

    #[pyfunction]
    fn py_lempel_ziv_complexity<'py>(
        data: PyReadonlyArray1<'py, u8>,
    ) -> PyResult<f64> {
        Ok(lempel_ziv_complexity(data.as_slice()?))
    }

    #[pyfunction]
    fn py_approximate_entropy<'py>(
        data: PyReadonlyArray1<'py, f64>,
        m: usize,
        r: f64,
    ) -> PyResult<f64> {
        Ok(approximate_entropy(data.as_slice()?, m, r))
    }

    #[pyfunction]
    fn py_sample_entropy<'py>(
        data: PyReadonlyArray1<'py, f64>,
        m: usize,
        r: f64,
    ) -> PyResult<f64> {
        Ok(sample_entropy(data.as_slice()?, m, r))
    }

    #[pyfunction]
    fn py_permutation_entropy<'py>(
        data: PyReadonlyArray1<'py, f64>,
        order: usize,
        delay: usize,
    ) -> PyResult<f64> {
        Ok(permutation_entropy(data.as_slice()?, order, delay))
    }

    #[pyfunction]
    fn py_kozachenko_leonenko_mi<'py>(
        x: PyReadonlyArray1<'py, f64>,
        y: PyReadonlyArray1<'py, f64>,
        k: usize,
    ) -> PyResult<f64> {
        Ok(kozachenko_leonenko_mi(
            x.as_slice()?,
            y.as_slice()?,
            k,
        ))
    }

    #[pyfunction]
    fn py_spike_train_entropy<'py>(
        binned: PyReadonlyArray1<'py, u8>,
        word_length: usize,
    ) -> PyResult<f64> {
        Ok(spike_train_entropy(binned.as_slice()?, word_length))
    }

    #[pymodule]
    fn spike_stats_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_function(wrap_pyfunction!(py_van_rossum_distance, m)?)?;
        m.add_function(wrap_pyfunction!(py_victor_purpura_distance, m)?)?;
        m.add_function(wrap_pyfunction!(py_multi_neuron_vp, m)?)?;
        m.add_function(wrap_pyfunction!(py_isi_distance, m)?)?;
        m.add_function(wrap_pyfunction!(py_spike_distance, m)?)?;
        m.add_function(wrap_pyfunction!(py_spike_sync, m)?)?;
        m.add_function(wrap_pyfunction!(py_hunter_milton, m)?)?;
        m.add_function(wrap_pyfunction!(py_earth_movers_distance, m)?)?;
        m.add_function(wrap_pyfunction!(py_cross_correlation, m)?)?;
        m.add_function(wrap_pyfunction!(py_event_synchronization, m)?)?;
        m.add_function(wrap_pyfunction!(py_sttc, m)?)?;
        m.add_function(wrap_pyfunction!(py_coincidence_index, m)?)?;
        m.add_function(wrap_pyfunction!(py_lempel_ziv_complexity, m)?)?;
        m.add_function(wrap_pyfunction!(py_approximate_entropy, m)?)?;
        m.add_function(wrap_pyfunction!(py_sample_entropy, m)?)?;
        m.add_function(wrap_pyfunction!(py_permutation_entropy, m)?)?;
        m.add_function(wrap_pyfunction!(py_kozachenko_leonenko_mi, m)?)?;
        m.add_function(wrap_pyfunction!(py_spike_train_entropy, m)?)?;
        Ok(())
    }
}
