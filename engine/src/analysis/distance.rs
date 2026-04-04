// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Spike train distance and similarity metrics

use super::basic::isi;
use super::rate::instantaneous_rate;

/// Van Rossum 2001 — exponential-kernel spike train distance.
pub fn van_rossum_distance(train_a: &[i32], train_b: &[i32], dt: f64, tau_ms: f64) -> f64 {
    let tau = tau_ms / 1000.0;
    let n = train_a.len().min(train_b.len());
    if n == 0 || tau <= 0.0 {
        return 0.0;
    }

    // Build exponential decay kernel
    let decay: Vec<f64> = (0..n).map(|i| (-((i as f64) * dt) / tau).exp()).collect();

    // Convolve (full, truncated to n)
    let fa = convolve_full_truncated(train_a, &decay, n);
    let fb = convolve_full_truncated(train_b, &decay, n);

    let sum_sq: f64 = fa.iter().zip(fb.iter()).map(|(a, b)| (a - b).powi(2)).sum();
    (sum_sq * dt / tau).sqrt()
}

fn convolve_full_truncated(signal: &[i32], kernel: &[f64], out_len: usize) -> Vec<f64> {
    let n = signal.len().min(out_len);
    let mut result = vec![0.0_f64; out_len];
    for i in 0..n {
        if signal[i] == 0 {
            continue;
        }
        let s = signal[i] as f64;
        for (j, &k) in kernel.iter().enumerate() {
            let idx = i + j;
            if idx >= out_len {
                break;
            }
            result[idx] += s * k;
        }
    }
    result
}

/// Victor-Purpura 1996 — edit distance between spike time arrays.
/// cost_per_s: cost of shifting a spike by 1 second (q parameter).
pub fn victor_purpura_distance(times_a: &[f64], times_b: &[f64], cost_per_s: f64) -> f64 {
    let na = times_a.len();
    let nb = times_b.len();
    if na == 0 {
        return nb as f64;
    }
    if nb == 0 {
        return na as f64;
    }

    // DP table
    let mut d = vec![vec![0.0_f64; nb + 1]; na + 1];
    for i in 0..=na {
        d[i][0] = i as f64;
    }
    for j in 0..=nb {
        d[0][j] = j as f64;
    }
    for i in 1..=na {
        for j in 1..=nb {
            let shift_cost = cost_per_s * (times_a[i - 1] - times_b[j - 1]).abs();
            d[i][j] = (d[i - 1][j] + 1.0)
                .min(d[i][j - 1] + 1.0)
                .min(d[i - 1][j - 1] + shift_cost);
        }
    }
    d[na][nb]
}

/// ISI-distance (Kreuz et al. 2007) — ratio-based ISI comparison.
pub fn isi_distance(train_a: &[i32], train_b: &[i32], dt: f64) -> f64 {
    let isi_a = isi(train_a, dt);
    let isi_b = isi(train_b, dt);
    let n = isi_a.len().min(isi_b.len());
    if n == 0 {
        return f64::NAN;
    }
    let sum: f64 = (0..n)
        .map(|i| {
            let a = isi_a[i];
            let b = isi_b[i];
            if a == 0.0 && b == 0.0 {
                0.0
            } else if a <= b {
                if b > 0.0 {
                    (a / b - 1.0).abs()
                } else {
                    0.0
                }
            } else if a > 0.0 {
                (b / a - 1.0).abs()
            } else {
                0.0
            }
        })
        .sum();
    sum / n as f64
}

/// SPIKE-distance (Kreuz et al. 2013).
/// Time-resolved distance based on nearest-neighbour spike differences.
pub fn spike_distance(times_a: &[f64], times_b: &[f64], t_start: f64, t_end: f64) -> f64 {
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
    ta.sort_by(|a, b| a.partial_cmp(b).unwrap());
    tb.sort_by(|a, b| a.partial_cmp(b).unwrap());

    if ta.is_empty() && tb.is_empty() {
        return 0.0;
    }
    if ta.is_empty() || tb.is_empty() {
        return 1.0;
    }

    let n_eval = 100_usize;
    let step = (t_end - t_start) / (n_eval - 1).max(1) as f64;
    let mut sum = 0.0_f64;

    for k in 0..n_eval {
        let t = t_start + k as f64 * step;
        let idx_a = ta.partition_point(|&x| x <= t);
        let idx_b = tb.partition_point(|&x| x <= t);

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

/// SPIKE-synchronization (Kreuz et al. 2015). Coincidence-based, normalised to [0, 1].
pub fn spike_sync(times_a: &[f64], times_b: &[f64], t_start: f64, t_end: f64) -> f64 {
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
    ta.sort_by(|a, b| a.partial_cmp(b).unwrap());
    tb.sort_by(|a, b| a.partial_cmp(b).unwrap());

    if ta.is_empty() || tb.is_empty() {
        return 0.0;
    }

    let total_possible = ta.len() + tb.len();
    let mut total_coincidences = 0_usize;

    // Forward pass: for each spike in A, find nearest in B
    for (i, &t) in ta.iter().enumerate() {
        let j = nearest_idx(&tb, t);
        let isi_a = local_isi(&ta, i);
        let isi_b = local_isi(&tb, j);
        let tau = isi_a.min(isi_b) / 2.0;
        if tau > 0.0 && (tb[j] - t).abs() < tau {
            total_coincidences += 1;
        }
    }

    // Reverse pass: for each spike in B, find nearest in A
    for (j, &t) in tb.iter().enumerate() {
        let i = nearest_idx(&ta, t);
        let isi_a = local_isi(&ta, i);
        let isi_b = local_isi(&tb, j);
        let tau = isi_a.min(isi_b) / 2.0;
        if tau > 0.0 && (ta[i] - t).abs() < tau {
            total_coincidences += 1;
        }
    }

    if total_possible == 0 {
        return 0.0;
    }
    total_coincidences as f64 / total_possible as f64
}

fn nearest_idx(sorted: &[f64], target: f64) -> usize {
    if sorted.is_empty() {
        return 0;
    }
    let pos = sorted.partition_point(|&x| x < target);
    if pos == 0 {
        return 0;
    }
    if pos >= sorted.len() {
        return sorted.len() - 1;
    }
    if (sorted[pos] - target).abs() < (sorted[pos - 1] - target).abs() {
        pos
    } else {
        pos - 1
    }
}

/// Binned SPIKE-synchronization profile (Kreuz et al. 2015).
pub fn spike_sync_profile(
    times_a: &[f64],
    times_b: &[f64],
    n_bins: usize,
    t_start: f64,
    t_end: f64,
) -> Vec<f64> {
    let mut profile = vec![0.0_f64; n_bins];
    let bin_width = (t_end - t_start) / n_bins as f64;
    for k in 0..n_bins {
        let lo = t_start + k as f64 * bin_width;
        let hi = lo + bin_width;
        let sub_a: Vec<f64> = times_a
            .iter()
            .copied()
            .filter(|&t| t >= lo && t < hi)
            .collect();
        let sub_b: Vec<f64> = times_b
            .iter()
            .copied()
            .filter(|&t| t >= lo && t < hi)
            .collect();
        if !sub_a.is_empty() || !sub_b.is_empty() {
            profile[k] = spike_sync(&sub_a, &sub_b, lo, hi);
        }
    }
    profile
}

/// Binned SPIKE-distance profile (Kreuz et al. 2013).
pub fn spike_profile(
    times_a: &[f64],
    times_b: &[f64],
    n_bins: usize,
    t_start: f64,
    t_end: f64,
) -> Vec<f64> {
    let mut profile = vec![0.0_f64; n_bins];
    let bin_width = (t_end - t_start) / n_bins as f64;
    for k in 0..n_bins {
        let lo = t_start + k as f64 * bin_width;
        let hi = lo + bin_width;
        let sub_a: Vec<f64> = times_a
            .iter()
            .copied()
            .filter(|&t| t >= lo && t < hi)
            .collect();
        let sub_b: Vec<f64> = times_b
            .iter()
            .copied()
            .filter(|&t| t >= lo && t < hi)
            .collect();
        profile[k] = spike_distance(&sub_a, &sub_b, lo, hi);
    }
    profile
}

/// Binned ISI-distance profile (Kreuz et al. 2007).
pub fn isi_profile(
    binary_train_a: &[i32],
    binary_train_b: &[i32],
    dt: f64,
    n_bins: usize,
) -> Vec<f64> {
    let n = binary_train_a.len().min(binary_train_b.len());
    let bin_size = (n / n_bins).max(1);
    let mut profile = vec![0.0_f64; n_bins];
    for k in 0..n_bins {
        let start = k * bin_size;
        let end = (start + bin_size).min(n);
        if start >= n {
            break;
        }
        profile[k] = isi_distance(&binary_train_a[start..end], &binary_train_b[start..end], dt);
    }
    profile
}

/// Adaptive SPIKE-distance (Kreuz et al. 2013).
/// cost=0: pure SPIKE-distance. cost=1: ISI-like weighting.
pub fn adaptive_spike_distance(
    times_a: &[f64],
    times_b: &[f64],
    t_start: f64,
    t_end: f64,
    cost: f64,
) -> f64 {
    let sd = spike_distance(times_a, times_b, t_start, t_end);
    let ta: Vec<f64> = times_a
        .iter()
        .copied()
        .filter(|&t| t >= t_start && t <= t_end)
        .collect();
    let tb: Vec<f64> = times_b
        .iter()
        .copied()
        .filter(|&t| t >= t_start && t <= t_end)
        .collect();

    let mean_isi = |times: &[f64]| -> f64 {
        if times.len() > 1 {
            let mut sorted = times.to_vec();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let diffs: Vec<f64> = sorted.windows(2).map(|w| w[1] - w[0]).collect();
            diffs.iter().sum::<f64>() / diffs.len() as f64
        } else {
            t_end - t_start
        }
    };

    let mean_a = mean_isi(&ta);
    let mean_b = mean_isi(&tb);
    let ratio = (mean_a - mean_b).abs() / (mean_a + mean_b).max(1e-30);
    (1.0 - cost) * sd + cost * ratio
}

/// Schreiber et al. 2003 — spike train similarity via smoothed correlation.
pub fn schreiber_similarity(train_a: &[i32], train_b: &[i32], dt: f64, sigma_ms: f64) -> f64 {
    let fa: Vec<f64> = train_a.iter().map(|&v| v as f64).collect();
    let fb: Vec<f64> = train_b.iter().map(|&v| v as f64).collect();
    let ra = instantaneous_rate(&fa, dt, "gaussian", sigma_ms);
    let rb = instantaneous_rate(&fb, dt, "gaussian", sigma_ms);
    let n = ra.len().min(rb.len());
    if n == 0 {
        return 0.0;
    }

    let mean_a: f64 = ra[..n].iter().sum::<f64>() / n as f64;
    let mean_b: f64 = rb[..n].iter().sum::<f64>() / n as f64;

    let mut sum_ab = 0.0_f64;
    let mut sum_aa = 0.0_f64;
    let mut sum_bb = 0.0_f64;
    for i in 0..n {
        let a = ra[i] - mean_a;
        let b = rb[i] - mean_b;
        sum_ab += a * b;
        sum_aa += a * a;
        sum_bb += b * b;
    }
    let denom = (sum_aa * sum_bb).sqrt();
    if denom == 0.0 {
        return 0.0;
    }
    sum_ab / denom
}

/// Hunter-Milton 2003 similarity. Fraction of spikes with nearest-neighbour < dt_max.
pub fn hunter_milton_similarity(times_a: &[f64], times_b: &[f64], dt_max: f64) -> f64 {
    if times_a.is_empty() || times_b.is_empty() {
        return 0.0;
    }
    let total = times_a.len() + times_b.len();
    let mut count = 0_usize;
    for &t in times_a {
        if times_b.iter().any(|&tb| (tb - t).abs() < dt_max) {
            count += 1;
        }
    }
    for &t in times_b {
        if times_a.iter().any(|&ta| (ta - t).abs() < dt_max) {
            count += 1;
        }
    }
    count as f64 / total as f64
}

/// Earth mover's distance between spike time distributions (Rubner et al. 1998).
pub fn earth_movers_distance(
    times_a: &[f64],
    times_b: &[f64],
    t_start: f64,
    t_end: f64,
    n_bins: usize,
) -> f64 {
    let bin_width = (t_end - t_start) / n_bins as f64;

    let histogram = |times: &[f64]| -> Vec<f64> {
        let mut h = vec![0.0_f64; n_bins];
        for &t in times {
            let idx = ((t - t_start) / bin_width) as usize;
            if idx < n_bins {
                h[idx] += 1.0;
            }
        }
        let s: f64 = h.iter().sum();
        if s > 0.0 {
            for v in &mut h {
                *v /= s;
            }
        }
        h
    };

    let ha = histogram(times_a);
    let hb = histogram(times_b);

    // Cumulative sum difference
    let mut cum_a = 0.0_f64;
    let mut cum_b = 0.0_f64;
    let mut emd = 0.0_f64;
    for i in 0..n_bins {
        cum_a += ha[i];
        cum_b += hb[i];
        emd += (cum_a - cum_b).abs();
    }
    emd * bin_width
}

/// All-pairs Victor-Purpura distance matrix for multiple neurons.
pub fn multi_neuron_victor_purpura(spike_times_list: &[&[f64]], cost_per_s: f64) -> Vec<Vec<f64>> {
    let n = spike_times_list.len();
    let mut mat = vec![vec![0.0_f64; n]; n];
    for i in 0..n {
        for j in (i + 1)..n {
            let d = victor_purpura_distance(spike_times_list[i], spike_times_list[j], cost_per_s);
            mat[i][j] = d;
            mat[j][i] = d;
        }
    }
    mat
}

/// Generalized Victor-Purpura with arbitrary cost function.
/// cost_func(dt) returns the cost of shifting a spike by dt seconds.
pub fn generalized_victor_purpura(
    times_a: &[f64],
    times_b: &[f64],
    cost_func: fn(f64) -> f64,
) -> f64 {
    let na = times_a.len();
    let nb = times_b.len();
    if na == 0 {
        return nb as f64;
    }
    if nb == 0 {
        return na as f64;
    }
    let mut d = vec![vec![0.0_f64; nb + 1]; na + 1];
    for i in 0..=na {
        d[i][0] = i as f64;
    }
    for j in 0..=nb {
        d[0][j] = j as f64;
    }
    for i in 1..=na {
        for j in 1..=nb {
            let shift = cost_func(times_a[i - 1] - times_b[j - 1]);
            d[i][j] = (d[i - 1][j] + 1.0)
                .min(d[i][j - 1] + 1.0)
                .min(d[i - 1][j - 1] + shift);
        }
    }
    d[na][nb]
}

/// All-pairs spike train distance matrix.
/// metric: "spike_distance", "spike_sync", "victor_purpura".
pub fn spike_distance_matrix(
    spike_times_list: &[&[f64]],
    metric: &str,
    t_start: f64,
    t_end: f64,
) -> Vec<Vec<f64>> {
    let n = spike_times_list.len();
    let mut mat = vec![vec![0.0_f64; n]; n];
    for i in 0..n {
        for j in (i + 1)..n {
            let d = match metric {
                "spike_sync" => {
                    1.0 - spike_sync(spike_times_list[i], spike_times_list[j], t_start, t_end)
                }
                "victor_purpura" => {
                    victor_purpura_distance(spike_times_list[i], spike_times_list[j], 1000.0)
                }
                _ => spike_distance(spike_times_list[i], spike_times_list[j], t_start, t_end),
            };
            mat[i][j] = d;
            mat[j][i] = d;
        }
    }
    mat
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_train(spikes: &[usize], len: usize) -> Vec<i32> {
        let mut t = vec![0i32; len];
        for &s in spikes {
            t[s] = 1;
        }
        t
    }

    // ── van_rossum_distance ─────────────────────────────────────────

    #[test]
    fn test_van_rossum_identical() {
        let train = make_train(&[10, 30, 50], 100);
        let d = van_rossum_distance(&train, &train, 0.001, 10.0);
        assert!(d.abs() < 1e-10, "identical trains → distance 0, got {d}");
    }

    #[test]
    fn test_van_rossum_different() {
        let a = make_train(&[10, 30, 50], 100);
        let b = make_train(&[20, 40, 60], 100);
        let d = van_rossum_distance(&a, &b, 0.001, 10.0);
        assert!(d > 0.0, "different trains → positive distance");
    }

    #[test]
    fn test_van_rossum_symmetry() {
        let a = make_train(&[10, 30], 100);
        let b = make_train(&[20, 40], 100);
        let d1 = van_rossum_distance(&a, &b, 0.001, 10.0);
        let d2 = van_rossum_distance(&b, &a, 0.001, 10.0);
        assert!((d1 - d2).abs() < 1e-10, "distance should be symmetric");
    }

    #[test]
    fn test_van_rossum_empty() {
        let a = vec![0i32; 100];
        let b = make_train(&[50], 100);
        let d = van_rossum_distance(&a, &b, 0.001, 10.0);
        assert!(d > 0.0, "empty vs non-empty → positive distance");
    }

    // ── victor_purpura_distance ─────────────────────────────────────

    #[test]
    fn test_vp_identical() {
        let times = vec![0.1, 0.3, 0.5];
        let d = victor_purpura_distance(&times, &times, 1000.0);
        assert!(d < 1e-10, "identical spike times → 0");
    }

    #[test]
    fn test_vp_empty_vs_spikes() {
        let d = victor_purpura_distance(&[], &[0.1, 0.2, 0.3], 1000.0);
        assert!((d - 3.0).abs() < 1e-10, "3 insertions → cost 3");
    }

    #[test]
    fn test_vp_close_spikes() {
        let a = vec![0.1];
        let b = vec![0.1001];
        let d = victor_purpura_distance(&a, &b, 1000.0);
        // shift cost = 1000 * 0.0001 = 0.1, cheaper than delete+insert = 2
        assert!(d < 0.2, "close spikes → small shift cost, got {d}");
    }

    #[test]
    fn test_vp_symmetry() {
        let a = vec![0.1, 0.5];
        let b = vec![0.3, 0.7];
        let d1 = victor_purpura_distance(&a, &b, 1000.0);
        let d2 = victor_purpura_distance(&b, &a, 1000.0);
        assert!((d1 - d2).abs() < 1e-10);
    }

    // ── isi_distance ────────────────────────────────────────────────

    #[test]
    fn test_isi_distance_identical() {
        let train = make_train(&[10, 30, 50, 70], 100);
        let d = isi_distance(&train, &train, 0.001);
        assert!(d.abs() < 1e-10, "identical trains → ISI distance 0");
    }

    #[test]
    fn test_isi_distance_empty() {
        let a = make_train(&[50], 100);
        let b = make_train(&[50], 100);
        // Only 1 spike each → no ISI → NaN
        let d = isi_distance(&a, &b, 0.001);
        assert!(d.is_nan(), "single spike → NaN ISI distance");
    }

    #[test]
    fn test_isi_distance_different() {
        let a = make_train(&[10, 20, 30], 100); // uniform ISIs
        let b = make_train(&[10, 15, 30], 100); // non-uniform
        let d = isi_distance(&a, &b, 0.001);
        assert!(d > 0.0, "different ISI patterns → positive distance");
    }

    // ── spike_distance ──────────────────────────────────────────────

    #[test]
    fn test_spike_distance_identical() {
        let times = vec![0.1, 0.3, 0.5, 0.7, 0.9];
        let d = spike_distance(&times, &times, 0.0, 1.0);
        assert!(d < 1e-10, "identical → 0, got {d}");
    }

    #[test]
    fn test_spike_distance_empty_both() {
        let d = spike_distance(&[], &[], 0.0, 1.0);
        assert_eq!(d, 0.0);
    }

    #[test]
    fn test_spike_distance_one_empty() {
        let d = spike_distance(&[0.5], &[], 0.0, 1.0);
        assert_eq!(d, 1.0);
    }

    #[test]
    fn test_spike_distance_symmetry() {
        let a = vec![0.2, 0.6];
        let b = vec![0.4, 0.8];
        let d1 = spike_distance(&a, &b, 0.0, 1.0);
        let d2 = spike_distance(&b, &a, 0.0, 1.0);
        assert!((d1 - d2).abs() < 1e-10);
    }

    // ── spike_sync ──────────────────────────────────────────────────

    #[test]
    fn test_spike_sync_identical() {
        let times = vec![0.1, 0.3, 0.5, 0.7, 0.9];
        let s = spike_sync(&times, &times, 0.0, 1.0);
        assert!(
            (s - 1.0).abs() < 1e-10,
            "identical trains → sync=1.0, got {s}"
        );
    }

    #[test]
    fn test_spike_sync_empty() {
        let s = spike_sync(&[], &[0.5], 0.0, 1.0);
        assert_eq!(s, 0.0);
    }

    #[test]
    fn test_spike_sync_far_apart() {
        let a = vec![0.1];
        let b = vec![0.9];
        let s = spike_sync(&a, &b, 0.0, 1.0);
        assert!(s < 0.5, "far apart → low sync, got {s}");
    }

    // ── spike_sync_profile ──────────────────────────────────────────

    #[test]
    fn test_sync_profile_length() {
        let a = vec![0.1, 0.3, 0.5];
        let b = vec![0.1, 0.3, 0.5];
        let p = spike_sync_profile(&a, &b, 10, 0.0, 1.0);
        assert_eq!(p.len(), 10);
    }

    // ── spike_profile ───────────────────────────────────────────────

    #[test]
    fn test_spike_profile_length() {
        let a = vec![0.2, 0.6];
        let b = vec![0.4, 0.8];
        let p = spike_profile(&a, &b, 5, 0.0, 1.0);
        assert_eq!(p.len(), 5);
    }

    // ── isi_profile ─────────────────────────────────────────────────

    #[test]
    fn test_isi_profile_length() {
        let a = make_train(&[5, 15, 25, 35, 45], 50);
        let b = make_train(&[5, 15, 25, 35, 45], 50);
        let p = isi_profile(&a, &b, 0.001, 5);
        assert_eq!(p.len(), 5);
    }

    // ── adaptive_spike_distance ─────────────────────────────────────

    #[test]
    fn test_adaptive_cost_zero_equals_spike() {
        let a = vec![0.2, 0.6];
        let b = vec![0.4, 0.8];
        let sd = spike_distance(&a, &b, 0.0, 1.0);
        let ad = adaptive_spike_distance(&a, &b, 0.0, 1.0, 0.0);
        assert!((sd - ad).abs() < 1e-10, "cost=0 → pure spike distance");
    }

    #[test]
    fn test_adaptive_cost_one() {
        let a = vec![0.2, 0.6];
        let b = vec![0.4, 0.8];
        let ad = adaptive_spike_distance(&a, &b, 0.0, 1.0, 1.0);
        assert!(ad >= 0.0, "cost=1 → non-negative");
    }

    // ── schreiber_similarity ────────────────────────────────────────

    #[test]
    fn test_schreiber_identical() {
        let train = make_train(&[10, 30, 50, 70, 90], 100);
        let s = schreiber_similarity(&train, &train, 0.001, 5.0);
        assert!(
            (s - 1.0).abs() < 1e-6,
            "identical trains → similarity 1.0, got {s}"
        );
    }

    #[test]
    fn test_schreiber_empty() {
        let a = vec![0i32; 100];
        let b = vec![0i32; 100];
        let s = schreiber_similarity(&a, &b, 0.001, 5.0);
        assert_eq!(s, 0.0, "zero trains → 0.0");
    }

    // ── hunter_milton_similarity ────────────────────────────────────

    #[test]
    fn test_hunter_identical() {
        let times = vec![0.1, 0.3, 0.5];
        let s = hunter_milton_similarity(&times, &times, 0.01);
        assert!((s - 1.0).abs() < 1e-10, "identical → 1.0, got {s}");
    }

    #[test]
    fn test_hunter_empty() {
        let s = hunter_milton_similarity(&[], &[0.5], 0.01);
        assert_eq!(s, 0.0);
    }

    #[test]
    fn test_hunter_far_apart() {
        let a = vec![0.1];
        let b = vec![0.9];
        let s = hunter_milton_similarity(&a, &b, 0.01);
        assert_eq!(s, 0.0, "far apart → 0.0");
    }

    // ── earth_movers_distance ───────────────────────────────────────

    #[test]
    fn test_emd_identical() {
        let times = vec![0.1, 0.3, 0.5, 0.7, 0.9];
        let d = earth_movers_distance(&times, &times, 0.0, 1.0, 100);
        assert!(d < 1e-10, "identical → 0, got {d}");
    }

    #[test]
    fn test_emd_different() {
        let a = vec![0.1, 0.2];
        let b = vec![0.8, 0.9];
        let d = earth_movers_distance(&a, &b, 0.0, 1.0, 100);
        assert!(d > 0.0, "different distributions → positive EMD");
    }

    #[test]
    fn test_emd_symmetry() {
        let a = vec![0.1, 0.2];
        let b = vec![0.8, 0.9];
        let d1 = earth_movers_distance(&a, &b, 0.0, 1.0, 100);
        let d2 = earth_movers_distance(&b, &a, 0.0, 1.0, 100);
        assert!((d1 - d2).abs() < 1e-10);
    }

    // ── multi_neuron_victor_purpura ─────────────────────────────────

    #[test]
    fn test_multi_vp_diagonal_zero() {
        let t1 = vec![0.1, 0.3];
        let t2 = vec![0.2, 0.4];
        let trains: Vec<&[f64]> = vec![&t1, &t2];
        let mat = multi_neuron_victor_purpura(&trains, 1000.0);
        assert_eq!(mat[0][0], 0.0);
        assert_eq!(mat[1][1], 0.0);
        assert!(mat[0][1] > 0.0);
        assert!((mat[0][1] - mat[1][0]).abs() < 1e-10);
    }

    // ── generalized_victor_purpura ──────────────────────────────────

    #[test]
    fn test_gen_vp_linear_matches_standard() {
        fn linear_cost(dt: f64) -> f64 {
            1000.0 * dt.abs()
        }
        let a = vec![0.1, 0.5];
        let b = vec![0.2, 0.6];
        let d_gen = generalized_victor_purpura(&a, &b, linear_cost);
        let d_std = victor_purpura_distance(&a, &b, 1000.0);
        assert!((d_gen - d_std).abs() < 1e-10);
    }

    #[test]
    fn test_gen_vp_quadratic_cost() {
        fn quad_cost(dt: f64) -> f64 {
            1000.0 * dt * dt
        }
        let a = vec![0.1];
        let b = vec![0.2];
        let d = generalized_victor_purpura(&a, &b, quad_cost);
        // shift cost = 1000 * 0.01 = 10 → cheaper to delete+insert (cost=2)
        assert!(
            (d - 2.0).abs() < 1e-10,
            "high shift cost → delete+insert, got {d}"
        );
    }

    // ── spike_distance_matrix ───────────────────────────────────────

    #[test]
    fn test_distance_matrix_shape() {
        let t1 = vec![0.1, 0.5];
        let t2 = vec![0.2, 0.6];
        let t3 = vec![0.3, 0.7];
        let trains: Vec<&[f64]> = vec![&t1, &t2, &t3];
        let mat = spike_distance_matrix(&trains, "spike_distance", 0.0, 1.0);
        assert_eq!(mat.len(), 3);
        assert_eq!(mat[0].len(), 3);
        assert_eq!(mat[0][0], 0.0);
        assert!((mat[0][1] - mat[1][0]).abs() < 1e-10);
    }

    #[test]
    fn test_distance_matrix_vp_metric() {
        let t1 = vec![0.1];
        let t2 = vec![0.9];
        let trains: Vec<&[f64]> = vec![&t1, &t2];
        let mat = spike_distance_matrix(&trains, "victor_purpura", 0.0, 1.0);
        assert!(mat[0][1] > 0.0);
    }

    // ── helpers ─────────────────────────────────────────────────────

    #[test]
    fn test_local_isi_boundaries() {
        let times = vec![0.1, 0.3, 0.7];
        assert!((local_isi(&times, 0) - 0.2).abs() < 1e-10);
        assert!((local_isi(&times, 2) - 0.4).abs() < 1e-10);
        // Middle: min(0.2, 0.4) = 0.2
        assert!((local_isi(&times, 1) - 0.2).abs() < 1e-10);
    }

    #[test]
    fn test_local_isi_single() {
        assert_eq!(local_isi(&[0.5], 0), 1.0);
    }

    #[test]
    fn test_nearest_idx() {
        let sorted = vec![0.1, 0.3, 0.5, 0.7];
        assert_eq!(nearest_idx(&sorted, 0.0), 0);
        assert_eq!(nearest_idx(&sorted, 0.29), 1);
        assert_eq!(nearest_idx(&sorted, 0.9), 3);
    }
}
