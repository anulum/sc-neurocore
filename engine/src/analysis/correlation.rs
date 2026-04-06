// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Cross-correlation, synchrony, and covariance measures

use rayon::prelude::*;
use rustfft::{num_complex::Complex, FftPlanner};

use super::basic::{bin_spike_train, spike_times};

/// Cross-correlogram between two binary spike trains.
/// Returns (correlation, lags_ms).
pub fn cross_correlation(
    train_a: &[i32],
    train_b: &[i32],
    max_lag_ms: f64,
    dt: f64,
) -> (Vec<f64>, Vec<f64>) {
    let max_lag = (max_lag_ms / (dt * 1000.0)) as isize;
    let n = train_a.len().min(train_b.len());
    if n == 0 {
        return (vec![], vec![]);
    }

    let mean_a: f64 = train_a[..n].iter().map(|&v| v as f64).sum::<f64>() / n as f64;
    let mean_b: f64 = train_b[..n].iter().map(|&v| v as f64).sum::<f64>() / n as f64;
    let a: Vec<f64> = train_a[..n].iter().map(|&v| v as f64 - mean_a).collect();
    let b: Vec<f64> = train_b[..n].iter().map(|&v| v as f64 - mean_b).collect();

    let norm = (a.iter().map(|x| x * x).sum::<f64>() * b.iter().map(|x| x * x).sum::<f64>()).sqrt();

    let n_lags = (2 * max_lag + 1) as usize;
    let mut cc = vec![0.0_f64; n_lags];
    let mut lags_ms = Vec::with_capacity(n_lags);
    for l in -max_lag..=max_lag {
        lags_ms.push(l as f64 * dt * 1000.0);
    }

    if norm == 0.0 {
        return (cc, lags_ms);
    }

    for (i, lag) in (-max_lag..=max_lag).enumerate() {
        let sum = if lag >= 0 {
            let l = lag as usize;
            crate::simd::dot_f64_dispatch(&a[..n - l], &b[l..n])
        } else {
            let l = (-lag) as usize;
            crate::simd::dot_f64_dispatch(&a[l..n], &b[..n - l])
        };
        cc[i] = sum / norm;
    }

    (cc, lags_ms)
}

/// Pairwise Pearson correlation matrix across neurons.
pub fn pairwise_correlation(trains: &[&[i32]], dt: f64) -> Vec<Vec<f64>> {
    let _ = dt;
    let n = trains.len();
    if n == 0 {
        return vec![vec![]];
    }
    let min_len = trains.iter().map(|t| t.len()).min().unwrap_or(0);
    if min_len == 0 {
        return vec![vec![0.0; n]; n];
    }

    let mat: Vec<Vec<f64>> = trains
        .iter()
        .map(|t| t[..min_len].iter().map(|&v| v as f64).collect::<Vec<f64>>())
        .collect();

    let means: Vec<f64> = mat
        .iter()
        .map(|row| row.iter().sum::<f64>() / min_len as f64)
        .collect();
    let stds: Vec<f64> = mat
        .iter()
        .enumerate()
        .map(|(i, row)| {
            (row.iter().map(|v| (v - means[i]).powi(2)).sum::<f64>() / min_len as f64).sqrt()
        })
        .collect();

    let mut corr = vec![vec![0.0_f64; n]; n];
    for i in 0..n {
        corr[i][i] = 1.0;
        for j in (i + 1)..n {
            if stds[i] > 0.0 && stds[j] > 0.0 {
                let cov: f64 = (0..min_len)
                    .map(|k| (mat[i][k] - means[i]) * (mat[j][k] - means[j]))
                    .sum::<f64>()
                    / min_len as f64;
                let r = cov / (stds[i] * stds[j]);
                corr[i][j] = r;
                corr[j][i] = r;
            }
        }
    }
    corr
}

/// Event synchronisation (Quian Quiroga et al. 2002).
/// Returns synchrony score in [0, 1].
pub fn event_synchronization(train_a: &[i32], train_b: &[i32], dt: f64, tau_ms: f64) -> f64 {
    let ta = spike_times(train_a, dt);
    let tb = spike_times(train_b, dt);
    let na = ta.len();
    let nb = tb.len();
    if na == 0 || nb == 0 {
        return 0.0;
    }
    let tau = tau_ms / 1000.0;
    let mut count = 0_usize;
    for &ti in &ta {
        for &tj in &tb {
            if (ti - tj).abs() < tau {
                count += 1;
            }
        }
    }
    count as f64 / (na as f64 * nb as f64).sqrt()
}

/// Magnitude-squared coherence between two binary spike trains.
/// Returns (coherence, freqs_hz).
pub fn spike_train_coherence(train_a: &[i32], train_b: &[i32], dt: f64) -> (Vec<f64>, Vec<f64>) {
    let n = train_a.len().min(train_b.len());
    if n < 2 {
        return (vec![], vec![]);
    }

    let mean_a: f64 = train_a[..n].iter().map(|&v| v as f64).sum::<f64>() / n as f64;
    let mean_b: f64 = train_b[..n].iter().map(|&v| v as f64).sum::<f64>() / n as f64;

    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(n);

    let mut buf_a: Vec<Complex<f64>> = train_a[..n]
        .iter()
        .map(|&v| Complex::new(v as f64 - mean_a, 0.0))
        .collect();
    let mut buf_b: Vec<Complex<f64>> = train_b[..n]
        .iter()
        .map(|&v| Complex::new(v as f64 - mean_b, 0.0))
        .collect();

    fft.process(&mut buf_a);
    fft.process(&mut buf_b);

    // Only positive frequencies (rfft equivalent): indices 0..=n/2
    let n_freqs = n / 2 + 1;
    let mut coh = Vec::with_capacity(n_freqs);
    let mut freqs = Vec::with_capacity(n_freqs);

    for i in 0..n_freqs {
        let fa = buf_a[i];
        let fb = buf_b[i];
        let pab = fa * fb.conj();
        let paa = fa.norm_sqr();
        let pbb = fb.norm_sqr();
        let denom = paa * pbb;
        if denom == 0.0 {
            coh.push(0.0);
        } else {
            coh.push(pab.norm_sqr() / denom);
        }
        freqs.push(i as f64 / (n as f64 * dt));
    }

    (coh, freqs)
}

/// Spike Time Tiling Coefficient (Cutts & Eglen 2014).
pub fn spike_time_tiling_coefficient(
    train_a: &[i32],
    train_b: &[i32],
    dt: f64,
    delta_ms: f64,
) -> f64 {
    let delta = delta_ms / 1000.0;
    let ta = spike_times(train_a, dt);
    let tb = spike_times(train_b, dt);
    let duration = train_a.len().max(train_b.len()) as f64 * dt;

    if ta.is_empty() || tb.is_empty() {
        return 0.0;
    }

    let pa = coincidence_fraction(&ta, &tb, delta);
    let pb = coincidence_fraction(&tb, &ta, delta);
    let ta_frac = tile_fraction(&ta, delta, duration);
    let tb_frac = tile_fraction(&tb, delta, duration);

    0.5 * (sttc_term(pa, tb_frac) + sttc_term(pb, ta_frac))
}

fn tile_fraction(times: &[f64], delta: f64, duration: f64) -> f64 {
    if times.is_empty() || duration <= 0.0 {
        return 0.0;
    }
    let mut intervals: Vec<(f64, f64)> = times.iter().map(|&t| (t - delta, t + delta)).collect();
    intervals.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    let mut merged = vec![intervals[0]];
    for &(lo, hi) in &intervals[1..] {
        let last = merged.last_mut().unwrap();
        if lo <= last.1 {
            last.1 = last.1.max(hi);
        } else {
            merged.push((lo, hi));
        }
    }

    let covered: f64 = merged
        .iter()
        .map(|&(lo, hi)| {
            let lo_c = lo.max(0.0);
            let hi_c = hi.min(duration);
            if hi_c > lo_c {
                hi_c - lo_c
            } else {
                0.0
            }
        })
        .sum();

    (covered / duration).min(1.0)
}

fn coincidence_fraction(times_ref: &[f64], times_target: &[f64], delta: f64) -> f64 {
    if times_ref.is_empty() {
        return 0.0;
    }
    let count = times_ref
        .iter()
        .filter(|&&t| times_target.iter().any(|&tt| (tt - t).abs() <= delta))
        .count();
    count as f64 / times_ref.len() as f64
}

fn sttc_term(p: f64, t: f64) -> f64 {
    if (1.0 - t).abs() < 1e-15 {
        return 0.0;
    }
    if (1.0 - p * t).abs() < 1e-15 {
        return 0.0;
    }
    (p - t) / (1.0 - p * t)
}

/// Spike count covariance matrix (de la Rocha et al. 2007).
pub fn covariance_matrix(trains: &[&[i32]], bin_size: usize) -> Vec<Vec<f64>> {
    let binned: Vec<Vec<i64>> = trains
        .iter()
        .map(|t| bin_spike_train(t, bin_size))
        .collect();
    let min_bins = binned.iter().map(|b| b.len()).min().unwrap_or(0);
    let n = trains.len();

    if n == 0 || min_bins == 0 {
        return vec![vec![]];
    }

    let mat: Vec<Vec<f64>> = binned
        .iter()
        .map(|b| b[..min_bins].iter().map(|&v| v as f64).collect())
        .collect();
    let means: Vec<f64> = mat
        .iter()
        .map(|row| row.iter().sum::<f64>() / min_bins as f64)
        .collect();

    if n == 1 {
        let var = mat[0].iter().map(|v| (v - means[0]).powi(2)).sum::<f64>()
            / (min_bins as f64 - 1.0).max(1.0);
        return vec![vec![var]];
    }

    let ddof = (min_bins as f64 - 1.0).max(1.0);
    let min_bins_f = min_bins as f64;
    let mut cov = vec![vec![0.0_f64; n]; n];
    cov.par_iter_mut().enumerate().for_each(|(i, row)| {
        for j in i..n {
            let dot = crate::simd::dot_f64_dispatch(&mat[i], &mat[j]);
            row[j] = (dot - min_bins_f * means[i] * means[j]) / ddof;
        }
    });
    // Mirror
    for i in 0..n {
        for j in (i + 1)..n {
            cov[j][i] = cov[i][j];
        }
    }
    cov
}

/// Autocorrelation time (seconds).
/// Integral of normalised autocorrelation until first zero crossing.
pub fn autocorrelation_time(binary_train: &[i32], dt: f64, max_lag_ms: f64) -> f64 {
    let max_lag = (max_lag_ms / (dt * 1000.0)) as usize;
    let n = binary_train.len();
    let mean: f64 = binary_train.iter().map(|&v| v as f64).sum::<f64>() / n as f64;
    let x: Vec<f64> = binary_train.iter().map(|&v| v as f64 - mean).collect();
    let var: f64 = x.iter().map(|v| v * v).sum();
    if var == 0.0 {
        return 0.0;
    }
    let mut tau = 0.0_f64;
    for lag in 1..max_lag.min(n) {
        let ac: f64 = (0..(n - lag)).map(|j| x[j] * x[j + lag]).sum::<f64>() / var;
        if ac < 0.0 {
            break;
        }
        tau += ac * dt;
    }
    tau
}

/// Noise correlation (Averbeck & Lee 2006).
/// Residuals after subtracting mean across neurons.
pub fn noise_correlation(trains: &[&[i32]], bin_size: usize) -> Vec<Vec<f64>> {
    let binned: Vec<Vec<i64>> = trains
        .iter()
        .map(|t| bin_spike_train(t, bin_size))
        .collect();
    let min_bins = binned.iter().map(|b| b.len()).min().unwrap_or(0);
    let n = trains.len();
    if n == 0 || min_bins == 0 {
        return vec![vec![]];
    }

    let mat: Vec<Vec<f64>> = binned
        .iter()
        .map(|b| b[..min_bins].iter().map(|&v| v as f64).collect())
        .collect();

    // Mean across neurons for each time bin
    let bin_means: Vec<f64> = (0..min_bins)
        .map(|k| mat.iter().map(|row| row[k]).sum::<f64>() / n as f64)
        .collect();

    // Residuals = mat - mean across time (axis=0)
    // Python: residuals = mat - mat.mean(axis=0, keepdims=True)
    let residuals: Vec<Vec<f64>> = mat
        .iter()
        .map(|row| {
            row.iter()
                .enumerate()
                .map(|(k, &v)| v - bin_means[k])
                .collect()
        })
        .collect();

    let mut corr = vec![vec![0.0_f64; n]; n];
    for i in 0..n {
        corr[i][i] = 1.0;
        let std_i = (residuals[i].iter().map(|v| v * v).sum::<f64>() / min_bins as f64).sqrt();
        for j in (i + 1)..n {
            let std_j = (residuals[j].iter().map(|v| v * v).sum::<f64>() / min_bins as f64).sqrt();
            if std_i > 0.0 && std_j > 0.0 {
                let r = residuals[i]
                    .iter()
                    .zip(residuals[j].iter())
                    .map(|(a, b)| a * b)
                    .sum::<f64>()
                    / min_bins as f64
                    / (std_i * std_j);
                corr[i][j] = r;
                corr[j][i] = r;
            }
        }
    }
    corr
}

/// Signal correlation (tuning similarity).
/// Pearson correlation of mean responses.
pub fn signal_correlation(trains: &[&[i32]], bin_size: usize) -> Vec<Vec<f64>> {
    let binned: Vec<Vec<i64>> = trains
        .iter()
        .map(|t| bin_spike_train(t, bin_size))
        .collect();
    let min_bins = binned.iter().map(|b| b.len()).min().unwrap_or(0);
    let n = trains.len();
    if n == 0 || min_bins == 0 {
        return vec![vec![]];
    }

    let mat: Vec<Vec<f64>> = binned
        .iter()
        .map(|b| b[..min_bins].iter().map(|&v| v as f64).collect())
        .collect();
    let means: Vec<f64> = mat
        .iter()
        .map(|row| row.iter().sum::<f64>() / min_bins as f64)
        .collect();
    let stds: Vec<f64> = mat
        .iter()
        .enumerate()
        .map(|(i, row)| {
            (row.iter().map(|v| (v - means[i]).powi(2)).sum::<f64>() / min_bins as f64).sqrt()
        })
        .collect();

    let mut corr = vec![vec![0.0_f64; n]; n];
    for i in 0..n {
        corr[i][i] = 1.0;
        for j in (i + 1)..n {
            if stds[i] > 0.0 && stds[j] > 0.0 {
                let c: f64 = (0..min_bins)
                    .map(|k| (mat[i][k] - means[i]) * (mat[j][k] - means[j]))
                    .sum::<f64>()
                    / min_bins as f64;
                let r = c / (stds[i] * stds[j]);
                corr[i][j] = r;
                corr[j][i] = r;
            }
        }
    }
    corr
}

/// Windowed spike count covariance (Kohn & Smith 2005).
pub fn spike_count_covariance(trains: &[&[i32]], window: usize) -> Vec<Vec<f64>> {
    covariance_matrix(trains, window)
}

/// Joint PSTH matrix (Aertsen et al. 1989).
/// Returns flattened n×n outer product of mean-subtracted binned counts.
pub fn joint_psth(train_a: &[i32], train_b: &[i32], bin_size: usize) -> (Vec<f64>, usize) {
    let ca_raw = bin_spike_train(train_a, bin_size);
    let cb_raw = bin_spike_train(train_b, bin_size);
    let n = ca_raw.len().min(cb_raw.len());
    if n == 0 {
        return (vec![], 0);
    }
    let mean_a = ca_raw[..n].iter().map(|&v| v as f64).sum::<f64>() / n as f64;
    let mean_b = cb_raw[..n].iter().map(|&v| v as f64).sum::<f64>() / n as f64;
    let ca: Vec<f64> = ca_raw[..n].iter().map(|&v| v as f64 - mean_a).collect();
    let cb: Vec<f64> = cb_raw[..n].iter().map(|&v| v as f64 - mean_b).collect();

    let mut result = Vec::with_capacity(n * n);
    for &ai in &ca {
        for &bj in &cb {
            result.push(ai * bj / n as f64);
        }
    }
    (result, n)
}

/// Coincidence index / kappa (Joris et al. 2006).
/// Corrects raw coincidence count for expected coincidences from rate.
pub fn coincidence_index(train_a: &[i32], train_b: &[i32], dt: f64, delta_ms: f64) -> f64 {
    let ta = spike_times(train_a, dt);
    let tb = spike_times(train_b, dt);
    if ta.is_empty() || tb.is_empty() {
        return 0.0;
    }
    let delta = delta_ms / 1000.0;
    let duration = train_a.len().max(train_b.len()) as f64 * dt;
    let mut raw_coinc = 0_usize;
    for &t in &ta {
        if tb.iter().any(|&tt| (tt - t).abs() <= delta) {
            raw_coinc += 1;
        }
    }
    let expected = if duration > 0.0 {
        2.0 * delta * ta.len() as f64 * tb.len() as f64 / duration
    } else {
        0.0
    };
    let norm = 0.5 * (ta.len() + tb.len()) as f64;
    if norm <= expected {
        return 0.0;
    }
    (raw_coinc as f64 - expected) / (norm - expected)
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

    // ── cross_correlation ───────────────────────────────────────────

    #[test]
    fn test_cross_correlation_identical() {
        let train = make_train(&[10, 30, 50, 70, 90], 100);
        let (cc, lags) = cross_correlation(&train, &train, 5.0, 0.001);
        // Peak at lag=0
        let zero_idx = lags.iter().position(|&l| l.abs() < 1e-10).unwrap();
        assert!(
            (cc[zero_idx] - 1.0).abs() < 1e-10,
            "autocorrelation peak should be 1.0"
        );
        // Symmetric
        for i in 0..cc.len() / 2 {
            assert!(
                (cc[i] - cc[cc.len() - 1 - i]).abs() < 1e-10,
                "autocorrelation should be symmetric"
            );
        }
    }

    #[test]
    fn test_cross_correlation_shifted() {
        let a = make_train(&[10, 30, 50], 100);
        let b = make_train(&[12, 32, 52], 100);
        let (cc, lags) = cross_correlation(&a, &b, 5.0, 0.001);
        // Peak should be near lag=+2ms (b lags a by 2 steps = 2ms)
        let peak_idx = cc
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0;
        assert!(
            (lags[peak_idx] - 2.0).abs() < 1.5,
            "peak lag should be near 2ms, got {}",
            lags[peak_idx]
        );
    }

    #[test]
    fn test_cross_correlation_empty() {
        let a = vec![0i32; 100];
        let b = make_train(&[10, 50], 100);
        let (cc, _) = cross_correlation(&a, &b, 5.0, 0.001);
        assert!(
            cc.iter().all(|&v| v == 0.0),
            "zero train → zero correlation"
        );
    }

    // ── pairwise_correlation ────────────────────────────────────────

    #[test]
    fn test_pairwise_correlation_identity() {
        let t1 = make_train(&[10, 30, 50], 100);
        let t2 = make_train(&[10, 30, 50], 100);
        let trains: Vec<&[i32]> = vec![&t1, &t2];
        let corr = pairwise_correlation(&trains, 0.001);
        assert!((corr[0][0] - 1.0).abs() < 1e-10);
        assert!((corr[0][1] - 1.0).abs() < 1e-10);
        assert!((corr[1][0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_pairwise_correlation_anticorrelated() {
        let t1 = make_train(&[0, 2, 4, 6, 8], 10);
        let t2 = make_train(&[1, 3, 5, 7, 9], 10);
        let trains: Vec<&[i32]> = vec![&t1, &t2];
        let corr = pairwise_correlation(&trains, 0.001);
        assert!(
            corr[0][1] < 0.0,
            "alternating trains should be negatively correlated"
        );
    }

    #[test]
    fn test_pairwise_correlation_empty() {
        let corr = pairwise_correlation(&[], 0.001);
        let expected: Vec<Vec<f64>> = vec![vec![]];
        assert_eq!(corr, expected);
    }

    // ── event_synchronization ───────────────────────────────────────

    #[test]
    fn test_event_sync_identical() {
        let train = make_train(&[10, 30, 50, 70], 100);
        let score = event_synchronization(&train, &train, 0.001, 5.0);
        // Only self-matches within tau: each spike matches itself → count=4, sqrt(4*4)=4 → 1.0
        assert!(
            (score - 1.0).abs() < 1e-10,
            "identical trains: count=4, sqrt(16)=4, score=1.0, got {}",
            score
        );
    }

    #[test]
    fn test_event_sync_no_overlap() {
        let a = make_train(&[10], 100);
        let b = make_train(&[90], 100);
        let score = event_synchronization(&a, &b, 0.001, 2.0);
        assert_eq!(score, 0.0, "far apart spikes → zero sync");
    }

    #[test]
    fn test_event_sync_empty() {
        let a = vec![0i32; 100];
        let b = make_train(&[50], 100);
        assert_eq!(event_synchronization(&a, &b, 0.001, 5.0), 0.0);
    }

    // ── spike_train_coherence ───────────────────────────────────────

    #[test]
    fn test_coherence_identical() {
        let train = make_train(&[10, 30, 50, 70, 90], 128);
        let (coh, freqs) = spike_train_coherence(&train, &train, 0.001);
        assert!(!coh.is_empty());
        assert_eq!(coh.len(), freqs.len());
        // Self-coherence should be 1.0 at non-DC frequencies (DC is 0/0 after mean subtraction)
        for (i, &c) in coh.iter().enumerate() {
            if i == 0 {
                continue; // DC bin is zero after mean subtraction
            }
            assert!(
                (c - 1.0).abs() < 1e-8,
                "self-coherence at freq idx {i} should be 1.0, got {c}"
            );
        }
    }

    #[test]
    fn test_coherence_short() {
        let a = vec![1i32];
        let b = vec![0i32];
        let (coh, _) = spike_train_coherence(&a, &b, 0.001);
        assert!(coh.is_empty(), "n<2 → empty");
    }

    // ── spike_time_tiling_coefficient ───────────────────────────────

    #[test]
    fn test_sttc_identical() {
        let train = make_train(&[10, 30, 50, 70, 90], 100);
        let sttc = spike_time_tiling_coefficient(&train, &train, 0.001, 5.0);
        assert!(sttc > 0.8, "identical trains → high STTC, got {sttc}");
    }

    #[test]
    fn test_sttc_no_overlap() {
        let a = make_train(&[5], 1000);
        let b = make_train(&[995], 1000);
        let sttc = spike_time_tiling_coefficient(&a, &b, 0.001, 1.0);
        assert!(sttc < 0.1, "far apart spikes → low STTC, got {sttc}");
    }

    #[test]
    fn test_sttc_empty() {
        let a = vec![0i32; 100];
        let b = make_train(&[50], 100);
        assert_eq!(spike_time_tiling_coefficient(&a, &b, 0.001, 5.0), 0.0);
    }

    // ── covariance_matrix ───────────────────────────────────────────

    #[test]
    fn test_covariance_identical() {
        let train = make_train(&[0, 1, 5, 6, 10, 11, 15, 16, 20, 21], 25);
        let trains: Vec<&[i32]> = vec![&train, &train];
        let cov = covariance_matrix(&trains, 5);
        assert!(
            (cov[0][0] - cov[0][1]).abs() < 1e-10,
            "identical trains → equal diagonal and off-diagonal"
        );
    }

    #[test]
    fn test_covariance_single() {
        let train = make_train(&[0, 1, 2, 5, 6, 10, 11, 12, 13, 14], 20);
        let trains: Vec<&[i32]> = vec![&train];
        let cov = covariance_matrix(&trains, 5);
        assert_eq!(cov.len(), 1);
        assert!(cov[0][0] > 0.0, "non-constant train → positive variance");
    }

    // ── autocorrelation_time ────────────────────────────────────────

    #[test]
    fn test_autocorr_time_bursty() {
        // Burst pattern: consecutive spikes have positive lag-1 autocorrelation
        let train = make_train(&[0, 1, 2, 10, 11, 12, 20, 21, 22, 30, 31, 32], 40);
        let tau = autocorrelation_time(&train, 0.001, 50.0);
        assert!(
            tau > 0.0,
            "bursty train should have positive autocorrelation time, got {tau}"
        );
    }

    #[test]
    fn test_autocorr_time_silent() {
        let train = vec![0i32; 100];
        assert_eq!(autocorrelation_time(&train, 0.001, 50.0), 0.0);
    }

    // ── noise_correlation ───────────────────────────────────────────

    #[test]
    fn test_noise_corr_identical() {
        let t1 = make_train(&[5, 15, 25, 35, 45], 50);
        let t2 = t1.clone();
        let trains: Vec<&[i32]> = vec![&t1, &t2];
        let corr = noise_correlation(&trains, 10);
        assert!((corr[0][0] - 1.0).abs() < 1e-10);
        // Identical trains: residuals are zero → correlation undefined but set to 1 if std>0
    }

    #[test]
    fn test_noise_corr_diagonal() {
        let t1 = make_train(&[2, 12, 22], 30);
        let t2 = make_train(&[7, 17, 27], 30);
        let trains: Vec<&[i32]> = vec![&t1, &t2];
        let corr = noise_correlation(&trains, 10);
        assert!((corr[0][0] - 1.0).abs() < 1e-10);
        assert!((corr[1][1] - 1.0).abs() < 1e-10);
    }

    // ── signal_correlation ──────────────────────────────────────────

    #[test]
    fn test_signal_corr_identical() {
        // bins: [1, 3, 0] — non-constant so std > 0
        let t1 = make_train(&[5, 10, 11, 12], 30);
        let t2 = t1.clone();
        let trains: Vec<&[i32]> = vec![&t1, &t2];
        let corr = signal_correlation(&trains, 10);
        assert!(
            (corr[0][1] - 1.0).abs() < 1e-10,
            "identical trains → r=1.0, got {}",
            corr[0][1]
        );
    }

    // ── spike_count_covariance ──────────────────────────────────────

    #[test]
    fn test_spike_count_cov_delegates() {
        let t1 = make_train(&[0, 1, 5, 6, 10, 11], 15);
        let trains: Vec<&[i32]> = vec![&t1];
        let cov1 = covariance_matrix(&trains, 5);
        let cov2 = spike_count_covariance(&trains, 5);
        assert_eq!(cov1, cov2);
    }

    // ── joint_psth ──────────────────────────────────────────────────

    #[test]
    fn test_joint_psth_shape() {
        let a = make_train(&[0, 1, 5, 6, 10, 11, 15, 16, 20, 21], 25);
        let b = make_train(&[2, 3, 7, 8, 12, 13, 17, 18, 22, 23], 25);
        let (result, n) = joint_psth(&a, &b, 5);
        assert_eq!(n, 5);
        assert_eq!(result.len(), 25);
    }

    #[test]
    fn test_joint_psth_symmetry() {
        let train = make_train(&[0, 1, 5, 6, 10, 11, 15, 16, 20, 21], 25);
        let (result, n) = joint_psth(&train, &train, 5);
        // Outer product of x with itself is symmetric
        for i in 0..n {
            for j in 0..n {
                assert!(
                    (result[i * n + j] - result[j * n + i]).abs() < 1e-10,
                    "JPSTH of identical trains should be symmetric"
                );
            }
        }
    }

    // ── coincidence_index ───────────────────────────────────────────

    #[test]
    fn test_coincidence_index_identical() {
        let train = make_train(&[10, 30, 50, 70, 90], 100);
        let ci = coincidence_index(&train, &train, 0.001, 2.0);
        assert!(
            ci > 0.5,
            "identical trains → high coincidence index, got {ci}"
        );
    }

    #[test]
    fn test_coincidence_index_no_overlap() {
        let a = make_train(&[5], 1000);
        let b = make_train(&[995], 1000);
        let ci = coincidence_index(&a, &b, 0.001, 1.0);
        assert!(ci <= 0.0, "far apart → zero or negative kappa, got {ci}");
    }

    #[test]
    fn test_coincidence_index_empty() {
        let a = vec![0i32; 100];
        let b = make_train(&[50], 100);
        assert_eq!(coincidence_index(&a, &b, 0.001, 2.0), 0.0);
    }

    // ── helpers ─────────────────────────────────────────────────────

    #[test]
    fn test_tile_fraction_single_spike() {
        let times = vec![0.05];
        let frac = tile_fraction(&times, 0.005, 0.1);
        // Window: [0.045, 0.055] → 0.01 / 0.1 = 0.1
        assert!((frac - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_tile_fraction_overlapping() {
        let times = vec![0.05, 0.052];
        let frac = tile_fraction(&times, 0.005, 0.1);
        // Windows [0.045, 0.055] and [0.047, 0.057] merge to [0.045, 0.057] → 0.012
        assert!((frac - 0.12).abs() < 1e-10);
    }

    #[test]
    fn test_sttc_term_edge_cases() {
        assert_eq!(sttc_term(0.5, 1.0), 0.0); // t=1
        assert_eq!(sttc_term(0.0, 0.0), 0.0); // p=t=0
    }
}
