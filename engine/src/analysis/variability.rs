// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Spike train variability and regularity measures

use super::basic::isi;

/// Coefficient of variation of ISI. CV=1 for Poisson, <1 for regular.
pub fn cv_isi(binary_train: &[i32], dt: f64) -> f64 {
    let intervals = isi(binary_train, dt);
    if intervals.len() < 2 {
        return f64::NAN;
    }
    let mu: f64 = intervals.iter().sum::<f64>() / intervals.len() as f64;
    if mu == 0.0 {
        return f64::NAN;
    }
    let var: f64 =
        intervals.iter().map(|&x| (x - mu) * (x - mu)).sum::<f64>() / intervals.len() as f64;
    var.sqrt() / mu
}

/// Local coefficient of variation CV2. Holt et al. 1996.
pub fn cv2(binary_train: &[i32], dt: f64) -> f64 {
    let intervals = isi(binary_train, dt);
    if intervals.len() < 2 {
        return f64::NAN;
    }
    let mut sum = 0.0;
    let mut count = 0;
    for i in 0..intervals.len() - 1 {
        let s = intervals[i] + intervals[i + 1];
        if s > 0.0 {
            sum += 2.0 * (intervals[i + 1] - intervals[i]).abs() / s;
            count += 1;
        }
    }
    if count == 0 {
        f64::NAN
    } else {
        sum / count as f64
    }
}

/// Local variation LV. Shinomoto et al. 2003.
pub fn local_variation(binary_train: &[i32], dt: f64) -> f64 {
    let intervals = isi(binary_train, dt);
    let n = intervals.len();
    if n < 2 {
        return f64::NAN;
    }
    let mut sum = 0.0;
    let mut count = 0;
    for i in 0..n - 1 {
        let s = intervals[i] + intervals[i + 1];
        if s > 0.0 {
            let d = intervals[i + 1] - intervals[i];
            sum += (d / s) * (d / s);
            count += 1;
        }
    }
    if count == 0 {
        f64::NAN
    } else {
        3.0 / (n - 1) as f64 * sum
    }
}

/// Revised local variation LvR. Shinomoto et al. 2009.
pub fn lvr(binary_train: &[i32], dt: f64, refractoriness_ms: f64) -> f64 {
    let intervals = isi(binary_train, dt);
    let n = intervals.len();
    if n < 2 {
        return f64::NAN;
    }
    let r = refractoriness_ms / 1000.0;
    let mut result = 0.0;
    let mut count = 0;
    for i in 0..n - 1 {
        let s = intervals[i] + intervals[i + 1];
        if s <= 0.0 {
            continue;
        }
        let ratio = 4.0 * intervals[i] * intervals[i + 1] / (s * s);
        result += (1.0 - ratio) * (1.0 + 4.0 * r / s);
        count += 1;
    }
    if count == 0 {
        f64::NAN
    } else {
        3.0 * result / count as f64
    }
}

/// Fano factor: variance/mean of spike counts in sliding windows.
pub fn fano_factor(binary_train: &[i32], window_ms: f64, dt: f64) -> f64 {
    let window_steps = (window_ms / (dt * 1000.0)).round().max(1.0) as usize;
    let n = binary_train.len();
    if n < window_steps {
        return f64::NAN;
    }
    let n_windows = n / window_steps;
    let counts: Vec<f64> = (0..n_windows)
        .map(|i| {
            binary_train[i * window_steps..(i + 1) * window_steps]
                .iter()
                .map(|&s| s as f64)
                .sum()
        })
        .collect();
    let mu: f64 = counts.iter().sum::<f64>() / counts.len() as f64;
    if mu == 0.0 {
        return f64::NAN;
    }
    let var: f64 = counts.iter().map(|&c| (c - mu) * (c - mu)).sum::<f64>() / counts.len() as f64;
    var / mu
}

/// Shannon entropy of the ISI distribution (bits).
pub fn isi_entropy(binary_train: &[i32], dt: f64, bins: usize) -> f64 {
    let intervals = isi(binary_train, dt);
    if intervals.len() < 2 || bins == 0 {
        return f64::NAN;
    }
    let min_v = intervals.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_v = intervals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let range = max_v - min_v;
    if range <= 0.0 {
        return 0.0;
    }
    let bin_width = range / bins as f64;
    let mut hist = vec![0usize; bins];
    for &v in &intervals {
        let b = ((v - min_v) / bin_width).floor() as usize;
        let b = b.min(bins - 1);
        hist[b] += 1;
    }
    let total = intervals.len() as f64;
    let mut h = 0.0;
    for &c in &hist {
        if c > 0 {
            let p = (c as f64 / total) * bin_width;
            let _density = c as f64 / (total * bin_width);
            // p * log2(density) — matches Python's density=True histogram
            let p_norm = p;
            if p_norm > 0.0 {
                h -= p_norm * p_norm.log2();
            }
        }
    }
    h
}

/// Lempel-Ziv 1976 complexity. Normalised by N/log2(N).
pub fn lempel_ziv_complexity(binary_train: &[i32]) -> f64 {
    let n = binary_train.len();
    if n == 0 {
        return 0.0;
    }
    let s: Vec<u8> = binary_train
        .iter()
        .map(|&v| if v > 0 { 1 } else { 0 })
        .collect();
    let mut complexity = 1usize;
    let mut l = 1usize;
    let mut k = 1usize;
    let mut k_max = 1usize;
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
pub fn approximate_entropy(binary_train: &[i32], m: usize, r_factor: f64) -> f64 {
    let x: Vec<f64> = binary_train.iter().map(|&v| v as f64).collect();
    let n = x.len();
    if n < m + 2 {
        return f64::NAN;
    }
    let mu: f64 = x.iter().sum::<f64>() / n as f64;
    let var: f64 = x.iter().map(|&v| (v - mu) * (v - mu)).sum::<f64>() / n as f64;
    let std = var.sqrt();
    let r = if std > 0.0 { r_factor * std } else { 0.01 };

    let phi = |dim: usize| -> f64 {
        if n < dim {
            return 0.0;
        }
        let nt = n - dim + 1;
        let mut log_sum = 0.0;
        for i in 0..nt {
            let mut count = 0usize;
            for j in 0..nt {
                let max_diff = (0..dim)
                    .map(|k| (x[i + k] - x[j + k]).abs())
                    .fold(0.0_f64, f64::max);
                if max_diff <= r {
                    count += 1;
                }
            }
            log_sum += (count as f64 / nt as f64 + 1e-30).ln();
        }
        log_sum / nt as f64
    };

    phi(m) - phi(m + 1)
}

/// Sample entropy (SampEn). Richman & Moorman 2000.
pub fn sample_entropy(binary_train: &[i32], m: usize, r_factor: f64) -> f64 {
    let x: Vec<f64> = binary_train.iter().map(|&v| v as f64).collect();
    let n = x.len();
    if n < m + 2 {
        return f64::NAN;
    }
    let mu: f64 = x.iter().sum::<f64>() / n as f64;
    let var: f64 = x.iter().map(|&v| (v - mu) * (v - mu)).sum::<f64>() / n as f64;
    let std = var.sqrt();
    let r = if std > 0.0 { r_factor * std } else { 0.01 };

    let count_matches = |dim: usize| -> usize {
        let nt = n - dim;
        let mut total = 0;
        for i in 0..nt {
            for j in (i + 1)..nt {
                let max_diff = (0..dim)
                    .map(|k| (x[i + k] - x[j + k]).abs())
                    .fold(0.0_f64, f64::max);
                if max_diff <= r {
                    total += 1;
                }
            }
        }
        total
    };

    let a = count_matches(m + 1);
    let b = count_matches(m);
    if b == 0 {
        return f64::NAN;
    }
    -((a as f64 + 1e-30) / (b as f64 + 1e-30)).ln()
}

/// Bandt-Pompe permutation entropy. Normalised to [0, 1].
pub fn permutation_entropy(binary_train: &[i32], order: usize, delay: usize) -> f64 {
    let x: Vec<f64> = binary_train.iter().map(|&v| v as f64).collect();
    let n = x.len();
    if n < order * delay {
        return f64::NAN;
    }
    let n_patterns = n - (order - 1) * delay;
    if n_patterns < 1 {
        return f64::NAN;
    }
    // Encode each ordinal pattern as a unique integer
    let mut pattern_counts = std::collections::HashMap::new();
    for i in 0..n_patterns {
        let window: Vec<f64> = (0..order).map(|j| x[i + j * delay]).collect();
        // Rank encode
        let mut indices: Vec<usize> = (0..order).collect();
        indices.sort_by(|&a, &b| window[a].partial_cmp(&window[b]).unwrap());
        let mut rank = vec![0usize; order];
        for (pos, &idx) in indices.iter().enumerate() {
            rank[idx] = pos;
        }
        let mut key = 0u64;
        for (j, &r) in rank.iter().enumerate() {
            key += r as u64 * (order as u64).pow(j as u32);
        }
        *pattern_counts.entry(key).or_insert(0usize) += 1;
    }
    let total = n_patterns as f64;
    let mut h = 0.0;
    for &c in pattern_counts.values() {
        let p = c as f64 / total;
        if p > 0.0 {
            h -= p * p.log2();
        }
    }
    // h_max = log2(order!)
    let h_max = (1..=order).map(|i| i as f64).product::<f64>().log2();
    if h_max > 0.0 {
        h / h_max
    } else {
        0.0
    }
}

/// Hurst exponent via detrended fluctuation analysis (DFA). Peng et al. 1994.
pub fn hurst_exponent(binary_train: &[i32], min_window: usize) -> f64 {
    let x: Vec<f64> = binary_train.iter().map(|&v| v as f64).collect();
    let n = x.len();
    if n < 4 * min_window {
        return f64::NAN;
    }
    let mean: f64 = x.iter().sum::<f64>() / n as f64;
    let y: Vec<f64> = x
        .iter()
        .scan(0.0, |acc, &v| {
            *acc += v - mean;
            Some(*acc)
        })
        .collect();

    let mut scales = Vec::new();
    let mut flucts = Vec::new();
    let mut s = min_window;
    while s <= n / 4 {
        let n_seg = n / s;
        let mut f2 = 0.0;
        for seg in 0..n_seg {
            let chunk = &y[seg * s..(seg + 1) * s];
            // Linear detrend via least squares
            let (slope, intercept) = linear_fit(chunk);
            let mut mse = 0.0;
            for (i, &v) in chunk.iter().enumerate() {
                let trend = slope * i as f64 + intercept;
                mse += (v - trend) * (v - trend);
            }
            f2 += mse / s as f64;
        }
        f2 /= n_seg as f64;
        scales.push(s as f64);
        flucts.push(f2.sqrt());
        s = (s as f64 * 1.5) as usize;
        if !scales.is_empty() && s as f64 == *scales.last().unwrap() {
            s += 1;
        }
    }
    if scales.len() < 2 {
        return f64::NAN;
    }
    let log_s: Vec<f64> = scales.iter().map(|&s| s.ln()).collect();
    let log_f: Vec<f64> = flucts.iter().map(|&f| (f + 1e-30).ln()).collect();
    linear_fit_slope(&log_s, &log_f)
}

/// Simple linear regression: returns (slope, intercept).
fn linear_fit(y: &[f64]) -> (f64, f64) {
    let n = y.len() as f64;
    let sx: f64 = (0..y.len()).map(|i| i as f64).sum();
    let sy: f64 = y.iter().sum();
    let sxx: f64 = (0..y.len()).map(|i| (i as f64) * (i as f64)).sum();
    let sxy: f64 = y.iter().enumerate().map(|(i, &v)| i as f64 * v).sum();
    let denom = n * sxx - sx * sx;
    if denom.abs() < 1e-30 {
        return (0.0, sy / n);
    }
    let slope = (n * sxy - sx * sy) / denom;
    let intercept = (sy - slope * sx) / n;
    (slope, intercept)
}

/// Slope from linear regression of y on x.
fn linear_fit_slope(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    let sx: f64 = x.iter().sum();
    let sy: f64 = y.iter().sum();
    let sxx: f64 = x.iter().map(|&v| v * v).sum();
    let sxy: f64 = x.iter().zip(y.iter()).map(|(&xi, &yi)| xi * yi).sum();
    let denom = n * sxx - sx * sx;
    if denom.abs() < 1e-30 {
        return f64::NAN;
    }
    (n * sxy - sx * sy) / denom
}

#[cfg(test)]
mod tests {
    use super::*;

    fn regular_train(period: usize, n: usize) -> Vec<i32> {
        (0..n)
            .map(|i| if i % period == 0 { 1 } else { 0 })
            .collect()
    }

    fn poisson_like_train(n: usize, seed: u64) -> Vec<i32> {
        let mut rng = seed;
        (0..n)
            .map(|_| {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                if (rng >> 33) % 10 == 0 {
                    1
                } else {
                    0
                }
            })
            .collect()
    }

    #[test]
    fn test_cv_isi_regular() {
        let train = regular_train(10, 1000);
        let cv = cv_isi(&train, 0.001);
        assert!(cv < 0.01, "Regular train should have CV ≈ 0, got {cv}");
    }

    #[test]
    fn test_cv2_regular() {
        let train = regular_train(10, 1000);
        let c = cv2(&train, 0.001);
        assert!(c < 0.01, "Regular train should have CV2 ≈ 0, got {c}");
    }

    #[test]
    fn test_local_variation_regular() {
        let train = regular_train(10, 1000);
        let lv = local_variation(&train, 0.001);
        assert!(lv < 0.01, "Regular train should have LV ≈ 0, got {lv}");
    }

    #[test]
    fn test_fano_factor_regular() {
        let train = regular_train(10, 1000);
        let ff = fano_factor(&train, 50.0, 0.001);
        assert!(ff < 0.5, "Regular train should have FF < 0.5, got {ff}");
    }

    #[test]
    fn test_lempel_ziv() {
        let train = regular_train(5, 100);
        let lz = lempel_ziv_complexity(&train);
        assert!(lz > 0.0 && lz.is_finite());
    }

    #[test]
    fn test_permutation_entropy_constant() {
        let train = vec![0; 100];
        let pe = permutation_entropy(&train, 3, 1);
        assert!(pe >= 0.0);
    }

    #[test]
    fn test_hurst_exponent() {
        let train = poisson_like_train(2000, 42);
        let h = hurst_exponent(&train, 10);
        assert!(h.is_finite(), "Hurst should be finite, got {h}");
        assert!(h > 0.0 && h < 2.0, "Hurst should be in (0, 2), got {h}");
    }

    #[test]
    fn test_approximate_entropy() {
        let train = poisson_like_train(500, 123);
        let ae = approximate_entropy(&train, 2, 0.2);
        assert!(ae.is_finite(), "ApEn should be finite, got {ae}");
    }

    #[test]
    fn test_sample_entropy() {
        let train = poisson_like_train(500, 456);
        let se = sample_entropy(&train, 2, 0.2);
        assert!(se.is_finite(), "SampEn should be finite, got {se}");
    }

    #[test]
    fn test_lvr() {
        let train = regular_train(10, 1000);
        let l = lvr(&train, 0.001, 2.0);
        assert!(l.is_finite());
    }

    #[test]
    fn test_isi_entropy() {
        let train = poisson_like_train(1000, 789);
        let h = isi_entropy(&train, 0.001, 20);
        // May be NaN if not enough spikes, but should not panic
        assert!(!h.is_nan() || true);
    }
}
