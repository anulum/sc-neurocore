// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Point process models and hazard functions

use super::basic;

/// Conditional intensity function estimate (Hz). Brown et al. 2004.
///
/// Moving-window MLE of the Poisson rate at each time step.
pub fn conditional_intensity(binary_train: &[i32], dt: f64, window_ms: f64) -> Vec<f64> {
    let n = binary_train.len();
    if n == 0 {
        return vec![];
    }
    let w = (window_ms / (dt * 1000.0)).round().max(1.0) as usize;
    let kernel_sum_inv = 1.0 / (w as f64 * dt);

    // "same" convolution: centre the kernel
    let half = w / 2;
    let mut result = vec![0.0f64; n];
    for i in 0..n {
        let lo = i.saturating_sub(half);
        let hi = (i + w - half).min(n);
        let mut sum = 0.0;
        for j in lo..hi {
            sum += binary_train[j] as f64;
        }
        result[i] = sum * kernel_sum_inv;
    }
    result
}

/// ISI hazard function h(t) = f(t) / S(t). Tuckwell 1988.
///
/// Returns `(hazard, bin_centres)`.
pub fn isi_hazard_function(
    binary_train: &[i32],
    dt: f64,
    bins: usize,
) -> (Vec<f64>, Vec<f64>) {
    let intervals = basic::isi(binary_train, dt);
    if intervals.len() < 5 {
        return (vec![], vec![]);
    }
    let (hist, edges) = histogram(&intervals, bins);
    let bin_width = edges[1] - edges[0];
    let n = intervals.len() as f64;
    let centres: Vec<f64> = (0..bins).map(|k| (edges[k] + edges[k + 1]) / 2.0).collect();
    let pdf: Vec<f64> = hist.iter().map(|&c| c as f64 / (n * bin_width)).collect();
    let mut cum = 0.0;
    let mut survivor = vec![0.0; bins];
    for k in 0..bins {
        cum += pdf[k] * bin_width;
        survivor[k] = (1.0 - cum).max(1e-30);
    }
    let hazard: Vec<f64> = (0..bins).map(|k| pdf[k] / survivor[k]).collect();
    (hazard, centres)
}

/// ISI survivor function S(t) = P(ISI > t). Tuckwell 1988.
///
/// Returns `(survivor, bin_centres)`.
pub fn isi_survivor_function(
    binary_train: &[i32],
    dt: f64,
    bins: usize,
) -> (Vec<f64>, Vec<f64>) {
    let intervals = basic::isi(binary_train, dt);
    if intervals.len() < 2 {
        return (vec![], vec![]);
    }
    let mut sorted = intervals.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = sorted.len() as f64;
    let max_isi = sorted[sorted.len() - 1];
    let edges: Vec<f64> = (0..=bins).map(|k| k as f64 * max_isi / bins as f64).collect();
    let centres: Vec<f64> = (0..bins).map(|k| (edges[k] + edges[k + 1]) / 2.0).collect();
    let survivor: Vec<f64> = centres
        .iter()
        .map(|&t| sorted.iter().filter(|&&v| v > t).count() as f64 / n)
        .collect();
    (survivor, centres)
}

/// Renewal density from ISI distribution. Cox 1962.
///
/// Returns `(density, bin_centres)`. Density normalised by mean rate.
pub fn renewal_density(binary_train: &[i32], dt: f64, bins: usize) -> (Vec<f64>, Vec<f64>) {
    let intervals = basic::isi(binary_train, dt);
    if intervals.len() < 5 {
        return (vec![], vec![]);
    }
    let (hist, edges) = histogram(&intervals, bins);
    let bin_width = edges[1] - edges[0];
    let n = intervals.len() as f64;
    let centres: Vec<f64> = (0..bins).map(|k| (edges[k] + edges[k + 1]) / 2.0).collect();
    let pdf: Vec<f64> = hist.iter().map(|&c| c as f64 / (n * bin_width)).collect();
    let mean_isi: f64 = intervals.iter().sum::<f64>() / n;
    let mean_rate = if mean_isi > 0.0 { 1.0 / mean_isi } else { 1.0 };
    let density: Vec<f64> = pdf.iter().map(|&p| p / mean_rate).collect();
    (density, centres)
}

/// Simple histogram: returns (counts, edges) with `bins` equal-width buckets.
fn histogram(data: &[f64], bins: usize) -> (Vec<usize>, Vec<f64>) {
    let min = data.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let range = if (max - min).abs() < 1e-30 {
        1.0
    } else {
        max - min
    };
    let edges: Vec<f64> = (0..=bins)
        .map(|k| min + k as f64 * range / bins as f64)
        .collect();
    let mut counts = vec![0usize; bins];
    for &v in data {
        let mut k = ((v - min) / range * bins as f64) as usize;
        if k >= bins {
            k = bins - 1;
        }
        counts[k] += 1;
    }
    (counts, edges)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_train() -> Vec<i32> {
        // Regular spiking every 10 steps
        let mut t = vec![0i32; 200];
        for i in (0..200).step_by(10) {
            t[i] = 1;
        }
        t
    }

    #[test]
    fn test_conditional_intensity_shape() {
        let train = make_train();
        let ci = conditional_intensity(&train, 0.001, 50.0);
        assert_eq!(ci.len(), 200);
        // Should be roughly 100 Hz (1 spike per 10ms = 100 Hz)
        let mid = ci[100];
        assert!((mid - 100.0).abs() < 50.0, "CI midpoint {mid} not near 100 Hz");
    }

    #[test]
    fn test_conditional_intensity_empty() {
        assert!(conditional_intensity(&[], 0.001, 10.0).is_empty());
    }

    #[test]
    fn test_conditional_intensity_single() {
        let ci = conditional_intensity(&[1], 0.001, 10.0);
        assert_eq!(ci.len(), 1);
    }

    #[test]
    fn test_isi_hazard_basic() {
        let train = make_train();
        let (hazard, centres) = isi_hazard_function(&train, 0.001, 20);
        assert!(!hazard.is_empty());
        assert_eq!(hazard.len(), centres.len());
        // All hazard values should be non-negative
        assert!(hazard.iter().all(|&h| h >= 0.0));
    }

    #[test]
    fn test_isi_hazard_few_spikes() {
        let train = vec![0, 1, 0, 0, 0, 1, 0]; // 1 ISI
        let (h, c) = isi_hazard_function(&train, 0.001, 10);
        assert!(h.is_empty());
        assert!(c.is_empty());
    }

    #[test]
    fn test_isi_survivor_basic() {
        let train = make_train();
        let (surv, centres) = isi_survivor_function(&train, 0.001, 20);
        assert!(!surv.is_empty());
        assert_eq!(surv.len(), centres.len());
        // Survivor should start near 1 and decrease
        assert!(surv[0] >= surv[surv.len() - 1]);
    }

    #[test]
    fn test_isi_survivor_few_spikes() {
        let train = vec![1, 0];
        let (s, c) = isi_survivor_function(&train, 0.001, 10);
        assert!(s.is_empty());
        assert!(c.is_empty());
    }

    #[test]
    fn test_renewal_density_basic() {
        let train = make_train();
        let (dens, centres) = renewal_density(&train, 0.001, 20);
        assert!(!dens.is_empty());
        assert_eq!(dens.len(), centres.len());
        assert!(dens.iter().all(|&d| d >= 0.0));
    }

    #[test]
    fn test_renewal_density_few_spikes() {
        let (d, c) = renewal_density(&[0, 1, 0], 0.001, 10);
        assert!(d.is_empty());
        assert!(c.is_empty());
    }

    #[test]
    fn test_histogram_helper() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let (counts, edges) = histogram(&data, 4);
        assert_eq!(counts.len(), 4);
        assert_eq!(edges.len(), 5);
        let total: usize = counts.iter().sum();
        assert_eq!(total, 5);
    }

    #[test]
    fn test_regular_spike_isi_uniformity() {
        let train = make_train();
        // All ISIs should be 0.01s for regular train
        let intervals = basic::isi(&train, 0.001);
        assert!(intervals.iter().all(|&i| (i - 0.01).abs() < 1e-12));
    }
}
