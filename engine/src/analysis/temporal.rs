// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Temporal pattern detection: bursts, latency, onset, change points

use super::basic::{bin_spike_train, spike_times};

/// Detect bursts: consecutive spikes with ISI < max_isi_ms.
/// Returns Vec of (start_time_s, end_time_s, spike_count).
pub fn burst_detection(
    binary_train: &[i32],
    dt: f64,
    max_isi_ms: f64,
    min_spikes: usize,
) -> Vec<(f64, f64, usize)> {
    let times = spike_times(binary_train, dt);
    if times.len() < min_spikes {
        return vec![];
    }
    let max_isi = max_isi_ms / 1000.0;
    let intervals: Vec<f64> = times.windows(2).map(|w| w[1] - w[0]).collect();
    let mut bursts = Vec::new();
    let mut i = 0;
    while i < intervals.len() {
        if intervals[i] < max_isi {
            let start = i;
            while i < intervals.len() && intervals[i] < max_isi {
                i += 1;
            }
            let n_spikes = i - start + 1;
            if n_spikes >= min_spikes {
                bursts.push((times[start], times[i], n_spikes));
            }
        } else {
            i += 1;
        }
    }
    bursts
}

/// Time to first spike (seconds). Returns NaN if no spike.
pub fn first_spike_latency(binary_train: &[i32], dt: f64) -> f64 {
    for (i, &v) in binary_train.iter().enumerate() {
        if v > 0 {
            return i as f64 * dt;
        }
    }
    f64::NAN
}

/// Detect response onset as first bin exceeding baseline + threshold_sigma * std.
/// Returns onset time (seconds), or NaN.
pub fn response_onset(
    binary_train: &[i32],
    baseline_steps: usize,
    dt: f64,
    threshold_sigma: f64,
) -> f64 {
    if binary_train.len() <= baseline_steps {
        return f64::NAN;
    }
    let baseline = &binary_train[..baseline_steps];
    let mean: f64 = baseline.iter().map(|&v| v as f64).sum::<f64>() / baseline_steps as f64;
    let std: f64 = (baseline
        .iter()
        .map(|&v| (v as f64 - mean).powi(2))
        .sum::<f64>()
        / baseline_steps as f64)
        .sqrt()
        .max(1e-10);
    let threshold = mean + threshold_sigma * std;
    for (i, &v) in binary_train[baseline_steps..].iter().enumerate() {
        if v as f64 > threshold {
            return (baseline_steps + i) as f64 * dt;
        }
    }
    f64::NAN
}

/// CUSUM-based change point detection (Page 1954).
/// Returns list of bin indices where significant rate changes occur.
pub fn change_point_detection(binary_train: &[i32], bin_size: usize, threshold: f64) -> Vec<usize> {
    let counts: Vec<f64> = bin_spike_train(binary_train, bin_size)
        .iter()
        .map(|&v| v as f64)
        .collect();
    let n = counts.len();
    if n < 5 {
        return vec![];
    }
    let mean_rate: f64 = counts.iter().sum::<f64>() / n as f64;
    let std_rate: f64 =
        (counts.iter().map(|v| (v - mean_rate).powi(2)).sum::<f64>() / n as f64).sqrt();
    if std_rate < 1e-10 {
        return vec![];
    }
    let mut cusum_pos = 0.0_f64;
    let mut cusum_neg = 0.0_f64;
    let mut change_points = Vec::new();
    for i in 1..n {
        let z = (counts[i] - mean_rate) / std_rate;
        cusum_pos = (cusum_pos + z).max(0.0);
        cusum_neg = (cusum_neg - z).max(0.0);
        if cusum_pos > threshold || cusum_neg > threshold {
            change_points.push(i);
            cusum_pos = 0.0;
            cusum_neg = 0.0;
        }
    }
    change_points
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

    #[test]
    fn test_burst_detection_regular() {
        // Spikes every 5ms — ISI = 5ms, max_isi = 10ms → single burst
        let train = make_train(&[10, 15, 20, 25, 30], 100);
        let bursts = burst_detection(&train, 0.001, 10.0, 3);
        assert_eq!(bursts.len(), 1);
        assert_eq!(bursts[0].2, 5, "5 spikes in burst");
    }

    #[test]
    fn test_burst_detection_no_burst() {
        // Spikes every 50ms — ISI = 50ms > 10ms
        let train = make_train(&[10, 60, 110, 160], 200);
        let bursts = burst_detection(&train, 0.001, 10.0, 3);
        assert!(bursts.is_empty());
    }

    #[test]
    fn test_burst_detection_too_few_spikes() {
        let train = make_train(&[10], 100);
        assert!(burst_detection(&train, 0.001, 10.0, 3).is_empty());
    }

    #[test]
    fn test_first_spike_latency() {
        let train = make_train(&[50], 100);
        assert!((first_spike_latency(&train, 0.001) - 0.050).abs() < 1e-10);
    }

    #[test]
    fn test_first_spike_latency_no_spike() {
        let train = vec![0i32; 100];
        assert!(first_spike_latency(&train, 0.001).is_nan());
    }

    #[test]
    fn test_response_onset_detected() {
        // Baseline: 100 zeros. Response: spike at step 100
        let mut train = vec![0i32; 200];
        train[100] = 1;
        let onset = response_onset(&train, 100, 0.001, 1.0);
        assert!(!onset.is_nan(), "should detect onset");
        assert!((onset - 0.100).abs() < 1e-10);
    }

    #[test]
    fn test_response_onset_no_response() {
        let train = vec![0i32; 200];
        assert!(response_onset(&train, 100, 0.001, 3.0).is_nan());
    }

    #[test]
    fn test_response_onset_too_short() {
        let train = vec![0i32; 50];
        assert!(response_onset(&train, 100, 0.001, 3.0).is_nan());
    }

    #[test]
    fn test_change_point_step() {
        // Low rate then high rate
        let mut train = vec![0i32; 500];
        for i in 250..500 {
            if i % 2 == 0 {
                train[i] = 1;
            }
        }
        let cps = change_point_detection(&train, 50, 3.0);
        assert!(!cps.is_empty(), "should detect rate change");
    }

    #[test]
    fn test_change_point_constant() {
        let train = vec![0i32; 500];
        assert!(change_point_detection(&train, 50, 3.0).is_empty());
    }

    #[test]
    fn test_change_point_too_short() {
        let train = vec![1i32; 10];
        assert!(change_point_detection(&train, 50, 3.0).is_empty());
    }
}
