// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Spike-LFP coupling: phase locking, coherence, phase

use rustfft::{num_complex::Complex64, FftPlanner};
use std::f64::consts::PI;

/// Compute instantaneous phase via analytic signal (one-sided FFT Hilbert).
fn instantaneous_phase(signal: &[f64]) -> Vec<f64> {
    let n = signal.len();
    if n == 0 {
        return vec![];
    }
    let mut buf: Vec<Complex64> = signal.iter().map(|&s| Complex64::new(s, 0.0)).collect();
    let mut planner = FftPlanner::new();
    let fft_fwd = planner.plan_fft_forward(n);
    fft_fwd.process(&mut buf);

    // Zero negative frequencies, double positive (analytic signal)
    buf[0] = Complex64::new(0.0, 0.0); // DC = 0 after centering isn't required here
    for k in 1..n {
        if k <= n / 2 {
            buf[k] *= 2.0;
        } else {
            buf[k] = Complex64::new(0.0, 0.0);
        }
    }
    if n > 0 {
        buf[0] = Complex64::new(signal.iter().sum::<f64>(), 0.0); // preserve DC? No — match Python
    }
    // Actually match the Python: fft * 2 * (arange(n) > 0)
    // So: bin 0 is zeroed, bins 1..n are doubled
    // Let me redo:
    let mut buf2: Vec<Complex64> = signal.iter().map(|&s| Complex64::new(s, 0.0)).collect();
    fft_fwd.process(&mut buf2);
    buf2[0] = Complex64::new(0.0, 0.0);
    for k in 1..n {
        buf2[k] *= 2.0;
    }
    let fft_inv = planner.plan_fft_inverse(n);
    fft_inv.process(&mut buf2);
    let scale = 1.0 / n as f64;
    buf2.iter().map(|c| (c * scale).arg()).collect()
}

/// Phase locking value (PLV) between spikes and LFP phase.
///
/// PLV = |mean(exp(j * phase_at_spikes))|.
pub fn phase_locking_value(binary_train: &[i32], lfp_signal: &[f64]) -> f64 {
    let n = binary_train.len().min(lfp_signal.len());
    if n == 0 {
        return 0.0;
    }
    let phase = instantaneous_phase(&lfp_signal[..n]);
    let spike_idx: Vec<usize> = (0..n).filter(|&i| binary_train[i] > 0).collect();
    if spike_idx.is_empty() {
        return 0.0;
    }
    let mut sum_re = 0.0;
    let mut sum_im = 0.0;
    for &i in &spike_idx {
        sum_re += phase[i].cos();
        sum_im += phase[i].sin();
    }
    let count = spike_idx.len() as f64;
    ((sum_re / count).powi(2) + (sum_im / count).powi(2)).sqrt()
}

/// Spike-field coherence (SFC).
///
/// Returns `(coherence, freqs_hz)`. SFC = |S_xy|^2 / (S_xx * S_yy).
pub fn spike_field_coherence(
    binary_train: &[i32],
    lfp_signal: &[f64],
    dt: f64,
) -> (Vec<f64>, Vec<f64>) {
    let n = binary_train.len().min(lfp_signal.len());
    if n < 2 {
        return (vec![], vec![]);
    }
    let mean_a: f64 = binary_train[..n].iter().map(|&s| s as f64).sum::<f64>() / n as f64;
    let mean_b: f64 = lfp_signal[..n].iter().sum::<f64>() / n as f64;

    let mut buf_a: Vec<Complex64> = binary_train[..n]
        .iter()
        .map(|&s| Complex64::new(s as f64 - mean_a, 0.0))
        .collect();
    let mut buf_b: Vec<Complex64> = lfp_signal[..n]
        .iter()
        .map(|&s| Complex64::new(s - mean_b, 0.0))
        .collect();

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n);
    fft.process(&mut buf_a);
    fft.process(&mut buf_b);

    let n_rfft = n / 2 + 1;
    let mut sfc = vec![0.0f64; n_rfft];
    for k in 0..n_rfft {
        let sab = buf_a[k] * buf_b[k].conj();
        let saa = buf_a[k].norm_sqr();
        let sbb = buf_b[k].norm_sqr();
        let denom = saa * sbb;
        sfc[k] = if denom > 1e-30 {
            sab.norm_sqr() / denom
        } else {
            0.0
        };
    }
    let freqs: Vec<f64> = (0..n_rfft).map(|k| k as f64 / (n as f64 * dt)).collect();
    (sfc, freqs)
}

/// Histogram of LFP phase at spike times.
///
/// Returns `(counts, bin_centres_rad)` with bins spanning [-pi, pi].
pub fn spike_phase_histogram(
    binary_train: &[i32],
    lfp_signal: &[f64],
    n_bins: usize,
) -> (Vec<i64>, Vec<f64>) {
    let n = binary_train.len().min(lfp_signal.len());
    let phase = instantaneous_phase(&lfp_signal[..n]);

    let edges: Vec<f64> = (0..=n_bins)
        .map(|k| -PI + k as f64 * 2.0 * PI / n_bins as f64)
        .collect();
    let centres: Vec<f64> = (0..n_bins)
        .map(|k| (edges[k] + edges[k + 1]) / 2.0)
        .collect();

    let mut hist = vec![0i64; n_bins];
    for i in 0..n {
        if binary_train[i] > 0 {
            let p = phase[i];
            let mut k = ((p + PI) / (2.0 * PI) * n_bins as f64).floor() as usize;
            if k >= n_bins {
                k = n_bins - 1;
            }
            hist[k] += 1;
        }
    }
    (hist, centres)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_lfp(n: usize, freq_hz: f64, dt: f64) -> Vec<f64> {
        (0..n)
            .map(|i| (2.0 * PI * freq_hz * i as f64 * dt).sin())
            .collect()
    }

    #[test]
    fn test_instantaneous_phase_length() {
        let lfp = make_lfp(100, 10.0, 0.001);
        let phase = instantaneous_phase(&lfp);
        assert_eq!(phase.len(), 100);
    }

    #[test]
    fn test_phase_locking_value_basic() {
        let n = 1000;
        let lfp = make_lfp(n, 10.0, 0.001);
        // Spikes locked to peaks (phase ~ 0)
        let phase = instantaneous_phase(&lfp);
        let mut train = vec![0i32; n];
        for i in 0..n {
            if phase[i].abs() < 0.3 {
                train[i] = 1;
            }
        }
        let plv = phase_locking_value(&train, &lfp);
        assert!(plv > 0.5, "PLV={plv} should be high for phase-locked spikes");
    }

    #[test]
    fn test_phase_locking_value_no_spikes() {
        let lfp = make_lfp(100, 10.0, 0.001);
        let train = vec![0i32; 100];
        assert_eq!(phase_locking_value(&train, &lfp), 0.0);
    }

    #[test]
    fn test_spike_field_coherence_shape() {
        let n = 200;
        let lfp = make_lfp(n, 20.0, 0.001);
        let mut train = vec![0i32; n];
        for i in (0..n).step_by(10) {
            train[i] = 1;
        }
        let (sfc, freqs) = spike_field_coherence(&train, &lfp, 0.001);
        assert_eq!(sfc.len(), n / 2 + 1);
        assert_eq!(freqs.len(), sfc.len());
        assert!(sfc.iter().all(|&v| v >= 0.0 && v <= 1.0 + 1e-10));
    }

    #[test]
    fn test_spike_field_coherence_empty() {
        let (sfc, freqs) = spike_field_coherence(&[1], &[0.5], 0.001);
        assert!(sfc.is_empty());
        assert!(freqs.is_empty());
    }

    #[test]
    fn test_spike_phase_histogram_shape() {
        let n = 500;
        let lfp = make_lfp(n, 10.0, 0.001);
        let mut train = vec![0i32; n];
        for i in (0..n).step_by(5) {
            train[i] = 1;
        }
        let (hist, centres) = spike_phase_histogram(&train, &lfp, 36);
        assert_eq!(hist.len(), 36);
        assert_eq!(centres.len(), 36);
        let total: i64 = hist.iter().sum();
        let expected: i64 = train.iter().map(|&v| v as i64).sum();
        assert_eq!(total, expected);
    }

    #[test]
    fn test_spike_phase_histogram_no_spikes() {
        let lfp = make_lfp(100, 10.0, 0.001);
        let train = vec![0i32; 100];
        let (hist, _) = spike_phase_histogram(&train, &lfp, 12);
        assert!(hist.iter().all(|&v| v == 0));
    }

    #[test]
    fn test_phase_locking_random_spikes_low() {
        // Random spikes -> PLV should be low
        let n = 2000;
        let lfp = make_lfp(n, 15.0, 0.001);
        let mut train = vec![0i32; n];
        let mut rng = 42u64;
        for i in 0..n {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            if rng % 20 == 0 {
                train[i] = 1;
            }
        }
        let plv = phase_locking_value(&train, &lfp);
        assert!(plv < 0.3, "PLV={plv} should be low for random spikes");
    }
}
