// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Spectral analysis of spike trains

use rustfft::{num_complex::Complex64, FftPlanner};

/// Power spectral density of a binary spike train.
///
/// Returns `(psd, freqs_hz)` where `psd[k] = |FFT[k]|^2 / n` and
/// `freqs_hz` are the corresponding one-sided frequency bins.
pub fn power_spectrum(binary_train: &[i32], dt: f64) -> (Vec<f64>, Vec<f64>) {
    let n = binary_train.len();
    if n < 2 {
        return (vec![], vec![]);
    }
    let mean: f64 = binary_train.iter().map(|&s| s as f64).sum::<f64>() / n as f64;
    let mut buf: Vec<Complex64> = binary_train
        .iter()
        .map(|&s| Complex64::new(s as f64 - mean, 0.0))
        .collect();

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n);
    fft.process(&mut buf);

    let n_rfft = n / 2 + 1;
    let nf = n as f64;
    let psd: Vec<f64> = buf[..n_rfft]
        .iter()
        .map(|c| (c.re * c.re + c.im * c.im) / nf)
        .collect();
    let freqs: Vec<f64> = (0..n_rfft).map(|k| k as f64 / (nf * dt)).collect();
    (psd, freqs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_power_spectrum_basic() {
        let train = vec![1, 0, 1, 0, 1, 0, 1, 0, 1, 0];
        let (psd, freqs) = power_spectrum(&train, 0.001);
        assert_eq!(psd.len(), 6);
        assert_eq!(freqs.len(), 6);
        assert!((freqs[0]).abs() < 1e-12);
        // Nyquist at 500 Hz
        assert!((freqs[5] - 500.0).abs() < 1e-6);
    }

    #[test]
    fn test_power_spectrum_empty() {
        let (psd, freqs) = power_spectrum(&[1], 0.001);
        assert!(psd.is_empty());
        assert!(freqs.is_empty());
    }

    #[test]
    fn test_power_spectrum_silence() {
        let train = vec![0; 100];
        let (psd, _) = power_spectrum(&train, 0.001);
        // All zeros -> PSD all zero
        assert!(psd.iter().all(|&v| v.abs() < 1e-12));
    }

    #[test]
    fn test_power_spectrum_dc_removed() {
        // Constant 1 -> after mean subtraction all zero
        let train = vec![1; 64];
        let (psd, _) = power_spectrum(&train, 0.001);
        assert!(psd.iter().all(|&v| v.abs() < 1e-12));
    }

    #[test]
    fn test_power_spectrum_peak_at_periodic() {
        // Strong periodicity at bin 5 in 20-sample signal
        let mut train = vec![0i32; 20];
        for i in (0..20).step_by(4) {
            train[i] = 1;
        }
        let (psd, _) = power_spectrum(&train, 0.001);
        // Peak should be at the periodic frequency
        let peak_idx = psd
            .iter()
            .enumerate()
            .skip(1) // skip DC
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        // Period-4 pattern in 20 samples -> frequency at n/period = 20/4 = 5
        // But FFT index = frequency * n * dt / (1/(n*dt))...
        // Actually for n=20, period=4: peak at k = n/period = 5
        // The actual peak may be at k=5 or a harmonic depending on DC removal
        assert!(peak_idx == 5 || peak_idx == 10, "Expected peak at index 5 or 10, got {peak_idx}");
    }

    #[test]
    fn test_power_spectrum_length() {
        let train = vec![0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1];
        let (psd, freqs) = power_spectrum(&train, 0.001);
        assert_eq!(psd.len(), 13 / 2 + 1);
        assert_eq!(psd.len(), freqs.len());
    }
}
