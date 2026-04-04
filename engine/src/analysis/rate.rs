// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Instantaneous and population firing rate estimation

/// Build a 1-D kernel for rate estimation.
fn build_kernel(kernel: &str, sigma_steps: usize) -> Vec<f64> {
    match kernel {
        "gaussian" => {
            let hw = 3 * sigma_steps;
            let mut k: Vec<f64> = (0..2 * hw + 1)
                .map(|i| {
                    let x = i as f64 - hw as f64;
                    (-0.5 * (x / sigma_steps as f64).powi(2)).exp()
                })
                .collect();
            let s: f64 = k.iter().sum();
            k.iter_mut().for_each(|v| *v /= s);
            k
        }
        "exponential" => {
            let hw = 5 * sigma_steps;
            let mut k: Vec<f64> = (0..hw)
                .map(|i| (-(i as f64) / sigma_steps as f64).exp())
                .collect();
            let s: f64 = k.iter().sum();
            k.iter_mut().for_each(|v| *v /= s);
            k
        }
        "rectangular" => {
            let hw = sigma_steps;
            let len = 2 * hw + 1;
            vec![1.0 / len as f64; len]
        }
        _ => vec![1.0],
    }
}

/// Convolve signal with kernel (mode="same").
fn convolve_same(signal: &[f64], kernel: &[f64]) -> Vec<f64> {
    let n = signal.len();
    let kn = kernel.len();
    let pad = kn / 2;
    let mut out = vec![0.0; n];
    for i in 0..n {
        let mut acc = 0.0;
        for j in 0..kn {
            let si = i as isize + j as isize - pad as isize;
            if si >= 0 && (si as usize) < n {
                acc += signal[si as usize] * kernel[j];
            }
        }
        out[i] = acc;
    }
    out
}

/// Instantaneous firing rate via kernel convolution (Hz).
pub fn instantaneous_rate(binary_train: &[f64], dt: f64, kernel: &str, sigma_ms: f64) -> Vec<f64> {
    let sigma_steps = (sigma_ms / (dt * 1000.0)).round().max(1.0) as usize;
    let mut k = build_kernel(kernel, sigma_steps);
    // Normalise to produce Hz: divide by dt
    let s: f64 = k.iter().sum();
    if s > 0.0 {
        let norm = s * dt;
        k.iter_mut().for_each(|v| *v /= norm);
    }
    // Re-normalise because build_kernel already normalised by sum
    // The Python does: k /= k.sum() * dt, applied to un-normalised kernel.
    // So we need to undo the build_kernel normalisation and redo correctly.
    let mut k_raw = build_kernel(kernel, sigma_steps);
    let s_raw: f64 = k_raw.iter().sum();
    let norm = s_raw * dt;
    k_raw.iter_mut().for_each(|v| *v /= norm);

    convolve_same(binary_train, &k_raw)
}

/// Population-level instantaneous firing rate (Hz).
pub fn population_rate(trains: &[Vec<f64>], dt: f64, sigma_ms: f64) -> Vec<f64> {
    if trains.is_empty() {
        return vec![];
    }
    let min_len = trains.iter().map(|t| t.len()).min().unwrap_or(0);
    if min_len == 0 {
        return vec![];
    }
    let mut total = vec![0.0; min_len];
    for t in trains {
        for (i, &v) in t.iter().take(min_len).enumerate() {
            total[i] += v;
        }
    }
    instantaneous_rate(&total, dt, "gaussian", sigma_ms)
}

/// Peri-stimulus time histogram.
/// Returns (rates_hz, bin_centers_ms).
pub fn psth(trials: &[Vec<f64>], bin_ms: f64, dt: f64) -> (Vec<f64>, Vec<f64>) {
    if trials.is_empty() {
        return (vec![], vec![]);
    }
    let max_len = trials.iter().map(|t| t.len()).max().unwrap_or(0);
    let bin_steps = (bin_ms / (dt * 1000.0)).round().max(1.0) as usize;
    let n_bins = max_len / bin_steps;
    if n_bins == 0 {
        return (vec![], vec![]);
    }
    let mut counts = vec![0.0; n_bins];
    for trial in trials {
        let usable = trial.len().min(n_bins * bin_steps);
        let n_full = usable / bin_steps;
        for b in 0..n_full {
            let s: f64 = trial[b * bin_steps..(b + 1) * bin_steps].iter().sum();
            counts[b] += s;
        }
    }
    let n_trials = trials.len() as f64;
    let bin_sec = bin_ms / 1000.0;
    let rates: Vec<f64> = counts.iter().map(|&c| c / (n_trials * bin_sec)).collect();
    let centers: Vec<f64> = (0..n_bins).map(|i| (i as f64 + 0.5) * bin_ms).collect();
    (rates, centers)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_instantaneous_rate_gaussian() {
        let mut train = vec![0.0; 100];
        train[50] = 1.0;
        let rate = instantaneous_rate(&train, 0.001, "gaussian", 5.0);
        assert_eq!(rate.len(), 100);
        // Peak should be near index 50
        let peak_idx = rate
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        assert!((peak_idx as i32 - 50).abs() <= 1);
        assert!(rate[50] > 0.0);
    }

    #[test]
    fn test_instantaneous_rate_rectangular() {
        let train = vec![1.0; 10];
        let rate = instantaneous_rate(&train, 0.001, "rectangular", 1.0);
        assert_eq!(rate.len(), 10);
        // Middle values should be ~1000 Hz (1 spike per ms)
        assert!(rate[5] > 500.0);
    }

    #[test]
    fn test_population_rate() {
        let t1 = vec![1.0, 0.0, 1.0, 0.0, 1.0];
        let t2 = vec![0.0, 1.0, 0.0, 1.0, 0.0];
        let rate = population_rate(&[t1, t2], 0.001, 1.0);
        assert_eq!(rate.len(), 5);
        // Combined: all ones → high rate
        assert!(rate[2] > 0.0);
    }

    #[test]
    fn test_psth() {
        let trial = vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        let (rates, centers) = psth(&[trial.clone(), trial], 5.0, 0.001);
        assert_eq!(rates.len(), 2);
        assert_eq!(centers.len(), 2);
        assert!((centers[0] - 2.5).abs() < 0.01);
        assert!(rates[0] > 0.0);
    }

    #[test]
    fn test_psth_empty() {
        let (r, c) = psth(&[], 10.0, 0.001);
        assert!(r.is_empty());
        assert!(c.is_empty());
    }
}
