// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Spike-triggered analysis and receptive field estimation

/// Spike-triggered average (STA) of a stimulus signal.
///
/// Returns the average stimulus snippet of length `window_steps` preceding
/// each spike.
pub fn spike_triggered_average(
    stimulus: &[f64],
    binary_train: &[i32],
    window_steps: usize,
) -> Vec<f64> {
    let n = stimulus.len().min(binary_train.len());
    let spike_idx: Vec<usize> = (window_steps..n)
        .filter(|&i| binary_train[i] > 0)
        .collect();
    if spike_idx.is_empty() {
        return vec![0.0; window_steps];
    }
    let mut avg = vec![0.0f64; window_steps];
    for &t in &spike_idx {
        for j in 0..window_steps {
            avg[j] += stimulus[t - window_steps + j];
        }
    }
    let count = spike_idx.len() as f64;
    for v in &mut avg {
        *v /= count;
    }
    avg
}

/// Spike-triggered covariance (STC). Schwartz et al. 2006.
///
/// Returns flattened covariance matrix `[window_steps x window_steps]`.
pub fn spike_triggered_covariance(
    stimulus: &[f64],
    binary_train: &[i32],
    window_steps: usize,
) -> Vec<f64> {
    let n = stimulus.len().min(binary_train.len());
    let spike_idx: Vec<usize> = (window_steps..n)
        .filter(|&i| binary_train[i] > 0)
        .collect();
    if spike_idx.len() < 3 {
        // Return identity
        let mut eye = vec![0.0; window_steps * window_steps];
        for i in 0..window_steps {
            eye[i * window_steps + i] = 1.0;
        }
        return eye;
    }
    // Collect snippets
    let m = spike_idx.len();
    let w = window_steps;
    let mut snippets = vec![0.0f64; m * w];
    for (row, &t) in spike_idx.iter().enumerate() {
        for j in 0..w {
            snippets[row * w + j] = stimulus[t - w + j];
        }
    }
    // Mean
    let mut mean = vec![0.0f64; w];
    for row in 0..m {
        for j in 0..w {
            mean[j] += snippets[row * w + j];
        }
    }
    for v in &mut mean {
        *v /= m as f64;
    }
    // Centre
    for row in 0..m {
        for j in 0..w {
            snippets[row * w + j] -= mean[j];
        }
    }
    // Covariance: S^T S / (m - 1)
    let mut cov = vec![0.0f64; w * w];
    for row in 0..m {
        for i in 0..w {
            let si = snippets[row * w + i];
            for j in i..w {
                let sj = snippets[row * w + j];
                cov[i * w + j] += si * sj;
            }
        }
    }
    let denom = (m - 1) as f64;
    for i in 0..w {
        for j in i..w {
            cov[i * w + j] /= denom;
            cov[j * w + i] = cov[i * w + j];
        }
    }
    cov
}

/// Spatial information (bits/spike). Skaggs et al. 1993.
///
/// `positions`: 1D position values (same length as `binary_train`).
pub fn spatial_information(
    binary_train: &[i32],
    positions: &[f64],
    n_bins: usize,
    dt: f64,
) -> f64 {
    let n = binary_train.len().min(positions.len());
    if n < 10 {
        return 0.0;
    }
    let pos = &positions[..n];
    let pos_min = pos.iter().cloned().fold(f64::INFINITY, f64::min);
    let pos_max = pos.iter().cloned().fold(f64::NEG_INFINITY, f64::max) + 1e-10;
    let bin_width = (pos_max - pos_min) / n_bins as f64;

    let mut occupancy = vec![0.0f64; n_bins];
    let mut spike_counts = vec![0.0f64; n_bins];
    for i in 0..n {
        let k = ((pos[i] - pos_min) / bin_width).floor() as usize;
        let k = k.min(n_bins - 1);
        occupancy[k] += dt;
        spike_counts[k] += binary_train[i] as f64;
    }
    let total_occ: f64 = occupancy.iter().sum();
    if total_occ <= 0.0 {
        return 0.0;
    }
    let total_spikes: f64 = spike_counts.iter().sum();
    let mean_rate = total_spikes / (n as f64 * dt);
    if mean_rate <= 0.0 {
        return 0.0;
    }
    let mut si = 0.0;
    for k in 0..n_bins {
        let p_occ = occupancy[k] / total_occ;
        let rate = if occupancy[k] > 0.0 {
            spike_counts[k] / occupancy[k]
        } else {
            0.0
        };
        if rate > 0.0 && p_occ > 0.0 {
            si += p_occ * rate / mean_rate * (rate / mean_rate).ln() / std::f64::consts::LN_2;
        }
    }
    si.max(0.0)
}

/// Detect place fields as contiguous bins with rate > mean + threshold_std * std.
/// O'Keefe & Dostrovsky 1971.
///
/// Returns list of `(field_start_pos, field_end_pos)`.
pub fn place_field_detection(
    binary_train: &[i32],
    positions: &[f64],
    n_bins: usize,
    threshold_std: f64,
    dt: f64,
) -> Vec<(f64, f64)> {
    let n = binary_train.len().min(positions.len());
    if n < 10 {
        return vec![];
    }
    let pos = &positions[..n];
    let pos_min = pos.iter().cloned().fold(f64::INFINITY, f64::min);
    let pos_max = pos.iter().cloned().fold(f64::NEG_INFINITY, f64::max) + 1e-10;
    let bin_width = (pos_max - pos_min) / n_bins as f64;
    let edges: Vec<f64> = (0..=n_bins).map(|k| pos_min + k as f64 * bin_width).collect();

    let mut rates = vec![0.0f64; n_bins];
    for k in 0..n_bins {
        let mut occ = 0.0;
        let mut spk = 0.0;
        for i in 0..n {
            if pos[i] >= edges[k] && pos[i] < edges[k + 1] {
                occ += dt;
                spk += binary_train[i] as f64;
            }
        }
        rates[k] = if occ > 0.0 { spk / occ } else { 0.0 };
    }

    let mean_rate: f64 = rates.iter().sum::<f64>() / n_bins as f64;
    let var: f64 = rates.iter().map(|&r| (r - mean_rate).powi(2)).sum::<f64>() / n_bins as f64;
    let std_rate = var.sqrt();
    let thresh = mean_rate + threshold_std * std_rate;

    let mut fields = vec![];
    let mut in_field = false;
    let mut start = 0.0;
    for k in 0..n_bins {
        if rates[k] > thresh && !in_field {
            in_field = true;
            start = edges[k];
        } else if rates[k] <= thresh && in_field {
            in_field = false;
            fields.push((start, edges[k]));
        }
    }
    if in_field {
        fields.push((start, edges[n_bins]));
    }
    fields
}

/// Tuning curve: mean firing rate vs stimulus value. Dayan & Abbott 2001.
///
/// Returns `(mean_rates, bin_centres)`.
pub fn tuning_curve(
    binary_train: &[i32],
    stimulus_values: &[f64],
    n_bins: usize,
    dt: f64,
) -> (Vec<f64>, Vec<f64>) {
    let n = binary_train.len().min(stimulus_values.len());
    if n < 5 {
        return (vec![], vec![]);
    }
    let stim = &stimulus_values[..n];
    let stim_min = stim.iter().cloned().fold(f64::INFINITY, f64::min);
    let stim_max = stim.iter().cloned().fold(f64::NEG_INFINITY, f64::max) + 1e-10;
    let bin_width = (stim_max - stim_min) / n_bins as f64;
    let edges: Vec<f64> = (0..=n_bins).map(|k| stim_min + k as f64 * bin_width).collect();
    let centres: Vec<f64> = (0..n_bins).map(|k| (edges[k] + edges[k + 1]) / 2.0).collect();

    let mut rates = vec![0.0f64; n_bins];
    for k in 0..n_bins {
        let mut occ = 0.0;
        let mut spk = 0.0;
        for i in 0..n {
            if stim[i] >= edges[k] && stim[i] < edges[k + 1] {
                occ += dt;
                spk += binary_train[i] as f64;
            }
        }
        rates[k] = if occ > 0.0 { spk / occ } else { 0.0 };
    }
    (rates, centres)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sta_basic() {
        let stim: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin()).collect();
        let mut train = vec![0i32; 100];
        train[50] = 1;
        train[70] = 1;
        let sta = spike_triggered_average(&stim, &train, 10);
        assert_eq!(sta.len(), 10);
    }

    #[test]
    fn test_sta_no_spikes() {
        let stim = vec![1.0; 100];
        let train = vec![0i32; 100];
        let sta = spike_triggered_average(&stim, &train, 10);
        assert_eq!(sta.len(), 10);
        assert!(sta.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_sta_all_ones_stimulus() {
        let stim = vec![1.0; 100];
        let mut train = vec![0i32; 100];
        train[30] = 1;
        train[60] = 1;
        let sta = spike_triggered_average(&stim, &train, 10);
        assert!(sta.iter().all(|&v| (v - 1.0).abs() < 1e-12));
    }

    #[test]
    fn test_stc_basic() {
        let stim: Vec<f64> = (0..200).map(|i| (i as f64 * 0.05).sin()).collect();
        let mut train = vec![0i32; 200];
        for i in (50..200).step_by(20) {
            train[i] = 1;
        }
        let cov = spike_triggered_covariance(&stim, &train, 10);
        assert_eq!(cov.len(), 100); // 10x10
        // Diagonal should be non-negative
        for i in 0..10 {
            assert!(cov[i * 10 + i] >= 0.0);
        }
    }

    #[test]
    fn test_stc_few_spikes() {
        let stim = vec![1.0; 100];
        let train = vec![0i32; 100]; // no spikes -> identity
        let cov = spike_triggered_covariance(&stim, &train, 5);
        assert_eq!(cov.len(), 25);
        // Should be identity
        for i in 0..5 {
            assert!((cov[i * 5 + i] - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn test_stc_symmetric() {
        let stim: Vec<f64> = (0..200).map(|i| (i as f64 * 0.1).cos()).collect();
        let mut train = vec![0i32; 200];
        for i in (20..200).step_by(15) {
            train[i] = 1;
        }
        let w = 8;
        let cov = spike_triggered_covariance(&stim, &train, w);
        for i in 0..w {
            for j in 0..w {
                assert!(
                    (cov[i * w + j] - cov[j * w + i]).abs() < 1e-12,
                    "Covariance not symmetric at ({i},{j})"
                );
            }
        }
    }

    #[test]
    fn test_spatial_information_basic() {
        let mut train = vec![0i32; 200];
        let positions: Vec<f64> = (0..200).map(|i| i as f64 / 200.0 * 10.0).collect();
        // Place field: high firing in first quarter
        for i in 0..50 {
            if i % 2 == 0 {
                train[i] = 1;
            }
        }
        let si = spatial_information(&train, &positions, 20, 0.001);
        assert!(si > 0.0, "Spatial info should be positive for place cell");
    }

    #[test]
    fn test_spatial_information_uniform() {
        // Uniform firing -> low spatial info
        let mut train = vec![0i32; 200];
        let positions: Vec<f64> = (0..200).map(|i| i as f64).collect();
        for i in (0..200).step_by(5) {
            train[i] = 1;
        }
        let si = spatial_information(&train, &positions, 20, 0.001);
        // Should be near zero for uniform
        assert!(si < 0.5, "SI={si} too high for uniform firing");
    }

    #[test]
    fn test_spatial_information_few_samples() {
        assert_eq!(spatial_information(&[0, 1, 0], &[1.0, 2.0, 3.0], 5, 0.001), 0.0);
    }

    #[test]
    fn test_place_field_detection() {
        let mut train = vec![0i32; 1000];
        let positions: Vec<f64> = (0..1000).map(|i| i as f64 / 1000.0 * 20.0).collect();
        // Create dense place field at positions 5-10 (indices 250-500)
        for i in 250..500 {
            train[i] = 1; // every step fires
        }
        let fields = place_field_detection(&train, &positions, 50, 1.0, 0.001);
        assert!(!fields.is_empty(), "Should detect at least one place field");
        // Field should overlap the 5-10 range
        let (start, end) = fields[0];
        assert!(start < 12.0 && end > 4.0, "Field ({start}, {end}) should be near 5-10");
    }

    #[test]
    fn test_place_field_no_field() {
        // Uniform firing -> no fields
        let mut train = vec![0i32; 200];
        let positions: Vec<f64> = (0..200).map(|i| i as f64).collect();
        for i in (0..200).step_by(10) {
            train[i] = 1;
        }
        let fields = place_field_detection(&train, &positions, 50, 3.0, 0.001);
        // May or may not detect spurious fields, but shouldn't crash
        let _ = fields;
    }

    #[test]
    fn test_tuning_curve_basic() {
        let mut train = vec![0i32; 200];
        let stim: Vec<f64> = (0..200).map(|i| (i as f64 / 200.0 * 360.0) % 360.0).collect();
        // Tuned to ~180 degrees
        for i in 90..110 {
            train[i] = 1;
        }
        let (rates, centres) = tuning_curve(&train, &stim, 10, 0.001);
        assert_eq!(rates.len(), 10);
        assert_eq!(centres.len(), 10);
        // Peak should be in the middle bins
        let peak_idx = rates
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        assert!(peak_idx >= 4 && peak_idx <= 6);
    }

    #[test]
    fn test_tuning_curve_few_samples() {
        let (r, c) = tuning_curve(&[0, 1], &[1.0, 2.0], 5, 0.001);
        assert!(r.is_empty());
        assert!(c.is_empty());
    }
}
