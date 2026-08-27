// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Basic spike train operations

/// Extract spike times (seconds) from a binary 0/1 array.
pub fn spike_times(binary_train: &[i32], dt: f64) -> Vec<f64> {
    binary_train
        .iter()
        .enumerate()
        .filter(|(_, &s)| s > 0)
        .map(|(i, _)| i as f64 * dt)
        .collect()
}

/// Inter-spike intervals (seconds) from a binary train.
pub fn isi(binary_train: &[i32], dt: f64) -> Vec<f64> {
    let times = spike_times(binary_train, dt);
    if times.len() < 2 {
        return vec![];
    }
    times.windows(2).map(|w| w[1] - w[0]).collect()
}

/// Mean firing rate (Hz).
pub fn firing_rate(binary_train: &[i32], dt: f64) -> f64 {
    let duration = binary_train.len() as f64 * dt;
    if duration <= 0.0 {
        return 0.0;
    }
    let count: i64 = binary_train.iter().map(|&s| s as i64).sum();
    count as f64 / duration
}

/// Total spike count.
pub fn spike_count(binary_train: &[i32]) -> i64 {
    let mut total = 0_i64;
    let (chunks, remainder) = binary_train.as_chunks::<4>();
    for c in chunks {
        total += (c[0] + c[1] + c[2] + c[3]) as i64;
    }
    for &s in remainder {
        total += s as i64;
    }
    total
}

/// Bin a binary spike train into spike counts per bin.
pub fn bin_spike_train(binary_train: &[i32], bin_size: usize) -> Vec<i64> {
    let n = binary_train.len();
    let n_bins = n / bin_size;
    if n_bins == 0 {
        return vec![binary_train.iter().map(|&s| s as i64).sum()];
    }
    let mut res = Vec::with_capacity(n_bins);
    for i in 0..n_bins {
        let chunk = &binary_train[i * bin_size..(i + 1) * bin_size];
        let mut total = 0_i64;
        let (chunks, remainder) = chunk.as_chunks::<4>();
        for c in chunks {
            total += (c[0] + c[1] + c[2] + c[3]) as i64;
        }
        for &s in remainder {
            total += s as i64;
        }
        res.push(total);
    }
    res
}

// Also accept f64 spike trains (common in Python pipelines)

/// Spike times from f64 binary train (threshold > 0.5).
pub fn spike_times_f64(binary_train: &[f64], dt: f64) -> Vec<f64> {
    binary_train
        .iter()
        .enumerate()
        .filter(|(_, &s)| s > 0.5)
        .map(|(i, _)| i as f64 * dt)
        .collect()
}

/// ISI from f64 binary train.
pub fn isi_f64(binary_train: &[f64], dt: f64) -> Vec<f64> {
    let times = spike_times_f64(binary_train, dt);
    if times.len() < 2 {
        return vec![];
    }
    times.windows(2).map(|w| w[1] - w[0]).collect()
}

/// Firing rate from f64 binary train.
pub fn firing_rate_f64(binary_train: &[f64], dt: f64) -> f64 {
    let duration = binary_train.len() as f64 * dt;
    if duration <= 0.0 {
        return 0.0;
    }
    let count: f64 = binary_train.iter().sum();
    count / duration
}

/// Spike count from f64 binary train.
pub fn spike_count_f64(binary_train: &[f64]) -> i64 {
    binary_train.iter().filter(|&&s| s > 0.5).count() as i64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spike_times_basic() {
        let train = vec![0, 1, 0, 0, 1, 1, 0];
        let times = spike_times(&train, 0.001);
        assert_eq!(times.len(), 3);
        assert!((times[0] - 0.001).abs() < 1e-12);
        assert!((times[1] - 0.004).abs() < 1e-12);
        assert!((times[2] - 0.005).abs() < 1e-12);
    }

    #[test]
    fn test_spike_times_empty() {
        let train = vec![0, 0, 0];
        assert!(spike_times(&train, 0.001).is_empty());
    }

    #[test]
    fn test_isi_basic() {
        let train = vec![0, 1, 0, 0, 1, 0];
        let intervals = isi(&train, 0.001);
        assert_eq!(intervals.len(), 1);
        assert!((intervals[0] - 0.003).abs() < 1e-12);
    }

    #[test]
    fn test_isi_single_spike() {
        let train = vec![0, 1, 0];
        assert!(isi(&train, 0.001).is_empty());
    }

    #[test]
    fn test_firing_rate() {
        let train = vec![1, 0, 1, 0, 1, 0, 1, 0, 1, 0];
        let rate = firing_rate(&train, 0.001);
        assert!((rate - 500.0).abs() < 0.01);
    }

    #[test]
    fn test_firing_rate_zero() {
        let train = vec![0, 0, 0];
        assert_eq!(firing_rate(&train, 0.001), 0.0);
    }

    #[test]
    fn test_spike_count() {
        let train = vec![1, 0, 1, 1, 0, 1];
        assert_eq!(spike_count(&train), 4);
    }

    #[test]
    fn test_bin_spike_train() {
        let train = vec![1, 0, 1, 1, 0, 0, 1, 1, 1, 0];
        let bins = bin_spike_train(&train, 5);
        assert_eq!(bins, vec![3, 3]);
    }

    #[test]
    fn test_bin_spike_train_remainder() {
        let train = vec![1, 1, 1, 1, 1, 1, 1];
        let bins = bin_spike_train(&train, 3);
        assert_eq!(bins, vec![3, 3]); // last 1 trimmed
    }

    #[test]
    fn test_f64_variants() {
        let train = vec![0.0, 1.0, 0.0, 1.0, 0.0];
        assert_eq!(spike_count_f64(&train), 2);
        assert!((firing_rate_f64(&train, 0.001) - 400.0).abs() < 0.01);
        assert_eq!(spike_times_f64(&train, 0.001).len(), 2);
        assert_eq!(isi_f64(&train, 0.001).len(), 1);
    }
}
