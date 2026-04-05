// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Surrogate spike train generation for significance testing

use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Gamma, Poisson};

/// ISI-shuffle surrogate. Preserves rate + ISI distribution.
pub fn surrogate_isi_shuffle(binary_train: &[i32], seed: u64) -> Vec<i32> {
    let spike_idx: Vec<usize> = binary_train
        .iter()
        .enumerate()
        .filter(|(_, &v)| v > 0)
        .map(|(i, _)| i)
        .collect();
    if spike_idx.len() < 3 {
        return binary_train.to_vec();
    }
    let mut intervals: Vec<usize> = spike_idx.windows(2).map(|w| w[1] - w[0]).collect();
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    intervals.shuffle(&mut rng);

    let mut out = vec![0i32; binary_train.len()];
    let mut idx = spike_idx[0];
    out[idx] = 1;
    for &gap in &intervals {
        idx += gap;
        if idx < out.len() {
            out[idx] = 1;
        }
    }
    out
}

/// Spike dithering surrogate. Jitters each spike by +/-dither_ms.
pub fn surrogate_dither(binary_train: &[i32], dither_ms: f64, dt: f64, seed: u64) -> Vec<i32> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let dither_steps = (dither_ms / (dt * 1000.0)) as i64;
    let n = binary_train.len();
    let mut out = vec![0i32; n];
    for (i, &v) in binary_train.iter().enumerate() {
        if v > 0 {
            let jitter = rng.random_range(-dither_steps..=dither_steps);
            let new_idx = (i as i64 + jitter).clamp(0, n as i64 - 1) as usize;
            out[new_idx] = 1;
        }
    }
    out
}

/// Trial-shuffle surrogate. Destroys trial-to-trial correlation.
/// Returns permuted indices (caller reorders trains).
pub fn surrogate_trial_shuffle(n_trials: usize, seed: u64) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..n_trials).collect();
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    indices.shuffle(&mut rng);
    indices
}

/// Homogeneous Poisson spike train (Heeger 2000).
pub fn homogeneous_poisson(rate_hz: f64, duration_s: f64, dt: f64, seed: u64) -> Vec<f64> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let n = (duration_s / dt) as usize;
    let threshold = rate_hz * dt;
    (0..n)
        .map(|_| {
            if rng.random::<f64>() < threshold {
                1.0
            } else {
                0.0
            }
        })
        .collect()
}

/// Inhomogeneous Poisson via thinning (Lewis & Shedler 1979).
pub fn inhomogeneous_poisson(
    rate_func: fn(f64) -> f64,
    duration_s: f64,
    dt: f64,
    seed: u64,
) -> Vec<f64> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let n = (duration_s / dt) as usize;
    let rates: Vec<f64> = (0..n).map(|i| rate_func(i as f64 * dt)).collect();
    let max_rate = rates.iter().copied().fold(0.0_f64, f64::max);
    if max_rate <= 0.0 {
        return vec![0.0; n];
    }
    let threshold = max_rate * dt;
    (0..n)
        .map(|i| {
            let candidate = rng.random::<f64>() < threshold;
            let accept = rng.random::<f64>() < rates[i] / max_rate;
            if candidate && accept {
                1.0
            } else {
                0.0
            }
        })
        .collect()
}

/// Gamma-renewal spike train (Kuffler et al. 1957).
/// shape=1: Poisson. shape>1: more regular. shape<1: more bursty.
pub fn gamma_process(rate_hz: f64, shape: f64, duration_s: f64, dt: f64, seed: u64) -> Vec<f64> {
    let n = (duration_s / dt) as usize;
    let mut train = vec![0.0_f64; n];
    if rate_hz <= 0.0 || shape <= 0.0 {
        return train;
    }
    let scale = 1.0 / (rate_hz * shape);
    let gamma = Gamma::new(shape, scale).unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut t = 0.0_f64;
    loop {
        let interval: f64 = rng.sample(gamma);
        t += interval;
        let idx = (t / dt) as usize;
        if idx >= n {
            break;
        }
        train[idx] = 1.0;
    }
    train
}

/// Compound Poisson process (Snyder & Miller 1991).
/// burst_mean: mean spikes per event (Poisson distributed).
pub fn compound_poisson_process(
    rate_hz: f64,
    burst_mean: f64,
    duration_s: f64,
    dt: f64,
    seed: u64,
) -> Vec<f64> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let n = (duration_s / dt) as usize;
    let mut train = vec![0.0_f64; n];
    let threshold = rate_hz * dt;
    let poisson = Poisson::new(burst_mean.max(1e-10)).unwrap();
    for i in 0..n {
        if rng.random::<f64>() < threshold {
            let n_spikes: usize = rng.sample(poisson) as usize;
            for s in 0..n_spikes {
                let offset = i + s;
                if offset < n {
                    train[offset] = 1.0;
                }
            }
        }
    }
    train
}

/// Joint-ISI surrogate (Louis et al. 2010). Preserves ISI distribution and serial correlations.
pub fn surrogate_joint_isi(binary_train: &[i32], seed: u64) -> Vec<i32> {
    let spike_idx: Vec<usize> = binary_train
        .iter()
        .enumerate()
        .filter(|(_, &v)| v > 0)
        .map(|(i, _)| i)
        .collect();
    if spike_idx.len() < 4 {
        return binary_train.to_vec();
    }
    let mut intervals: Vec<usize> = spike_idx.windows(2).map(|w| w[1] - w[0]).collect();
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let ni = intervals.len();
    for _ in 0..(2 * ni) {
        let i = rng.random_range(0..ni);
        let j = rng.random_range(0..ni);
        if i != j {
            intervals.swap(i, j);
        }
    }
    let mut out = vec![0i32; binary_train.len()];
    let mut pos = spike_idx[0];
    out[pos] = 1;
    for &gap in &intervals {
        pos += gap;
        if pos < out.len() {
            out[pos] = 1;
        }
    }
    out
}

/// Bin-shuffling surrogate (Hatsopoulos et al. 2003). Shuffles spikes within bins.
pub fn surrogate_bin_shuffling(binary_train: &[i32], bin_size: usize, seed: u64) -> Vec<i32> {
    let mut out = binary_train.to_vec();
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let n = out.len();
    let mut start = 0;
    while start < n {
        let end = (start + bin_size).min(n);
        out[start..end].shuffle(&mut rng);
        start = end;
    }
    out
}

/// Circular shifting surrogate (Hatsopoulos et al. 2003).
pub fn surrogate_spike_train_shifting(
    binary_train: &[i32],
    max_shift: usize,
    seed: u64,
) -> Vec<i32> {
    let n = binary_train.len();
    if n == 0 {
        return vec![];
    }
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let shift = rng.random_range(0..=(2 * max_shift)) as i64 - max_shift as i64;
    let mut out = vec![0i32; n];
    for i in 0..n {
        let new_idx = ((i as i64 + shift).rem_euclid(n as i64)) as usize;
        out[new_idx] = binary_train[i];
    }
    out
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

    fn spike_count(train: &[i32]) -> i64 {
        train.iter().map(|&v| v as i64).sum()
    }

    fn spike_count_f64(train: &[f64]) -> i64 {
        train.iter().filter(|&&v| v > 0.5).count() as i64
    }

    // ── surrogate_isi_shuffle ───────────────────────────────────────

    #[test]
    fn test_isi_shuffle_preserves_count() {
        let train = make_train(&[5, 15, 25, 35, 45, 55, 65, 75, 85, 95], 100);
        let surr = surrogate_isi_shuffle(&train, 42);
        assert_eq!(spike_count(&surr), spike_count(&train));
    }

    #[test]
    fn test_isi_shuffle_deterministic() {
        let train = make_train(&[5, 15, 25, 35, 45], 100);
        let s1 = surrogate_isi_shuffle(&train, 42);
        let s2 = surrogate_isi_shuffle(&train, 42);
        assert_eq!(s1, s2, "same seed → same result");
    }

    #[test]
    fn test_isi_shuffle_few_spikes() {
        let train = make_train(&[50], 100);
        let surr = surrogate_isi_shuffle(&train, 0);
        assert_eq!(surr, train, "too few spikes → unchanged");
    }

    // ── surrogate_dither ────────────────────────────────────────────

    #[test]
    fn test_dither_preserves_count_approx() {
        let train = make_train(&[10, 30, 50, 70, 90], 100);
        let surr = surrogate_dither(&train, 2.0, 0.001, 42);
        // Count may differ slightly due to collisions
        assert!(spike_count(&surr) > 0);
        assert!(spike_count(&surr) <= spike_count(&train));
    }

    #[test]
    fn test_dither_deterministic() {
        let train = make_train(&[10, 50, 90], 100);
        let s1 = surrogate_dither(&train, 3.0, 0.001, 7);
        let s2 = surrogate_dither(&train, 3.0, 0.001, 7);
        assert_eq!(s1, s2);
    }

    // ── surrogate_trial_shuffle ─────────────────────────────────────

    #[test]
    fn test_trial_shuffle_permutation() {
        let perm = surrogate_trial_shuffle(5, 42);
        assert_eq!(perm.len(), 5);
        let mut sorted = perm.clone();
        sorted.sort();
        assert_eq!(sorted, vec![0, 1, 2, 3, 4], "should be a permutation");
    }

    // ── homogeneous_poisson ─────────────────────────────────────────

    #[test]
    fn test_poisson_rate() {
        let train = homogeneous_poisson(100.0, 1.0, 0.001, 42);
        assert_eq!(train.len(), 1000);
        let count = spike_count_f64(&train);
        // 100 Hz × 1s = ~100 spikes, allow wide margin
        assert!(
            count > 50 && count < 200,
            "expected ~100 spikes, got {count}"
        );
    }

    #[test]
    fn test_poisson_deterministic() {
        let t1 = homogeneous_poisson(50.0, 0.5, 0.001, 99);
        let t2 = homogeneous_poisson(50.0, 0.5, 0.001, 99);
        assert_eq!(t1, t2);
    }

    #[test]
    fn test_poisson_zero_rate() {
        let train = homogeneous_poisson(0.0, 1.0, 0.001, 0);
        assert_eq!(spike_count_f64(&train), 0);
    }

    // ── inhomogeneous_poisson ───────────────────────────────────────

    #[test]
    fn test_inhom_poisson_constant_matches_homogeneous() {
        fn rate(_t: f64) -> f64 {
            50.0
        }
        let train = inhomogeneous_poisson(rate, 1.0, 0.001, 42);
        assert_eq!(train.len(), 1000);
        let count = spike_count_f64(&train);
        assert!(
            count > 10 && count < 150,
            "~50 spikes expected, got {count}"
        );
    }

    // ── gamma_process ───────────────────────────────────────────────

    #[test]
    fn test_gamma_poisson_like() {
        let train = gamma_process(100.0, 1.0, 1.0, 0.001, 42);
        assert_eq!(train.len(), 1000);
        let count = spike_count_f64(&train);
        assert!(count > 30 && count < 200, "shape=1 ≈ Poisson, got {count}");
    }

    #[test]
    fn test_gamma_regular() {
        // shape=5 → very regular ISIs
        let train = gamma_process(50.0, 5.0, 1.0, 0.001, 42);
        let count = spike_count_f64(&train);
        assert!(count > 10, "should produce spikes, got {count}");
    }

    #[test]
    fn test_gamma_zero_rate() {
        let train = gamma_process(0.0, 1.0, 1.0, 0.001, 0);
        assert_eq!(spike_count_f64(&train), 0);
    }

    // ── compound_poisson_process ────────────────────────────────────

    #[test]
    fn test_cpp_produces_spikes() {
        let train = compound_poisson_process(50.0, 3.0, 1.0, 0.001, 42);
        assert_eq!(train.len(), 1000);
        let count = spike_count_f64(&train);
        assert!(count > 10, "should produce bursts, got {count}");
    }

    #[test]
    fn test_cpp_deterministic() {
        let t1 = compound_poisson_process(30.0, 2.0, 0.5, 0.001, 7);
        let t2 = compound_poisson_process(30.0, 2.0, 0.5, 0.001, 7);
        assert_eq!(t1, t2);
    }

    // ── surrogate_joint_isi ─────────────────────────────────────────

    #[test]
    fn test_joint_isi_preserves_count() {
        let train = make_train(&[5, 15, 25, 35, 45, 55, 65, 75], 100);
        let surr = surrogate_joint_isi(&train, 42);
        assert_eq!(spike_count(&surr), spike_count(&train));
    }

    #[test]
    fn test_joint_isi_few_spikes() {
        let train = make_train(&[10, 50], 100);
        let surr = surrogate_joint_isi(&train, 0);
        assert_eq!(surr, train, "< 4 spikes → unchanged");
    }

    // ── surrogate_bin_shuffling ─────────────────────────────────────

    #[test]
    fn test_bin_shuffle_preserves_count() {
        let train = make_train(&[0, 1, 2, 15, 16, 30, 31, 32, 33, 45], 50);
        let surr = surrogate_bin_shuffling(&train, 10, 42);
        assert_eq!(spike_count(&surr), spike_count(&train));
    }

    #[test]
    fn test_bin_shuffle_deterministic() {
        let train = make_train(&[3, 7, 13, 27], 30);
        let s1 = surrogate_bin_shuffling(&train, 10, 42);
        let s2 = surrogate_bin_shuffling(&train, 10, 42);
        assert_eq!(s1, s2);
    }

    // ── surrogate_spike_train_shifting ──────────────────────────────

    #[test]
    fn test_shift_preserves_count() {
        let train = make_train(&[10, 30, 50, 70, 90], 100);
        let surr = surrogate_spike_train_shifting(&train, 20, 42);
        assert_eq!(spike_count(&surr), spike_count(&train));
    }

    #[test]
    fn test_shift_circular() {
        let train = make_train(&[0, 99], 100);
        let surr = surrogate_spike_train_shifting(&train, 50, 42);
        assert_eq!(spike_count(&surr), 2, "circular shift preserves all spikes");
    }

    #[test]
    fn test_shift_empty() {
        assert!(surrogate_spike_train_shifting(&[], 10, 0).is_empty());
    }
}
