// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Bootstrap and permutation significance testing
//
// NOTE: significance_bootstrap takes a function pointer, so it cannot
// be exposed via PyO3 directly. It is Rust-only, callable from other
// engine code and tests.

/// Bootstrap significance test for a pairwise statistic.
///
/// `statistic_func(a, b) -> f64` is any function computing a scalar
/// between two spike trains.  Returns `(observed_value, p_value)`.
pub fn significance_bootstrap<F>(
    statistic_func: F,
    train_a: &[f64],
    train_b: &[f64],
    n_surrogates: usize,
    seed: u64,
) -> (f64, f64)
where
    F: Fn(&[f64], &[f64]) -> f64,
{
    let observed = statistic_func(train_a, train_b);

    let mut combined: Vec<f64> = Vec::with_capacity(train_a.len() + train_b.len());
    combined.extend_from_slice(train_a);
    combined.extend_from_slice(train_b);

    let n_a = train_a.len();
    let n_total = combined.len();

    // Simple xoshiro256-based PRNG for permutations (Fisher-Yates)
    let mut rng = SimpleRng::new(seed);
    let mut count_extreme = 0u64;
    let mut perm = combined.clone();

    for _ in 0..n_surrogates {
        perm.copy_from_slice(&combined);
        // Fisher-Yates shuffle
        for i in (1..n_total).rev() {
            let j = rng.next_u64() as usize % (i + 1);
            perm.swap(i, j);
        }
        let surr_val = statistic_func(&perm[..n_a], &perm[n_a..]);
        if surr_val.abs() >= observed.abs() {
            count_extreme += 1;
        }
    }

    let p_value = (count_extreme + 1) as f64 / (n_surrogates + 1) as f64;
    (observed, p_value)
}

/// Minimal PRNG (splitmix64) for deterministic shuffling.
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_statistic(a: &[f64], b: &[f64]) -> f64 {
        let mean_a: f64 = if a.is_empty() {
            0.0
        } else {
            a.iter().sum::<f64>() / a.len() as f64
        };
        let mean_b: f64 = if b.is_empty() {
            0.0
        } else {
            b.iter().sum::<f64>() / b.len() as f64
        };
        mean_a - mean_b
    }

    #[test]
    fn test_bootstrap_identical() {
        let a = vec![1.0, 1.0, 1.0, 1.0, 1.0];
        let b = vec![1.0, 1.0, 1.0, 1.0, 1.0];
        let (obs, p) = significance_bootstrap(dummy_statistic, &a, &b, 200, 42);
        assert!((obs).abs() < 1e-12);
        // p-value should be close to 1 (no difference)
        assert!(p > 0.5);
    }

    #[test]
    fn test_bootstrap_different() {
        let a = vec![10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0];
        let b = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let (obs, p) = significance_bootstrap(dummy_statistic, &a, &b, 500, 42);
        assert!((obs - 10.0).abs() < 1e-12);
        // Should be significant
        assert!(p < 0.05, "p-value {p} not significant for clearly different groups");
    }

    #[test]
    fn test_bootstrap_p_range() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let (_, p) = significance_bootstrap(dummy_statistic, &a, &b, 100, 42);
        assert!(p > 0.0 && p <= 1.0);
    }

    #[test]
    fn test_bootstrap_deterministic() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let (obs1, p1) = significance_bootstrap(dummy_statistic, &a, &b, 100, 42);
        let (obs2, p2) = significance_bootstrap(dummy_statistic, &a, &b, 100, 42);
        assert!((obs1 - obs2).abs() < 1e-12);
        assert!((p1 - p2).abs() < 1e-12);
    }

    #[test]
    fn test_bootstrap_single_element() {
        let a = vec![5.0];
        let b = vec![0.0];
        let (obs, p) = significance_bootstrap(dummy_statistic, &a, &b, 50, 42);
        assert!((obs - 5.0).abs() < 1e-12);
        assert!(p > 0.0);
    }

    #[test]
    fn test_simple_rng_deterministic() {
        let mut rng1 = SimpleRng::new(123);
        let mut rng2 = SimpleRng::new(123);
        for _ in 0..100 {
            assert_eq!(rng1.next_u64(), rng2.next_u64());
        }
    }
}
