use rayon::prelude::*;
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Neural population decoding algorithms

use nalgebra::{Cholesky, DMatrix};
use std::f64::consts::PI;

/// Georgopoulos population vector decoding.
/// trains: slice of binary spike trains (i32). preferred_directions: angle per neuron (radians).
/// Returns decoded angle per time bin.
pub fn population_vector_decode(
    trains: &[&[i32]],
    preferred_directions: &[f64],
    window: usize,
) -> Vec<f64> {
    if trains.is_empty() || window == 0 {
        return vec![];
    }
    let min_len = trains.iter().map(|t| t.len()).min().unwrap_or(0);
    let n_bins = min_len / window;
    if n_bins == 0 {
        return vec![];
    }
    // Pre-calculate cos/sin for preferred directions
    let dirs_cos: Vec<f64> = preferred_directions.iter().map(|&d| d.cos()).collect();
    let dirs_sin: Vec<f64> = preferred_directions.iter().map(|&d| d.sin()).collect();

    let decoded: Vec<f64> = (0..n_bins)
        .into_par_iter()
        .map(|b| {
            let mut sx = 0.0_f64;
            let mut sy = 0.0_f64;
            let start = b * window;
            let end = (b + 1) * window;
            for (i, t) in trains.iter().enumerate() {
                let count: i64 = t[start..end].iter().map(|&v| v as i64).sum();
                let c = dirs_cos.get(i).copied().unwrap_or(1.0);
                let s = dirs_sin.get(i).copied().unwrap_or(0.0);
                sx += count as f64 * c;
                sy += count as f64 * s;
            }
            sy.atan2(sx)
        })
        .collect();
    decoded
}

/// Bayesian MAP decoder (Dayan & Abbott 2001).
/// spike_counts: (n_neurons,). tuning_rates: (n_stimuli × n_neurons, row-major flat).
/// prior: (n_stimuli,) or empty for uniform. Returns MAP stimulus index.
pub fn bayesian_decode(
    spike_counts: &[f64],
    tuning_rates: &[f64],
    n_stimuli: usize,
    n_neurons: usize,
    prior: &[f64],
) -> usize {
    if n_stimuli == 0 || n_neurons == 0 {
        return 0;
    }
    let use_uniform = prior.is_empty();
    let log_prior_uniform = -(n_stimuli as f64).ln();

    let (best_s, _best_lp) = (0..n_stimuli)
        .into_par_iter()
        .map(|s| {
            let mut lp = if use_uniform {
                log_prior_uniform
            } else {
                (prior.get(s).copied().unwrap_or(1e-30) + 1e-30).ln()
            };
            let row_rates = &tuning_rates[s * n_neurons..(s + 1) * n_neurons];
            let mut j = 0;
            while j + 3 < n_neurons {
                let lam0 = row_rates[j].max(1e-10);
                let lam1 = row_rates[j + 1].max(1e-10);
                let lam2 = row_rates[j + 2].max(1e-10);
                let lam3 = row_rates[j + 3].max(1e-10);

                lp += spike_counts[j] * lam0.ln() - lam0;
                lp += spike_counts[j + 1] * lam1.ln() - lam1;
                lp += spike_counts[j + 2] * lam2.ln() - lam2;
                lp += spike_counts[j + 3] * lam3.ln() - lam3;
                j += 4;
            }
            while j < n_neurons {
                let lam = row_rates[j].max(1e-10);
                lp += spike_counts[j] * lam.ln() - lam;
                j += 1;
            }
            (s, lp)
        })
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or((0, f64::NEG_INFINITY));
    best_s
}

/// Maximum likelihood stimulus decoder (Dayan & Abbott 2001). Uniform prior.
pub fn maximum_likelihood_decode(
    spike_counts: &[f64],
    tuning_rates: &[f64],
    n_stimuli: usize,
    n_neurons: usize,
) -> usize {
    bayesian_decode(spike_counts, tuning_rates, n_stimuli, n_neurons, &[])
}

/// Fisher linear discriminant decoder (Fisher 1936).
/// train_data: (n_samples × n_features, row-major flat). labels: (n_samples,).
/// test_point: (n_features,). Returns predicted class label.
pub fn linear_discriminant_decode(
    train_data: &[f64],
    n_samples: usize,
    n_features: usize,
    labels: &[i64],
    test_point: &[f64],
) -> i64 {
    if n_samples == 0 || n_features == 0 {
        return 0;
    }

    // Unique classes
    let mut classes: Vec<i64> = labels[..n_samples].to_vec();
    classes.sort();
    classes.dedup();
    if classes.len() < 2 {
        return classes.first().copied().unwrap_or(0);
    }

    // Class means (parallelised)
    let (class_means, class_indices): (Vec<Vec<f64>>, Vec<Vec<usize>>) = classes
        .par_iter()
        .map(|&c| {
            let indices: Vec<usize> = (0..n_samples).filter(|&i| labels[i] == c).collect();
            let mut mean = vec![0.0_f64; n_features];
            for &idx in &indices {
                let row = &train_data[idx * n_features..(idx + 1) * n_features];
                for f in 0..n_features {
                    mean[f] += row[f];
                }
            }
            let n_c = indices.len() as f64;
            for v in &mut mean {
                *v /= n_c;
            }
            (mean, indices)
        })
        .unzip();

    // Within-class scatter S_w (n_features × n_features)
    let nf = n_features;
    let mut s_w = vec![0.0_f64; nf * nf];
    for (ci, indices) in class_indices.iter().enumerate() {
        let mean = &class_means[ci];
        for &idx in indices {
            for i in 0..nf {
                let di = train_data[idx * nf + i] - mean[i];
                for j in 0..nf {
                    let dj = train_data[idx * nf + j] - mean[j];
                    s_w[i * nf + j] += di * dj;
                }
            }
        }
    }
    // Regularise
    for i in 0..nf {
        s_w[i * nf + i] += 1e-8;
    }

    // Overall mean
    let mut overall_mean = vec![0.0_f64; nf];
    for i in 0..n_samples {
        for f in 0..nf {
            overall_mean[f] += train_data[i * nf + f];
        }
    }
    for v in &mut overall_mean {
        *v /= n_samples as f64;
    }

    // Every class's Fisher weights share the same SPD system S_w; assemble the
    // (mean_c - overall_mean) right-hand sides as columns and solve them all from
    // a single Cholesky factorisation. Column ci of `weights` is
    // w_ci = S_w^{-1} (mean_ci - overall_mean); the decision is score = w_ci · test.
    let n_classes = classes.len();
    let mut diffs = vec![0.0_f64; nf * n_classes];
    for (ci, mean) in class_means.iter().enumerate() {
        for f in 0..nf {
            diffs[f * n_classes + ci] = mean[f] - overall_mean[f];
        }
    }
    let weights = solve_spd(&s_w, &diffs, nf, n_classes);

    let mut best_class = classes[0];
    let mut best_score = f64::NEG_INFINITY;
    for (ci, &c) in classes.iter().enumerate() {
        let score: f64 = (0..nf)
            .map(|f| weights[f * n_classes + ci] * test_point[f])
            .sum();
        if score > best_score {
            best_score = score;
            best_class = c;
        }
    }
    best_class
}

/// Gaussian naive Bayes decoder (Mitchell 1997).
/// train_data: (n_samples × n_features, row-major flat). labels: (n_samples,).
/// test_point: (n_features,). Returns predicted class label.
pub fn naive_bayes_decode(
    train_data: &[f64],
    n_samples: usize,
    n_features: usize,
    labels: &[i64],
    test_point: &[f64],
) -> i64 {
    if n_samples == 0 || n_features == 0 {
        return 0;
    }

    let mut classes: Vec<i64> = labels[..n_samples].to_vec();
    classes.sort();
    classes.dedup();

    let mut best_class = classes.first().copied().unwrap_or(0);
    let mut best_log_p = f64::NEG_INFINITY;

    for &c in &classes {
        let indices: Vec<usize> = (0..n_samples).filter(|&i| labels[i] == c).collect();
        let n_c = indices.len() as f64;
        let log_prior = (n_c / n_samples as f64).ln();

        // Per-feature mean and variance
        let mut log_likelihood = 0.0_f64;
        for f in 0..n_features {
            let vals: Vec<f64> = indices
                .iter()
                .map(|&i| train_data[i * n_features + f])
                .collect();
            let mu: f64 = vals.iter().sum::<f64>() / n_c;
            let var: f64 = vals.iter().map(|&v| (v - mu).powi(2)).sum::<f64>() / n_c + 1e-10;
            let x = test_point[f];
            log_likelihood += -0.5 * ((2.0 * PI * var).ln() + (x - mu).powi(2) / var);
        }

        let log_p = log_prior + log_likelihood;
        if log_p > best_log_p {
            best_log_p = log_p;
            best_class = c;
        }
    }
    best_class
}

/// Solve the symmetric positive-definite system `A X = B` via Cholesky
/// factorisation. `a` is n×n row-major — here the ridge-regularised within-class
/// scatter, which is SPD — and `b` is n×m row-major; returns X (n×m, row-major).
/// A single factorisation solves every right-hand-side column (one per class),
/// the numerically optimal route for an SPD system: half the arithmetic of a
/// general elimination and unconditionally stable without pivoting. Falls back to
/// a zero solution if `a` is not positive-definite, which the ridge precludes.
fn solve_spd(a: &[f64], b: &[f64], n: usize, m: usize) -> Vec<f64> {
    let a_mat = DMatrix::<f64>::from_row_slice(n, n, a);
    let b_mat = DMatrix::<f64>::from_row_slice(n, m, b);
    match Cholesky::new(a_mat) {
        Some(chol) => {
            let x = chol.solve(&b_mat);
            let mut out = vec![0.0_f64; n * m];
            for i in 0..n {
                for j in 0..m {
                    out[i * m + j] = x[(i, j)];
                }
            }
            out
        }
        None => vec![0.0_f64; n * m],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── population_vector_decode ────────────────────────────────────

    #[test]
    fn test_pv_single_neuron_right() {
        // Single neuron with preferred direction 0 (right)
        let train = vec![1i32; 100];
        let trains: Vec<&[i32]> = vec![&train];
        let dirs = vec![0.0_f64]; // 0 radians = right
        let decoded = population_vector_decode(&trains, &dirs, 50);
        assert_eq!(decoded.len(), 2);
        assert!((decoded[0] - 0.0).abs() < 1e-10, "should decode to 0 rad");
    }

    #[test]
    fn test_pv_two_neurons_45deg() {
        // Two neurons: one at 0, one at π/2, equal firing → 45°
        let train = vec![1i32; 100];
        let trains: Vec<&[i32]> = vec![&train, &train];
        let dirs = vec![0.0, PI / 2.0];
        let decoded = population_vector_decode(&trains, &dirs, 100);
        assert_eq!(decoded.len(), 1);
        assert!(
            (decoded[0] - PI / 4.0).abs() < 1e-10,
            "equal firing at 0 and π/2 → π/4, got {}",
            decoded[0]
        );
    }

    #[test]
    fn test_pv_empty() {
        let decoded = population_vector_decode(&[], &[], 50);
        assert!(decoded.is_empty());
    }

    #[test]
    fn test_pv_no_bins() {
        let train = vec![1i32; 10];
        let trains: Vec<&[i32]> = vec![&train];
        let decoded = population_vector_decode(&trains, &[0.0], 100);
        assert!(decoded.is_empty(), "train shorter than window → empty");
    }

    // ── bayesian_decode ─────────────────────────────────────────────

    #[test]
    fn test_bayesian_obvious() {
        // 2 stimuli, 2 neurons. Stimulus 0: high rate neuron 0, low neuron 1.
        let tuning = vec![10.0, 0.1, 0.1, 10.0]; // 2×2
        let counts = vec![8.0, 0.0]; // neuron 0 fires a lot → stimulus 0
        let s = bayesian_decode(&counts, &tuning, 2, 2, &[]);
        assert_eq!(s, 0, "high neuron 0 firing → stimulus 0");
    }

    #[test]
    fn test_bayesian_with_prior() {
        let tuning = vec![5.0, 5.0, 5.0, 5.0]; // equal tuning
        let counts = vec![5.0, 5.0];
        let prior = vec![0.1, 0.9]; // strong prior for stimulus 1
        let s = bayesian_decode(&counts, &tuning, 2, 2, &prior);
        assert_eq!(s, 1, "equal evidence + strong prior → stimulus 1");
    }

    #[test]
    fn test_bayesian_empty() {
        assert_eq!(bayesian_decode(&[], &[], 0, 0, &[]), 0);
    }

    // ── maximum_likelihood_decode ───────────────────────────────────

    #[test]
    fn test_ml_matches_bayesian_uniform() {
        let tuning = vec![10.0, 0.1, 0.1, 10.0];
        let counts = vec![0.0, 8.0]; // neuron 1 fires → stimulus 1
        let s_ml = maximum_likelihood_decode(&counts, &tuning, 2, 2);
        let s_bay = bayesian_decode(&counts, &tuning, 2, 2, &[]);
        assert_eq!(s_ml, s_bay);
        assert_eq!(s_ml, 1);
    }

    // ── linear_discriminant_decode ──────────────────────────────────

    #[test]
    fn test_lda_separable() {
        // Class 0: features around (0, 0). Class 1: around (10, 10).
        // Fisher score = w . test where w = S_w_inv * (mean_c - overall_mean).
        // Test points at class centroids for unambiguous projection.
        #[rustfmt::skip]
        let data = vec![
            0.0, 0.0,
            0.1, 0.1,
            -0.1, 0.1,
            10.0, 10.0,
            10.1, 9.9,
            9.9, 10.1,
        ];
        let labels = vec![0_i64, 0, 0, 1, 1, 1];
        // Test at class centroids: class 1 centroid should decode to 1
        let test_1 = vec![10.0, 10.0];
        assert_eq!(linear_discriminant_decode(&data, 6, 2, &labels, &test_1), 1);
        // Two different tests should give different classes
        let r0 = linear_discriminant_decode(&data, 6, 2, &labels, &[-5.0, -5.0]);
        let r1 = linear_discriminant_decode(&data, 6, 2, &labels, &[15.0, 15.0]);
        assert_ne!(r0, r1, "distant points should decode to different classes");
    }

    #[test]
    fn test_lda_single_class() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let labels = vec![5_i64, 5];
        let test = vec![2.0, 3.0];
        assert_eq!(linear_discriminant_decode(&data, 2, 2, &labels, &test), 5);
    }

    #[test]
    fn test_lda_empty() {
        assert_eq!(linear_discriminant_decode(&[], 0, 0, &[], &[]), 0);
    }

    // ── naive_bayes_decode ──────────────────────────────────────────

    #[test]
    fn test_nb_separable() {
        #[rustfmt::skip]
        let data = vec![
            0.0, 0.0,
            0.1, 0.1,
            -0.1, -0.1,
            10.0, 10.0,
            10.1, 10.1,
            9.9, 9.9,
        ];
        let labels = vec![0_i64, 0, 0, 1, 1, 1];
        let test_0 = vec![0.2, 0.2];
        let test_1 = vec![9.8, 9.8];
        assert_eq!(naive_bayes_decode(&data, 6, 2, &labels, &test_0), 0);
        assert_eq!(naive_bayes_decode(&data, 6, 2, &labels, &test_1), 1);
    }

    #[test]
    fn test_nb_single_class() {
        let data = vec![1.0, 2.0];
        let labels = vec![7_i64];
        assert_eq!(naive_bayes_decode(&data, 1, 2, &labels, &[1.0, 2.0]), 7);
    }

    #[test]
    fn test_nb_agrees_with_lda_simple() {
        // For well-separated Gaussian data, NB and LDA should agree
        #[rustfmt::skip]
        let data = vec![
            -5.0, -5.0,
            -4.9, -5.1,
            5.0, 5.0,
            5.1, 4.9,
        ];
        let labels = vec![0_i64, 0, 1, 1];
        let test = vec![4.0, 4.0];
        let lda = linear_discriminant_decode(&data, 4, 2, &labels, &test);
        let nb = naive_bayes_decode(&data, 4, 2, &labels, &test);
        assert_eq!(lda, nb, "well-separated → both predict same class");
    }

    // ── solve_spd ───────────────────────────────────────────────────

    #[test]
    fn test_solve_spd_2x2() {
        // [2 1; 1 3] x = [5; 10] → x = [1, 3]  (matrix is SPD)
        let a = vec![2.0, 1.0, 1.0, 3.0];
        let b = vec![5.0, 10.0];
        let x = solve_spd(&a, &b, 2, 1);
        assert!((x[0] - 1.0).abs() < 1e-10);
        assert!((x[1] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_solve_spd_multi_rhs() {
        // diag(2, 4) X = [[2, 4]; [4, 8]] → X = [[1, 2]; [1, 2]] (row-major)
        let a = vec![2.0, 0.0, 0.0, 4.0];
        let b = vec![2.0, 4.0, 4.0, 8.0];
        let x = solve_spd(&a, &b, 2, 2);
        assert!((x[0] - 1.0).abs() < 1e-10); // X[0,0]
        assert!((x[1] - 2.0).abs() < 1e-10); // X[0,1]
        assert!((x[2] - 1.0).abs() < 1e-10); // X[1,0]
        assert!((x[3] - 2.0).abs() < 1e-10); // X[1,1]
    }

    #[test]
    fn test_solve_spd_non_pd_falls_back_to_zero() {
        // Indefinite matrix → Cholesky fails → zero solution.
        let a = vec![0.0, 1.0, 1.0, 0.0];
        let b = vec![1.0, 1.0];
        let x = solve_spd(&a, &b, 2, 1);
        assert_eq!(x, vec![0.0, 0.0]);
    }
}
