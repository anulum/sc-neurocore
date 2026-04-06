// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Dimensionality reduction for spike train populations

use rayon::prelude::*;
use super::basic;

// ── helpers ─────────────────────────────────────────────────────────

/// Symmetric eigendecomposition via Jacobi rotations.
/// `a` is row-major `n x n` (symmetric).
/// Returns `(eigenvalues, eigenvectors_col_major)` sorted descending.
fn symmetric_eigen(a: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut mat = a.to_vec();
    let mut vecs = vec![0.0f64; n * n];
    for i in 0..n {
        vecs[i * n + i] = 1.0;
    }
    let max_iter = 100 * n * n;
    for _ in 0..max_iter {
        // Find largest off-diagonal
        let mut p = 0;
        let mut q = 1;
        let mut max_val = 0.0f64;
        for i in 0..n {
            for j in i + 1..n {
                let v = mat[i * n + j].abs();
                if v > max_val {
                    max_val = v;
                    p = i;
                    q = j;
                }
            }
        }
        if max_val < 1e-15 {
            break;
        }
        let app = mat[p * n + p];
        let aqq = mat[q * n + q];
        let apq = mat[p * n + q];
        let theta = if (app - aqq).abs() < 1e-30 {
            std::f64::consts::FRAC_PI_4
        } else {
            0.5 * (2.0 * apq / (app - aqq)).atan()
        };
        let c = theta.cos();
        let s = theta.sin();
        // Apply rotation (unrolled)
        for i in 0..n {
            let i_off = i * n;
            let ip = mat[i_off + p];
            let iq = mat[i_off + q];
            mat[i_off + p] = c * ip + s * iq;
            mat[i_off + q] = -s * ip + c * iq;
        }
        let p_off = p * n;
        let q_off = q * n;
        for j in 0..n {
            let pj = mat[p_off + j];
            let qj = mat[q_off + j];
            mat[p_off + j] = c * pj + s * qj;
            mat[q_off + j] = -s * pj + c * qj;
        }
        // Update eigenvectors (unrolled)
        for i in 0..n {
            let i_off = i * n;
            let vip = vecs[i_off + p];
            let viq = vecs[i_off + q];
            vecs[i_off + p] = c * vip + s * viq;
            vecs[i_off + q] = -s * vip + c * viq;
        }
    }
    let eigenvalues: Vec<f64> = (0..n).map(|i| mat[i * n + i]).collect();
    // Sort descending
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| eigenvalues[b].partial_cmp(&eigenvalues[a]).unwrap());
    let sorted_vals: Vec<f64> = idx.iter().map(|&i| eigenvalues[i]).collect();
    let mut sorted_vecs = vec![0.0f64; n * n];
    for (new_col, &old_col) in idx.iter().enumerate() {
        for row in 0..n {
            sorted_vecs[row * n + new_col] = vecs[row * n + old_col];
        }
    }
    (sorted_vals, sorted_vecs)
}

/// PCA on binned spike count matrix (neurons x time_bins).
///
/// `trains`: list of binary trains (each `&[i32]`).
/// Returns `(projected, explained_variance_ratio)` where
/// `projected` is row-major `(n_components, min_bins)`.
pub fn spike_train_pca(
    trains: &[&[i32]],
    n_components: usize,
    bin_size: usize,
) -> (Vec<f64>, Vec<f64>) {
    if trains.is_empty() {
        return (vec![], vec![]);
    }
    let binned: Vec<Vec<f64>> = trains
        .iter()
        .map(|t| {
            basic::bin_spike_train(t, bin_size)
                .into_iter()
                .map(|c| c as f64)
                .collect()
        })
        .collect();
    let min_bins = binned.iter().map(|b| b.len()).min().unwrap_or(0);
    if min_bins == 0 {
        return (vec![], vec![]);
    }
    let d = trains.len(); // neurons
                          // Mean-centre each neuron
    let mut mat = vec![0.0f64; d * min_bins];
    for i in 0..d {
        let mean: f64 = binned[i][..min_bins].iter().sum::<f64>() / min_bins as f64;
        for j in 0..min_bins {
            mat[i * min_bins + j] = binned[i][j] - mean;
        }
    }
    if d < 2 {
        return (mat[..min_bins].to_vec(), vec![1.0]);
    }
    // Covariance matrix (d x d)
    let mut cov = vec![0.0f64; d * d];
    for i in 0..d {
        for j in i..d {
            let mut s = 0.0;
            for t in 0..min_bins {
                s += mat[i * min_bins + t] * mat[j * min_bins + t];
            }
            s /= (min_bins - 1).max(1) as f64;
            cov[i * d + j] = s;
            cov[j * d + i] = s;
        }
    }
    let (eigvals, eigvecs) = symmetric_eigen(&cov, d);
    let nc = n_components.min(d);
    let total: f64 = eigvals.iter().sum();
    let explained: Vec<f64> = eigvals[..nc]
        .iter()
        .map(|&v| if total > 0.0 { v / total } else { v })
        .collect();
    // Project: components^T @ mat
    // components are first nc columns of eigvecs
    let mut projected = vec![0.0f64; nc * min_bins];
    for c in 0..nc {
        for t in 0..min_bins {
            let mut s = 0.0;
            for i in 0..d {
                s += eigvecs[i * d + c] * mat[i * min_bins + t];
            }
            projected[c * min_bins + t] = s;
        }
    }
    (projected, explained)
}

/// Demixed PCA. Kobak et al. 2016.
///
/// `conditions`: list of (condition trains) where each is a `Vec<&[i32]>`.
/// Returns `(projected, explained_variance_ratio)`.
pub fn demixed_pca(
    conditions: &[Vec<&[i32]>],
    n_components: usize,
    bin_size: usize,
) -> (Vec<f64>, Vec<f64>) {
    if conditions.len() < 2 {
        return (vec![], vec![]);
    }
    // Compute mean binned vector per condition
    let mut all_means: Vec<Vec<f64>> = Vec::new();
    for trains in conditions {
        let binned: Vec<Vec<f64>> = trains
            .iter()
            .map(|t| {
                basic::bin_spike_train(t, bin_size)
                    .into_iter()
                    .map(|c| c as f64)
                    .collect()
            })
            .collect();
        let min_bins = binned.iter().map(|b| b.len()).min().unwrap_or(0);
        if min_bins == 0 {
            continue;
        }
        let n = binned.len();
        let mut mean = vec![0.0f64; min_bins];
        for b in &binned {
            for j in 0..min_bins {
                mean[j] += b[j];
            }
        }
        for v in &mut mean {
            *v /= n as f64;
        }
        all_means.push(mean);
    }
    if all_means.len() < 2 {
        return (vec![], vec![]);
    }
    let min_bins = all_means.iter().map(|m| m.len()).min().unwrap();
    let n_cond = all_means.len();
    // Grand mean
    let mut grand = vec![0.0f64; min_bins];
    for m in &all_means {
        for j in 0..min_bins {
            grand[j] += m[j];
        }
    }
    for v in &mut grand {
        *v /= n_cond as f64;
    }
    // Centre
    let mut mean_mat = vec![0.0f64; n_cond * min_bins];
    for i in 0..n_cond {
        for j in 0..min_bins {
            mean_mat[i * min_bins + j] = all_means[i][j] - grand[j];
        }
    }
    // Covariance: M^T M / n_cond (min_bins x min_bins) - unrolled & SIMD
    let t = min_bins;
    let mut cov = vec![0.0f64; t * t];
    let n_cond_f = n_cond as f64;
    
    // Transpose mean_mat to column-major for SIMD dots: (t x n_cond)
    let mut m_cols = vec![vec![0.0_f64; n_cond]; t];
    for c in 0..n_cond {
        for i in 0..t {
            m_cols[i][c] = mean_mat[c * t + i];
        }
    }

    cov.par_chunks_exact_mut(t)
        .enumerate()
        .for_each(|(i, row)| {
            for j in i..t {
                let dot = crate::simd::dot_f64_dispatch(&m_cols[i], &m_cols[j]);
                row[j] = dot / n_cond_f;
            }
        });
    // Mirror
    for i in 0..t {
        for j in (i + 1)..t {
            cov[j * t + i] = cov[i * t + j];
        }
    }
    let (eigvals, eigvecs) = symmetric_eigen(&cov, t);
    let nc = n_components.min(t);
    let total: f64 = eigvals.iter().sum();
    let explained: Vec<f64> = eigvals[..nc]
        .iter()
        .map(|&v| if total > 0.0 { v / total } else { v })
        .collect();
    // Project: mean_mat @ eigvecs[:, :nc]
    let mut projected = vec![0.0f64; n_cond * nc];
    for c in 0..n_cond {
        for k in 0..nc {
            let mut s = 0.0;
            for j in 0..t {
                s += mean_mat[c * t + j] * eigvecs[j * t + k];
            }
            projected[c * nc + k] = s;
        }
    }
    (projected, explained)
}

/// Factor analysis via EM. Rubin & Thayer 1982.
///
/// Returns `(loading_matrix [d*n_factors], uniquenesses [d])`.
pub fn factor_analysis(
    trains: &[&[i32]],
    n_factors: usize,
    bin_size: usize,
    n_iter: usize,
) -> (Vec<f64>, Vec<f64>) {
    let d = trains.len();
    if d == 0 {
        return (vec![], vec![]);
    }
    let binned: Vec<Vec<f64>> = trains
        .iter()
        .map(|t| {
            basic::bin_spike_train(t, bin_size)
                .into_iter()
                .map(|c| c as f64)
                .collect()
        })
        .collect();
    let t = binned.iter().map(|b| b.len()).min().unwrap_or(0);
    if t == 0 {
        return (vec![0.0; d * n_factors], vec![1.0; d]);
    }
    // Mean-centre
    let mut mat = vec![0.0f64; d * t];
    for i in 0..d {
        let mean: f64 = binned[i][..t].iter().sum::<f64>() / t as f64;
        for j in 0..t {
            mat[i * t + j] = binned[i][j] - mean;
        }
    }
    // Covariance (d x d)
    let mut cov = vec![0.0f64; d * d];
    for i in 0..d {
        for j in i..d {
            let mut s = 0.0;
            for k in 0..t {
                s += mat[i * t + k] * mat[j * t + k];
            }
            s /= t as f64;
            cov[i * d + j] = s;
            cov[j * d + i] = s;
        }
    }
    let nf = n_factors.min(d);
    let mut psi: Vec<f64> = (0..d).map(|i| cov[i * d + i]).collect();
    // Initialise loadings (deterministic pseudo-random)
    let mut loadings = vec![0.0f64; d * nf];
    let mut rng = 42u64;
    for v in &mut loadings {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        *v = ((rng >> 33) as f64 / (1u64 << 31) as f64 - 0.5) * 0.2;
    }

    for _ in 0..n_iter {
        // psi_inv: diagonal
        let psi_inv: Vec<f64> = psi.iter().map(|&p| 1.0 / (p + 1e-10)).collect();

        // M = L^T psi_inv L + I (nf x nf)
        let mut m = vec![0.0f64; nf * nf];
        for i in 0..nf {
            for j in 0..nf {
                let mut s = 0.0;
                for k in 0..d {
                    s += loadings[k * nf + i] * psi_inv[k] * loadings[k * nf + j];
                }
                m[i * nf + j] = s + if i == j { 1.0 } else { 0.0 };
            }
        }
        let m_inv = mat_inv_small(&m, nf);

        // beta = m_inv @ L^T @ psi_inv (nf x d)
        let mut beta = vec![0.0f64; nf * d];
        for i in 0..nf {
            for j in 0..d {
                let mut s = 0.0;
                for k in 0..nf {
                    s += m_inv[i * nf + k] * loadings[j * nf + k] * psi_inv[j];
                }
                beta[i * d + j] = s;
            }
        }

        // E[z] = beta @ mat (nf x t)
        let mut ez = vec![0.0f64; nf * t];
        for i in 0..nf {
            for j in 0..t {
                let mut s = 0.0;
                for k in 0..d {
                    s += beta[i * d + k] * mat[k * t + j];
                }
                ez[i * t + j] = s;
            }
        }

        // E[zz^T] = nf * m_inv + ez @ ez^T / t (nf x nf)
        let mut ezzt = vec![0.0f64; nf * nf];
        for i in 0..nf {
            for j in 0..nf {
                let mut s = 0.0;
                for k in 0..t {
                    s += ez[i * t + k] * ez[j * t + k];
                }
                // The Python uses: nf * m_inv + ez @ ez.T / t
                // But the correct EM formula is: t * m_inv + ez @ ez.T
                // Python's formula: ezzt = n_factors * m_inv + ez @ ez.T / t
                ezzt[i * nf + j] = nf as f64 * m_inv[i * nf + j] + s / t as f64;
            }
        }

        // L_new = (mat @ ez^T / t) @ inv(ezzt / t * t) = (mat @ ez^T / t) @ inv(ezzt)
        // Actually Python: loadings = mat @ ez.T / t @ np.linalg.inv(ezzt / t * t)
        // This simplifies to: (mat @ ez^T) @ inv(t * ezzt)
        // Let's just follow the Python exactly:
        // mat_ez_t = mat @ ez.T (d x nf)
        let mut mat_ez_t = vec![0.0f64; d * nf];
        for i in 0..d {
            for j in 0..nf {
                let mut s = 0.0;
                for k in 0..t {
                    s += mat[i * t + k] * ez[j * t + k];
                }
                mat_ez_t[i * nf + j] = s / t as f64;
            }
        }
        // Scale ezzt for inversion: the Python does ezzt / t * t which is just ezzt
        let ezzt_inv = mat_inv_small(&ezzt, nf);
        // L_new = mat_ez_t @ ezzt_inv
        for i in 0..d {
            for j in 0..nf {
                let mut s = 0.0;
                for k in 0..nf {
                    s += mat_ez_t[i * nf + k] * ezzt_inv[k * nf + j];
                }
                loadings[i * nf + j] = s;
            }
        }

        // psi = diag(cov - L @ ez @ mat^T / t)
        // L @ ez (d x t)
        let mut l_ez = vec![0.0f64; d * t];
        for i in 0..d {
            for j in 0..t {
                let mut s = 0.0;
                for k in 0..nf {
                    s += loadings[i * nf + k] * ez[k * t + j];
                }
                l_ez[i * t + j] = s;
            }
        }
        for i in 0..d {
            let mut s = 0.0;
            for k in 0..t {
                s += l_ez[i * t + k] * mat[i * t + k];
            }
            psi[i] = (cov[i * d + i] - s / t as f64).max(1e-6);
        }
    }

    (loadings, psi)
}

/// Small matrix inverse (Gauss-Jordan) for nf x nf matrices.
fn mat_inv_small(a: &[f64], n: usize) -> Vec<f64> {
    let mut aug = vec![0.0f64; n * 2 * n];
    for i in 0..n {
        for j in 0..n {
            aug[i * 2 * n + j] = a[i * n + j];
        }
        aug[i * 2 * n + n + i] = 1.0;
    }
    for col in 0..n {
        let mut max_row = col;
        let mut max_val = aug[col * 2 * n + col].abs();
        for row in col + 1..n {
            let v = aug[row * 2 * n + col].abs();
            if v > max_val {
                max_val = v;
                max_row = row;
            }
        }
        if max_val < 1e-30 {
            continue;
        }
        if max_row != col {
            for k in 0..2 * n {
                aug.swap(col * 2 * n + k, max_row * 2 * n + k);
            }
        }
        let pivot = aug[col * 2 * n + col];
        for k in 0..2 * n {
            aug[col * 2 * n + k] /= pivot;
        }
        for row in 0..n {
            if row == col {
                continue;
            }
            let factor = aug[row * 2 * n + col];
            for k in 0..2 * n {
                aug[row * 2 * n + k] -= factor * aug[col * 2 * n + k];
            }
        }
    }
    let mut inv = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            inv[i * n + j] = aug[i * 2 * n + n + j];
        }
    }
    inv
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_trains() -> Vec<Vec<i32>> {
        // 5 neurons, 200 steps each, varied rates
        let mut trains = Vec::new();
        for n in 0..5 {
            let mut t = vec![0i32; 200];
            let step = 5 + n * 3;
            for i in (0..200).step_by(step) {
                t[i] = 1;
            }
            trains.push(t);
        }
        trains
    }

    #[test]
    fn test_spike_train_pca_basic() {
        let trains = make_trains();
        let refs: Vec<&[i32]> = trains.iter().map(|t| t.as_slice()).collect();
        let (proj, explained) = spike_train_pca(&refs, 3, 10);
        assert_eq!(explained.len(), 3);
        // Explained variances should sum to <= 1
        let total: f64 = explained.iter().sum();
        assert!(total <= 1.0 + 1e-6, "Total explained {total} > 1");
        // First component should explain the most
        assert!(explained[0] >= explained[1]);
        assert!(!proj.is_empty());
    }

    #[test]
    fn test_spike_train_pca_empty() {
        let (proj, expl) = spike_train_pca(&[], 3, 10);
        assert!(proj.is_empty());
        assert!(expl.is_empty());
    }

    #[test]
    fn test_spike_train_pca_single_neuron() {
        let train = vec![1, 0, 1, 0, 1, 0, 1, 0, 1, 0];
        let refs = vec![train.as_slice()];
        let (proj, expl) = spike_train_pca(&refs, 1, 2);
        // Single neuron -> 1 component
        assert_eq!(expl.len(), 1);
        assert!(!proj.is_empty());
    }

    #[test]
    fn test_demixed_pca_basic() {
        let trains_a = make_trains();
        let trains_b: Vec<Vec<i32>> = (0..5)
            .map(|n| {
                let mut t = vec![0i32; 200];
                let step = 3 + n * 2;
                for i in (0..200).step_by(step) {
                    t[i] = 1;
                }
                t
            })
            .collect();
        let cond_a: Vec<&[i32]> = trains_a.iter().map(|t| t.as_slice()).collect();
        let cond_b: Vec<&[i32]> = trains_b.iter().map(|t| t.as_slice()).collect();
        let conditions = vec![cond_a, cond_b];
        let (proj, expl) = demixed_pca(&conditions, 2, 10);
        assert!(!expl.is_empty());
        assert!(!proj.is_empty());
    }

    #[test]
    fn test_demixed_pca_single_condition() {
        let t = vec![vec![1, 0, 1, 0]];
        let refs: Vec<&[i32]> = t.iter().map(|v| v.as_slice()).collect();
        let (proj, expl) = demixed_pca(&[refs], 2, 2);
        assert!(proj.is_empty());
        assert!(expl.is_empty());
    }

    #[test]
    fn test_factor_analysis_basic() {
        let trains = make_trains();
        let refs: Vec<&[i32]> = trains.iter().map(|t| t.as_slice()).collect();
        let (loadings, psi) = factor_analysis(&refs, 2, 10, 20);
        assert_eq!(loadings.len(), 5 * 2);
        assert_eq!(psi.len(), 5);
        // Uniquenesses should be positive
        assert!(psi.iter().all(|&p| p > 0.0));
    }

    #[test]
    fn test_factor_analysis_empty() {
        let (l, p) = factor_analysis(&[], 2, 10, 20);
        assert!(l.is_empty());
        assert!(p.is_empty());
    }

    #[test]
    fn test_symmetric_eigen_identity() {
        let eye = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let (vals, _) = symmetric_eigen(&eye, 3);
        for v in &vals {
            assert!((v - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_symmetric_eigen_known() {
        // [[2, 1], [1, 2]] -> eigenvalues 3, 1
        let a = vec![2.0, 1.0, 1.0, 2.0];
        let (vals, _) = symmetric_eigen(&a, 2);
        assert!((vals[0] - 3.0).abs() < 1e-10);
        assert!((vals[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_pca_explains_variance() {
        let trains = make_trains();
        let refs: Vec<&[i32]> = trains.iter().map(|t| t.as_slice()).collect();
        let (_, explained) = spike_train_pca(&refs, 5, 10);
        // All components should explain the full variance
        let total: f64 = explained.iter().sum();
        assert!(
            (total - 1.0).abs() < 0.05,
            "Total explained {total} should be ~1.0"
        );
    }
}
