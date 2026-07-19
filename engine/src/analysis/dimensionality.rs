// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Dimensionality reduction for spike train populations

use super::basic;
use nalgebra::{DMatrix, SymmetricEigen};

// ── linear-algebra helpers ──────────────────────────────────────────

/// Descending eigenvalues and sign-canonicalised eigenvectors of a symmetric
/// matrix `a` (row-major `n × n`) via nalgebra's symmetric eigensolver
/// (tridiagonalisation + implicit QR — LAPACK-grade, replacing a hand-rolled
/// Jacobi sweep). Each eigenvector column is sign-fixed so its largest-magnitude
/// entry is positive, making downstream projections deterministic across
/// backends. Eigenvectors are returned row-major (`vecs[row * n + col]`).
fn symmetric_eigen(a: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    let se = SymmetricEigen::new(DMatrix::<f64>::from_row_slice(n, n, a));
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&i, &j| se.eigenvalues[j].partial_cmp(&se.eigenvalues[i]).unwrap());

    let vals: Vec<f64> = idx.iter().map(|&i| se.eigenvalues[i]).collect();
    let mut vecs = vec![0.0f64; n * n];
    for (new_col, &old_col) in idx.iter().enumerate() {
        // Sign convention: make the largest-magnitude entry positive.
        let mut pivot = 0usize;
        let mut max_abs = 0.0f64;
        for r in 0..n {
            let v = se.eigenvectors[(r, old_col)].abs();
            if v > max_abs {
                max_abs = v;
                pivot = r;
            }
        }
        let sign = if se.eigenvectors[(pivot, old_col)] < 0.0 {
            -1.0
        } else {
            1.0
        };
        for r in 0..n {
            vecs[r * n + new_col] = sign * se.eigenvectors[(r, old_col)];
        }
    }
    (vals, vecs)
}

/// Solve the SPD system `A X = B` via Cholesky. `a` is row-major `n × n`, `b` is
/// row-major `n × k`; returns `X` row-major `n × k`. `A` is never inverted.
fn spd_solve(a: &[f64], n: usize, b: &[f64], k: usize) -> Vec<f64> {
    let chol = DMatrix::<f64>::from_row_slice(n, n, a)
        .cholesky()
        .expect("factor-analysis system must be symmetric positive-definite");
    let solved = chol.solve(&DMatrix::<f64>::from_row_slice(n, k, b));
    let mut out = vec![0.0f64; n * k];
    for i in 0..n {
        for j in 0..k {
            out[i * k + j] = solved[(i, j)];
        }
    }
    out
}

/// SPD inverse via Cholesky (needed for the explicit `nf · M⁻¹` term).
fn spd_inverse(a: &[f64], n: usize) -> Vec<f64> {
    let inv = DMatrix::<f64>::from_row_slice(n, n, a)
        .cholesky()
        .expect("factor-analysis system must be symmetric positive-definite")
        .inverse();
    let mut out = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            out[i * n + j] = inv[(i, j)];
        }
    }
    out
}

// ── centred-matrix cores ────────────────────────────────────────────

/// PCA of a mean-centred matrix `mat` (`d × t`, row-major).
/// Returns `(projected [nc × t], explained_variance_ratio [nc])`.
pub fn pca_from_centered(
    mat: &[f64],
    d: usize,
    t: usize,
    n_components: usize,
) -> (Vec<f64>, Vec<f64>) {
    let denom = (t - 1).max(1) as f64;
    let mut cov = vec![0.0f64; d * d];
    for i in 0..d {
        for j in i..d {
            let mut s = 0.0;
            for k in 0..t {
                s += mat[i * t + k] * mat[j * t + k];
            }
            s /= denom;
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
    let mut projected = vec![0.0f64; nc * t];
    for c in 0..nc {
        for tt in 0..t {
            let mut s = 0.0;
            for i in 0..d {
                s += eigvecs[i * d + c] * mat[i * t + tt];
            }
            projected[c * t + tt] = s;
        }
    }
    (projected, explained)
}

/// Demixed PCA of a grand-mean-centred condition-mean matrix (`n_cond × t`).
/// Returns `(projected [n_cond × nc], explained_variance_ratio [nc])`.
pub fn demixed_from_centered(
    mean_mat: &[f64],
    n_cond: usize,
    t: usize,
    n_components: usize,
) -> (Vec<f64>, Vec<f64>) {
    let denom = n_cond as f64;
    let mut cov = vec![0.0f64; t * t];
    for i in 0..t {
        for j in i..t {
            let mut s = 0.0;
            for c in 0..n_cond {
                s += mean_mat[c * t + i] * mean_mat[c * t + j];
            }
            s /= denom;
            cov[i * t + j] = s;
            cov[j * t + i] = s;
        }
    }
    let (eigvals, eigvecs) = symmetric_eigen(&cov, t);
    let nc = n_components.min(t);
    let total: f64 = eigvals.iter().sum();
    let explained: Vec<f64> = eigvals[..nc]
        .iter()
        .map(|&v| if total > 0.0 { v / total } else { v })
        .collect();
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

/// Factor-analysis EM of a mean-centred matrix `mat` (`d × t`).
///
/// The loadings start from a deterministic PCA initialisation (top eigenvectors
/// of the sample covariance scaled by `sqrt` of the eigenvalues, sign-fixed), and
/// each EM step solves its SPD `M` and `E[zzᵀ]` systems by Cholesky.
/// Returns `(loadings [d × nf], uniquenesses [d])`.
pub fn fa_from_centered(
    mat: &[f64],
    d: usize,
    t: usize,
    n_factors: usize,
    n_iter: usize,
) -> (Vec<f64>, Vec<f64>) {
    let tf = t as f64;
    let mut cov = vec![0.0f64; d * d];
    for i in 0..d {
        for j in i..d {
            let mut s = 0.0;
            for k in 0..t {
                s += mat[i * t + k] * mat[j * t + k];
            }
            s /= tf;
            cov[i * d + j] = s;
            cov[j * d + i] = s;
        }
    }
    let nf = n_factors.min(d);
    let (eigvals, eigvecs) = symmetric_eigen(&cov, d);
    // Deterministic PCA initialisation (sign-canon preserved by the positive scale).
    let mut loadings = vec![0.0f64; d * nf];
    for c in 0..nf {
        let scale = eigvals[c].max(0.0).sqrt();
        for i in 0..d {
            loadings[i * nf + c] = eigvecs[i * d + c] * scale;
        }
    }
    let mut psi: Vec<f64> = (0..d).map(|i| cov[i * d + i]).collect();

    for _ in 0..n_iter {
        let psi_inv: Vec<f64> = psi.iter().map(|&p| 1.0 / (p + 1e-10)).collect();

        // M = Lᵀ diag(psi_inv) L + I  (nf × nf)
        let mut m = vec![0.0f64; nf * nf];
        for a in 0..nf {
            for b in 0..nf {
                let mut s = 0.0;
                for i in 0..d {
                    s += loadings[i * nf + a] * psi_inv[i] * loadings[i * nf + b];
                }
                m[a * nf + b] = s + if a == b { 1.0 } else { 0.0 };
            }
        }
        let m_inv = spd_inverse(&m, nf);

        // beta = M⁻¹ (Lᵀ diag(psi_inv))  (nf × d)
        let mut beta = vec![0.0f64; nf * d];
        for a in 0..nf {
            for i in 0..d {
                let mut s = 0.0;
                for kk in 0..nf {
                    s += m_inv[a * nf + kk] * loadings[i * nf + kk] * psi_inv[i];
                }
                beta[a * d + i] = s;
            }
        }

        // E[z] = beta mat  (nf × t)
        let mut ez = vec![0.0f64; nf * t];
        for a in 0..nf {
            for tt in 0..t {
                let mut s = 0.0;
                for i in 0..d {
                    s += beta[a * d + i] * mat[i * t + tt];
                }
                ez[a * t + tt] = s;
            }
        }

        // E[zzᵀ] = nf M⁻¹ + ez ezᵀ / t  (nf × nf)
        let mut ezzt = vec![0.0f64; nf * nf];
        for a in 0..nf {
            for b in 0..nf {
                let mut s = 0.0;
                for tt in 0..t {
                    s += ez[a * t + tt] * ez[b * t + tt];
                }
                ezzt[a * nf + b] = nf as f64 * m_inv[a * nf + b] + s / tf;
            }
        }

        // mat_ez_t = mat ezᵀ / t  (d × nf)
        let mut mat_ez_t = vec![0.0f64; d * nf];
        for i in 0..d {
            for a in 0..nf {
                let mut s = 0.0;
                for tt in 0..t {
                    s += mat[i * t + tt] * ez[a * t + tt];
                }
                mat_ez_t[i * nf + a] = s / tf;
            }
        }

        // loadings = mat_ez_t E[zzᵀ]⁻¹  via solving  E[zzᵀ] Xᵀ = mat_ez_tᵀ.
        let mut rhs = vec![0.0f64; nf * d];
        for a in 0..nf {
            for i in 0..d {
                rhs[a * d + i] = mat_ez_t[i * nf + a];
            }
        }
        let solved = spd_solve(&ezzt, nf, &rhs, d); // (nf × d) = E[zzᵀ]⁻¹ rhs
        for i in 0..d {
            for a in 0..nf {
                loadings[i * nf + a] = solved[a * d + i];
            }
        }

        // psi = diag(cov - loadings ez matᵀ / t)
        for i in 0..d {
            let mut s = 0.0;
            for tt in 0..t {
                let mut l_ez = 0.0;
                for a in 0..nf {
                    l_ez += loadings[i * nf + a] * ez[a * t + tt];
                }
                s += l_ez * mat[i * t + tt];
            }
            psi[i] = (cov[i * d + i] - s / tf).max(1e-6);
        }
    }

    (loadings, psi)
}

// ── binning wrappers (raw trains → centred matrix → core) ───────────

fn binned_centred(trains: &[&[i32]], bin_size: usize) -> (Vec<f64>, usize, usize) {
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
    let d = trains.len();
    if min_bins == 0 {
        return (vec![], d, 0);
    }
    let mut mat = vec![0.0f64; d * min_bins];
    for i in 0..d {
        let mean: f64 = binned[i][..min_bins].iter().sum::<f64>() / min_bins as f64;
        for j in 0..min_bins {
            mat[i * min_bins + j] = binned[i][j] - mean;
        }
    }
    (mat, d, min_bins)
}

/// PCA on binned spike trains. `trains`: list of binary trains.
pub fn spike_train_pca(
    trains: &[&[i32]],
    n_components: usize,
    bin_size: usize,
) -> (Vec<f64>, Vec<f64>) {
    if trains.is_empty() {
        return (vec![], vec![]);
    }
    let (mat, d, min_bins) = binned_centred(trains, bin_size);
    if min_bins == 0 {
        return (vec![], vec![]);
    }
    if d < 2 {
        return (mat[..min_bins].to_vec(), vec![1.0]);
    }
    pca_from_centered(&mat, d, min_bins, n_components)
}

/// Demixed PCA. Kobak et al. 2016. `conditions`: list of (condition trains).
pub fn demixed_pca(
    conditions: &[Vec<&[i32]>],
    n_components: usize,
    bin_size: usize,
) -> (Vec<f64>, Vec<f64>) {
    if conditions.len() < 2 {
        return (vec![], vec![]);
    }
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
            for (j, m) in mean.iter_mut().enumerate() {
                *m += b[j];
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
    let mut grand = vec![0.0f64; min_bins];
    for m in &all_means {
        for (j, g) in grand.iter_mut().enumerate() {
            *g += m[j];
        }
    }
    for v in &mut grand {
        *v /= n_cond as f64;
    }
    let mut mean_mat = vec![0.0f64; n_cond * min_bins];
    for (i, m) in all_means.iter().enumerate() {
        for j in 0..min_bins {
            mean_mat[i * min_bins + j] = m[j] - grand[j];
        }
    }
    demixed_from_centered(&mean_mat, n_cond, min_bins, n_components)
}

/// Factor analysis via EM. Rubin & Thayer 1982.
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
    let (mat, _d, t) = binned_centred(trains, bin_size);
    if t == 0 {
        return (vec![0.0; d * n_factors.min(d)], vec![1.0; d]);
    }
    fa_from_centered(&mat, d, t, n_factors, n_iter)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_trains() -> Vec<Vec<i32>> {
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
        let total: f64 = explained.iter().sum();
        assert!(total <= 1.0 + 1e-6, "Total explained {total} > 1");
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
        let t = [vec![1, 0, 1, 0]];
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
        // [[2, 1], [1, 2]] -> eigenvalues 3, 1 (descending)
        let a = vec![2.0, 1.0, 1.0, 2.0];
        let (vals, _) = symmetric_eigen(&a, 2);
        assert!((vals[0] - 3.0).abs() < 1e-10);
        assert!((vals[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_symmetric_eigen_sign_canonical() {
        // The dominant entry of each eigenvector column must be positive.
        let a = vec![2.0, 1.0, 1.0, 2.0];
        let (_, vecs) = symmetric_eigen(&a, 2);
        for c in 0..2 {
            let mut pivot = 0usize;
            let mut max_abs = 0.0f64;
            for r in 0..2 {
                if vecs[r * 2 + c].abs() > max_abs {
                    max_abs = vecs[r * 2 + c].abs();
                    pivot = r;
                }
            }
            assert!(vecs[pivot * 2 + c] > 0.0, "column {c} not sign-canonical");
        }
    }

    #[test]
    fn test_spd_solve_matches_inverse() {
        // SPD 2x2; solving A x = b equals A⁻¹ b.
        let a = vec![4.0, 1.0, 1.0, 3.0];
        let b = vec![1.0, 2.0];
        let x = spd_solve(&a, 2, &b, 1);
        let det = 4.0 * 3.0 - 1.0 * 1.0;
        let ref0 = (3.0 * 1.0 - 1.0 * 2.0) / det;
        let ref1 = (-1.0 + 4.0 * 2.0) / det;
        assert!((x[0] - ref0).abs() < 1e-12 && (x[1] - ref1).abs() < 1e-12);
    }

    #[test]
    fn test_pca_explains_variance() {
        let trains = make_trains();
        let refs: Vec<&[i32]> = trains.iter().map(|t| t.as_slice()).collect();
        let (_, explained) = spike_train_pca(&refs, 5, 10);
        let total: f64 = explained.iter().sum();
        assert!(
            (total - 1.0).abs() < 0.05,
            "Total explained {total} should be ~1.0"
        );
    }
}
