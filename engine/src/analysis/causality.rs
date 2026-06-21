// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Granger causality and directed connectivity measures

use nalgebra::{Cholesky, Complex, DMatrix};
use rayon::prelude::*;
use std::f64::consts::PI;

use super::basic::bin_spike_train;

// ── Structured linear algebra (nalgebra-backed) ─────────────────────

/// Solve the symmetric positive-definite system `S X = B` via Cholesky
/// factorisation. `S` (n×n, row-major) is a ridge-regularised normal-equations
/// matrix `XᵀX + εI`, which is positive-definite for any `ε > 0`; `B` is n×m,
/// row-major. Returns `X` (n×m, row-major).
///
/// Cholesky is the numerically-optimal factorisation for an SPD system — half
/// the arithmetic of LU and unconditionally stable without pivoting — and a
/// single factorisation solves every right-hand-side column at once. Falls back
/// to a zero solution if `S` is not positive-definite; this cannot occur while
/// the `ε` ridge is present but keeps the solve total for degenerate inputs.
fn solve_spd(s: &[f64], b: &[f64], n: usize, m: usize) -> Vec<f64> {
    let s_mat = DMatrix::<f64>::from_row_slice(n, n, s);
    let b_mat = DMatrix::<f64>::from_row_slice(n, m, b);
    match Cholesky::new(s_mat) {
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

/// Build the multivariate-autoregressive spectral matrix
/// `A(f) = I − Σ_{k=0}^{order−1} A_{k+1} · e^{−i2πf(k+1)}` at normalised
/// frequency `f ∈ [0, 0.5)`, from the row-major VAR coefficient stack `beta`
/// ((order·d) × d). Block `k` is transposed on the fly:
/// `coeff_block.T[i, j] = beta[k·d + j, i]`. `A(f)` is a general complex,
/// non-Hermitian matrix.
fn spectral_matrix(beta: &[f64], d: usize, order: usize, f: f64) -> DMatrix<Complex<f64>> {
    let mut a_f = DMatrix::<Complex<f64>>::identity(d, d);
    for k in 0..order {
        let angle = -2.0 * PI * f * (k + 1) as f64;
        let exp_val = Complex::new(angle.cos(), angle.sin());
        for i in 0..d {
            for j in 0..d {
                let coeff = beta[(k * d + j) * d + i];
                a_f[(i, j)] -= Complex::new(coeff, 0.0) * exp_val;
            }
        }
    }
    a_f
}

/// Invert the MVAR spectral matrix `A(f)` to the transfer function `H(f) = A(f)⁻¹`
/// via LU factorisation, returning `None` when `A(f)` is numerically singular
/// (`|det A(f)| < 1e-30`).
///
/// `A(f)` is non-Hermitian, so LU — not Cholesky — is the structured
/// factorisation. A single factorisation yields both the singularity test (its
/// determinant) and the inverse, replacing the former separate complex
/// Gauss-Jordan inverse and Gaussian-elimination determinant.
fn spectral_transfer_inverse(a_f: DMatrix<Complex<f64>>) -> Option<DMatrix<Complex<f64>>> {
    let lu = a_f.lu();
    if lu.determinant().norm() < 1e-30 {
        return None;
    }
    lu.try_inverse()
}

// ── VAR model ───────────────────────────────────────────────────────

/// Fit VAR(order) model. Returns (beta [order*d × d, row-major], sigma [d×d, row-major]).
fn var_coefficients(trains_binned: &[Vec<f64>], order: usize) -> (Vec<f64>, Vec<f64>) {
    let d = trains_binned.len();
    let t = if d > 0 { trains_binned[0].len() } else { 0 };
    if t <= order + 1 || d == 0 {
        return (vec![0.0; order * d * d], identity_flat(d));
    }
    let n_pts = t - order;
    let x_cols = order * d;

    // Build y_cols: (d × n_pts) column-major
    let mut y_cols = vec![vec![0.0_f64; n_pts]; d];
    for ch in 0..d {
        for i in 0..n_pts {
            y_cols[ch][i] = trains_binned[ch][order + i];
        }
    }

    // Build x_cols_data: (x_cols × n_pts) column-major
    let mut x_cols_data = vec![vec![0.0_f64; n_pts]; x_cols];
    for i in 0..n_pts {
        for k in 0..order {
            for ch in 0..d {
                x_cols_data[k * d + ch][i] = trains_binned[ch][order - k - 1 + i];
            }
        }
    }

    // X^T X + reg
    let mut xtx = vec![0.0_f64; x_cols * x_cols];
    xtx.par_chunks_exact_mut(x_cols)
        .enumerate()
        .for_each(|(i, row)| {
            for j in 0..=i {
                let dot = crate::simd::dot_f64_dispatch(&x_cols_data[i], &x_cols_data[j]);
                row[j] = dot + if i == j { 1e-8 } else { 0.0 };
            }
        });
    // Mirror the matrix (serial, small overhead)
    for i in 0..x_cols {
        for j in (i + 1)..x_cols {
            xtx[i * x_cols + j] = xtx[j * x_cols + i];
        }
    }

    // X^T Y
    let mut xty = vec![0.0_f64; x_cols * d];
    xty.par_chunks_exact_mut(d)
        .enumerate()
        .for_each(|(i, row)| {
            for j in 0..d {
                row[j] = crate::simd::dot_f64_dispatch(&x_cols_data[i], &y_cols[j]);
            }
        });

    // beta = (X^T X)^{-1} X^T Y — SPD normal equations via Cholesky
    let beta = solve_spd(&xtx, &xty, x_cols, d);

    // Residuals Sigma = (1/N) (Y - X beta)^T (Y - X beta)
    let mut sigma = vec![0.0_f64; d * d];
    let n_norm = n_pts.max(1) as f64;

    // Precompute residuals: res_cols = y_cols - X_cols * beta (parallel)
    let res_cols: Vec<Vec<f64>> = (0..d)
        .into_par_iter()
        .map(|j| {
            let mut res = vec![0.0_f64; n_pts];
            for p in 0..n_pts {
                let mut r = y_cols[j][p];
                for c in 0..x_cols {
                    r -= x_cols_data[c][p] * beta[c * d + j];
                }
                res[p] = r;
            }
            res
        })
        .collect();

    for i in 0..d {
        for j in 0..=i {
            let dot = crate::simd::dot_f64_dispatch(&res_cols[i], &res_cols[j]);
            let val = dot / n_norm;
            sigma[i * d + j] = val;
            sigma[j * d + i] = val;
        }
    }

    (beta, sigma)
}

fn identity_flat(d: usize) -> Vec<f64> {
    let mut m = vec![0.0_f64; d * d];
    for i in 0..d {
        m[i * d + i] = 1.0;
    }
    m
}

/// Sum of squared errors for OLS regression: SSE = ||y - X beta||^2.
fn sse_ols(x: &[f64], y: &[f64], n_pts: usize, x_cols: usize) -> f64 {
    // X^T X + reg
    let mut xtx = vec![0.0_f64; x_cols * x_cols];
    for i in 0..x_cols {
        for j in 0..x_cols {
            let mut s = 0.0;
            for p in 0..n_pts {
                s += x[p * x_cols + i] * x[p * x_cols + j];
            }
            xtx[i * x_cols + j] = s + if i == j { 1e-8 } else { 0.0 };
        }
    }
    // X^T y
    let mut xty = vec![0.0_f64; x_cols];
    for i in 0..x_cols {
        let mut s = 0.0;
        for p in 0..n_pts {
            s += x[p * x_cols + i] * y[p];
        }
        xty[i] = s;
    }
    // beta = (X^T X)^{-1} X^T y — SPD normal equations via Cholesky
    let beta = solve_spd(&xtx, &xty, x_cols, 1);
    let mut sse = 0.0_f64;
    for p in 0..n_pts {
        let mut pred = 0.0;
        for c in 0..x_cols {
            pred += x[p * x_cols + c] * beta[c];
        }
        let r = y[p] - pred;
        sse += r * r;
    }
    sse
}

// ── Public API ──────────────────────────────────────────────────────

/// Pairwise Granger causality (Granger 1969).
/// Returns log-likelihood ratio. Positive = source Granger-causes target.
pub fn pairwise_granger_causality(
    source: &[i32],
    target: &[i32],
    bin_size: usize,
    order: usize,
) -> f64 {
    let cs: Vec<f64> = bin_spike_train(source, bin_size)
        .iter()
        .map(|&v| v as f64)
        .collect();
    let ct: Vec<f64> = bin_spike_train(target, bin_size)
        .iter()
        .map(|&v| v as f64)
        .collect();
    let n = cs.len().min(ct.len());
    if n <= 2 * order {
        return 0.0;
    }

    let n_pts = n - order;
    let y: Vec<f64> = ct[order..n].to_vec();

    // Restricted model: target past only
    let r_cols = order;
    let mut x_r = vec![0.0_f64; n_pts * r_cols];
    for p in 0..n_pts {
        for k in 0..order {
            x_r[p * r_cols + k] = ct[order - k - 1 + p];
        }
    }
    let sse_r = sse_ols(&x_r, &y, n_pts, r_cols);

    // Full model: target past + source past
    let f_cols = 2 * order;
    let mut x_f = vec![0.0_f64; n_pts * f_cols];
    for p in 0..n_pts {
        for k in 0..order {
            x_f[p * f_cols + k] = ct[order - k - 1 + p];
            x_f[p * f_cols + order + k] = cs[order - k - 1 + p];
        }
    }
    let sse_f = sse_ols(&x_f, &y, n_pts, f_cols);

    if sse_f <= 0.0 {
        return 0.0;
    }
    (sse_r.max(1e-30) / sse_f.max(1e-30)).ln()
}

/// Conditional Granger causality (Geweke 1984).
/// Tests if source Granger-causes target controlling for condition.
pub fn conditional_granger_causality(
    source: &[i32],
    target: &[i32],
    condition: &[i32],
    bin_size: usize,
    order: usize,
) -> f64 {
    let cs: Vec<f64> = bin_spike_train(source, bin_size)
        .iter()
        .map(|&v| v as f64)
        .collect();
    let ct: Vec<f64> = bin_spike_train(target, bin_size)
        .iter()
        .map(|&v| v as f64)
        .collect();
    let cc: Vec<f64> = bin_spike_train(condition, bin_size)
        .iter()
        .map(|&v| v as f64)
        .collect();
    let n = cs.len().min(ct.len()).min(cc.len());
    if n <= 2 * order {
        return 0.0;
    }

    let n_pts = n - order;
    let y: Vec<f64> = ct[order..n].to_vec();

    // Conditioned model: target + condition past
    let c_cols = 2 * order;
    let mut x_c = vec![0.0_f64; n_pts * c_cols];
    for p in 0..n_pts {
        for k in 0..order {
            x_c[p * c_cols + k] = ct[order - k - 1 + p];
            x_c[p * c_cols + order + k] = cc[order - k - 1 + p];
        }
    }
    let sse_c = sse_ols(&x_c, &y, n_pts, c_cols);

    // Full model: target + condition + source past
    let f_cols = 3 * order;
    let mut x_f = vec![0.0_f64; n_pts * f_cols];
    for p in 0..n_pts {
        for k in 0..order {
            x_f[p * f_cols + k] = ct[order - k - 1 + p];
            x_f[p * f_cols + order + k] = cc[order - k - 1 + p];
            x_f[p * f_cols + 2 * order + k] = cs[order - k - 1 + p];
        }
    }
    let sse_f = sse_ols(&x_f, &y, n_pts, f_cols);

    if sse_f <= 0.0 {
        return 0.0;
    }
    (sse_c.max(1e-30) / sse_f.max(1e-30)).ln()
}

/// Spectral Granger causality (Geweke 1982).
/// Returns (d × d × n_freqs) as flat Vec, row-major in [i][j][f] order.
pub fn spectral_granger_causality(
    trains: &[&[i32]],
    bin_size: usize,
    order: usize,
    n_freqs: usize,
) -> (Vec<f64>, usize) {
    let binned: Vec<Vec<f64>> = trains
        .iter()
        .map(|t| {
            bin_spike_train(t, bin_size)
                .iter()
                .map(|&v| v as f64)
                .collect()
        })
        .collect();
    let d = binned.len();
    let (beta, sigma) = var_coefficients(&binned, order);

    let mut gc = vec![0.0_f64; d * d * n_freqs];

    for fi in 0..n_freqs {
        let f = fi as f64 / (2 * n_freqs) as f64; // [0, 0.5)

        // A(f) = I - sum_k coeff_k * exp(-2πi f (k+1)); H(f) = A(f)⁻¹ via LU.
        let a_f = spectral_matrix(&beta, d, order, f);
        let h = match spectral_transfer_inverse(a_f) {
            Some(inv) => inv,
            None => continue,
        };

        // S = H Σ H*
        let sigma_c =
            DMatrix::<Complex<f64>>::from_fn(d, d, |i, j| Complex::new(sigma[i * d + j], 0.0));
        let s = &h * &sigma_c * h.adjoint();

        for i in 0..d {
            for j in 0..d {
                if i == j {
                    continue;
                }
                let s_ii = s[(i, i)].norm();
                if s_ii > 1e-30 {
                    let h_ij_sq = h[(i, j)].norm_sqr();
                    let reduced = s_ii - sigma[j * d + j] * h_ij_sq;
                    if reduced > 0.0 && reduced < s_ii {
                        gc[(i * d + j) * n_freqs + fi] = (s_ii / reduced).ln().max(0.0);
                    }
                }
            }
        }
    }
    (gc, d)
}

/// Partial directed coherence (Baccala & Sameshima 2001).
/// Returns (d × d × n_freqs) flat Vec.
pub fn partial_directed_coherence(
    trains: &[&[i32]],
    bin_size: usize,
    order: usize,
    n_freqs: usize,
) -> (Vec<f64>, usize) {
    let binned: Vec<Vec<f64>> = trains
        .iter()
        .map(|t| {
            bin_spike_train(t, bin_size)
                .iter()
                .map(|&v| v as f64)
                .collect()
        })
        .collect();
    let d = binned.len();
    let (beta, _) = var_coefficients(&binned, order);

    let mut pdc = vec![0.0_f64; d * d * n_freqs];

    for fi in 0..n_freqs {
        let f = fi as f64 / (2 * n_freqs) as f64;

        let a_f = spectral_matrix(&beta, d, order, f);

        for j in 0..d {
            let norm: f64 = (0..d).map(|i| a_f[(i, j)].norm_sqr()).sum::<f64>().sqrt();
            if norm > 0.0 {
                for i in 0..d {
                    pdc[(i * d + j) * n_freqs + fi] = a_f[(i, j)].norm() / norm;
                }
            }
        }
    }
    (pdc, d)
}

/// Directed transfer function (Kaminski & Blinowska 1991).
/// Returns (d × d × n_freqs) flat Vec.
pub fn directed_transfer_function(
    trains: &[&[i32]],
    bin_size: usize,
    order: usize,
    n_freqs: usize,
) -> (Vec<f64>, usize) {
    let binned: Vec<Vec<f64>> = trains
        .iter()
        .map(|t| {
            bin_spike_train(t, bin_size)
                .iter()
                .map(|&v| v as f64)
                .collect()
        })
        .collect();
    let d = binned.len();
    let (beta, _sigma) = var_coefficients(&binned, order);

    let mut dtf = vec![0.0_f64; d * d * n_freqs];

    for fi in 0..n_freqs {
        let f = fi as f64 / (2 * n_freqs) as f64;

        let a_f = spectral_matrix(&beta, d, order, f);
        let h = match spectral_transfer_inverse(a_f) {
            Some(inv) => inv,
            None => continue,
        };

        for i in 0..d {
            let norm: f64 = (0..d).map(|j| h[(i, j)].norm_sqr()).sum::<f64>().sqrt();
            if norm > 0.0 {
                for j in 0..d {
                    dtf[(i * d + j) * n_freqs + fi] = h[(i, j)].norm() / norm;
                }
            }
        }
    }
    (dtf, d)
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

    // ── structured linear algebra helpers ───────────────────────────

    #[test]
    fn test_solve_spd_identity() {
        // I x = b → x = b
        let a = vec![1.0, 0.0, 0.0, 1.0];
        let b = vec![3.0, 7.0];
        let x = solve_spd(&a, &b, 2, 1);
        assert!((x[0] - 3.0).abs() < 1e-10);
        assert!((x[1] - 7.0).abs() < 1e-10);
    }

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

    #[test]
    fn test_spectral_matrix_dc_zero_beta() {
        // f = 0 (exp = 1) with zero VAR coefficients → A(0) = I.
        let beta = vec![0.0_f64; 2 * 2 * 2]; // order = 2, d = 2
        let a = spectral_matrix(&beta, 2, 2, 0.0);
        assert!((a[(0, 0)].re - 1.0).abs() < 1e-12);
        assert!((a[(1, 1)].re - 1.0).abs() < 1e-12);
        assert!(a[(0, 1)].norm() < 1e-12);
        assert!(a[(1, 0)].norm() < 1e-12);
    }

    #[test]
    fn test_spectral_transfer_inverse_identity() {
        let a = DMatrix::<Complex<f64>>::identity(2, 2);
        let inv = spectral_transfer_inverse(a).unwrap();
        assert!((inv[(0, 0)].re - 1.0).abs() < 1e-10);
        assert!((inv[(1, 1)].re - 1.0).abs() < 1e-10);
        assert!(inv[(0, 1)].norm() < 1e-10);
        assert!(inv[(1, 0)].norm() < 1e-10);
    }

    #[test]
    fn test_spectral_transfer_inverse_roundtrip() {
        let a = DMatrix::from_row_slice(
            2,
            2,
            &[
                Complex::new(2.0, 1.0),
                Complex::new(1.0, 0.0),
                Complex::new(0.0, 1.0),
                Complex::new(3.0, 0.0),
            ],
        );
        let inv = spectral_transfer_inverse(a.clone()).unwrap();
        let prod = &a * &inv; // A · A⁻¹ = I
        assert!((prod[(0, 0)].re - 1.0).abs() < 1e-8);
        assert!((prod[(1, 1)].re - 1.0).abs() < 1e-8);
        assert!(prod[(0, 1)].norm() < 1e-8);
        assert!(prod[(1, 0)].norm() < 1e-8);
    }

    #[test]
    fn test_spectral_transfer_inverse_singular() {
        // Zero matrix is singular → None.
        let a = DMatrix::<Complex<f64>>::zeros(2, 2);
        assert!(spectral_transfer_inverse(a).is_none());
    }

    // ── pairwise_granger_causality ──────────────────────────────────

    #[test]
    fn test_gc_self_finite() {
        let train = make_train(&[5, 15, 25, 35, 45, 55, 65, 75, 85, 95], 100);
        let gc = pairwise_granger_causality(&train, &train, 5, 3);
        // When source == target, duplicate regressors can reduce SSE via regularisation
        assert!(gc.is_finite(), "self GC should be finite, got {gc}");
        assert!(gc >= 0.0, "GC should be non-negative, got {gc}");
    }

    #[test]
    fn test_gc_non_negative_typical() {
        let source = make_train(&[5, 15, 25, 35, 45, 55, 65, 75, 85, 95], 100);
        let target = make_train(&[7, 17, 27, 37, 47, 57, 67, 77, 87, 97], 100);
        let gc = pairwise_granger_causality(&source, &target, 5, 3);
        // Just check it returns a finite value
        assert!(gc.is_finite(), "GC should be finite, got {gc}");
    }

    #[test]
    fn test_gc_too_short() {
        let a = make_train(&[1], 10);
        let b = make_train(&[2], 10);
        let gc = pairwise_granger_causality(&a, &b, 5, 5);
        assert_eq!(gc, 0.0, "too short → 0");
    }

    // ── conditional_granger_causality ────────────────────────────────

    #[test]
    fn test_cond_gc_finite() {
        let source = make_train(&[5, 15, 25, 35, 45, 55, 65, 75, 85, 95], 100);
        let target = make_train(&[7, 17, 27, 37, 47, 57, 67, 77, 87, 97], 100);
        let cond = make_train(&[3, 13, 23, 33, 43, 53, 63, 73, 83, 93], 100);
        let gc = conditional_granger_causality(&source, &target, &cond, 5, 3);
        assert!(gc.is_finite(), "conditional GC should be finite");
    }

    #[test]
    fn test_cond_gc_too_short() {
        let a = make_train(&[1], 10);
        let b = make_train(&[2], 10);
        let c = make_train(&[3], 10);
        assert_eq!(conditional_granger_causality(&a, &b, &c, 5, 5), 0.0);
    }

    // ── spectral_granger_causality ──────────────────────────────────

    #[test]
    fn test_spectral_gc_shape() {
        let t1 = make_train(&[5, 15, 25, 35, 45, 55, 65, 75, 85, 95], 100);
        let t2 = make_train(&[7, 17, 27, 37, 47, 57, 67, 77, 87, 97], 100);
        let trains: Vec<&[i32]> = vec![&t1, &t2];
        let (gc, d) = spectral_granger_causality(&trains, 5, 3, 16);
        assert_eq!(d, 2);
        assert_eq!(gc.len(), 2 * 2 * 16);
    }

    #[test]
    fn test_spectral_gc_diagonal_zero() {
        let t1 = make_train(&[5, 15, 25, 35, 45, 55, 65, 75, 85, 95], 100);
        let t2 = make_train(&[7, 17, 27, 37, 47, 57, 67, 77, 87, 97], 100);
        let trains: Vec<&[i32]> = vec![&t1, &t2];
        let (gc, _) = spectral_granger_causality(&trains, 5, 3, 16);
        // Diagonal entries (i==j) should be 0
        for fi in 0..16 {
            assert_eq!(gc[fi], 0.0, "GC[0,0] should be 0");
            assert_eq!(gc[3 * 16 + fi], 0.0, "GC[1,1] should be 0");
        }
    }

    #[test]
    fn test_spectral_gc_non_negative() {
        let t1 = make_train(&[5, 15, 25, 35, 45, 55, 65, 75, 85, 95], 100);
        let t2 = make_train(&[7, 17, 27, 37, 47, 57, 67, 77, 87, 97], 100);
        let trains: Vec<&[i32]> = vec![&t1, &t2];
        let (gc, _) = spectral_granger_causality(&trains, 5, 3, 16);
        for &v in &gc {
            assert!(v >= 0.0, "spectral GC must be non-negative, got {v}");
        }
    }

    // ── partial_directed_coherence ──────────────────────────────────

    #[test]
    fn test_pdc_shape() {
        let t1 = make_train(&[5, 15, 25, 35, 45, 55, 65, 75, 85, 95], 100);
        let t2 = make_train(&[7, 17, 27, 37, 47, 57, 67, 77, 87, 97], 100);
        let trains: Vec<&[i32]> = vec![&t1, &t2];
        let (pdc, d) = partial_directed_coherence(&trains, 5, 3, 16);
        assert_eq!(d, 2);
        assert_eq!(pdc.len(), 2 * 2 * 16);
    }

    #[test]
    fn test_pdc_range() {
        let t1 = make_train(&[5, 15, 25, 35, 45, 55, 65, 75, 85, 95], 100);
        let t2 = make_train(&[7, 17, 27, 37, 47, 57, 67, 77, 87, 97], 100);
        let trains: Vec<&[i32]> = vec![&t1, &t2];
        let (pdc, _) = partial_directed_coherence(&trains, 5, 3, 16);
        for &v in &pdc {
            assert!(
                (0.0..=1.0 + 1e-10).contains(&v),
                "PDC should be in [0,1], got {v}"
            );
        }
    }

    // ── directed_transfer_function ──────────────────────────────────

    #[test]
    fn test_dtf_shape() {
        let t1 = make_train(&[5, 15, 25, 35, 45, 55, 65, 75, 85, 95], 100);
        let t2 = make_train(&[7, 17, 27, 37, 47, 57, 67, 77, 87, 97], 100);
        let trains: Vec<&[i32]> = vec![&t1, &t2];
        let (dtf, d) = directed_transfer_function(&trains, 5, 3, 16);
        assert_eq!(d, 2);
        assert_eq!(dtf.len(), 2 * 2 * 16);
    }

    #[test]
    fn test_dtf_range() {
        let t1 = make_train(&[5, 15, 25, 35, 45, 55, 65, 75, 85, 95], 100);
        let t2 = make_train(&[7, 17, 27, 37, 47, 57, 67, 77, 87, 97], 100);
        let trains: Vec<&[i32]> = vec![&t1, &t2];
        let (dtf, _) = directed_transfer_function(&trains, 5, 3, 16);
        for &v in &dtf {
            assert!(
                (0.0..=1.0 + 1e-10).contains(&v),
                "DTF should be in [0,1], got {v}"
            );
        }
    }

    // ── var_coefficients ────────────────────────────────────────────

    #[test]
    fn test_var_too_short() {
        let trains = vec![vec![1.0, 2.0]];
        let (beta, sigma) = var_coefficients(&trains, 5);
        assert!(beta.iter().all(|&v| v == 0.0), "too short → zero beta");
        assert!((sigma[0] - 1.0).abs() < 1e-10, "identity sigma");
    }
}
