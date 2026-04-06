// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Granger causality and directed connectivity measures

use rayon::prelude::*;
use std::f64::consts::PI;

use super::basic::bin_spike_train;

// ── Small-matrix linear algebra (real) ──────────────────────────────

/// Solve A x = b via Gaussian elimination with partial pivoting.
/// A is n×n (row-major flat), b is n-length. Returns x.
fn solve_linear(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    let mut aug = vec![0.0_f64; n * (n + 1)];
    for i in 0..n {
        for j in 0..n {
            aug[i * (n + 1) + j] = a[i * n + j];
        }
        aug[i * (n + 1) + n] = b[i];
    }
    let stride = n + 1;

    for col in 0..n {
        // Partial pivoting
        let mut max_row = col;
        let mut max_val = aug[col * stride + col].abs();
        for row in (col + 1)..n {
            let v = aug[row * stride + col].abs();
            if v > max_val {
                max_val = v;
                max_row = row;
            }
        }
        if max_row != col {
            for j in 0..stride {
                aug.swap(col * stride + j, max_row * stride + j);
            }
        }
        let pivot = aug[col * stride + col];
        if pivot.abs() < 1e-30 {
            continue;
        }
        for row in (col + 1)..n {
            let factor = aug[row * stride + col] / pivot;
            let mut j = col;
            let r_off = row * stride;
            let c_off = col * stride;
            while j + 3 < stride {
                aug[r_off + j] -= factor * aug[c_off + j];
                aug[r_off + j + 1] -= factor * aug[c_off + j + 1];
                aug[r_off + j + 2] -= factor * aug[c_off + j + 2];
                aug[r_off + j + 3] -= factor * aug[c_off + j + 3];
                j += 4;
            }
            while j < stride {
                aug[r_off + j] -= factor * aug[c_off + j];
                j += 1;
            }
        }
    }

    // Back substitution
    let mut x = vec![0.0_f64; n];
    for i in (0..n).rev() {
        let mut sum = aug[i * stride + n];
        for j in (i + 1)..n {
            sum -= aug[i * stride + j] * x[j];
        }
        let diag = aug[i * stride + i];
        x[i] = if diag.abs() > 1e-30 { sum / diag } else { 0.0 };
    }
    x
}

/// Solve A X = B where A is n×n and B is n×m. Returns X (n×m, row-major).
fn solve_matrix(a: &[f64], b: &[f64], n: usize, m: usize) -> Vec<f64> {
    let mut result = vec![0.0_f64; n * m];
    (0..m).into_par_iter().for_each(|col| {
        let rhs: Vec<f64> = (0..n).map(|i| b[i * m + col]).collect();
        let x = solve_linear(a, &rhs, n);
        // SAFETY: Each thread writes to unique indices based on col.
        unsafe {
            let ptr = result.as_ptr() as *mut f64;
            for i in 0..n {
                *ptr.add(i * m + col) = x[i];
            }
        }
    });
    result
}

// ── Small-matrix linear algebra (complex) ───────────────────────────

#[derive(Clone, Copy)]
struct C64 {
    re: f64,
    im: f64,
}

impl C64 {
    fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }
    fn zero() -> Self {
        Self { re: 0.0, im: 0.0 }
    }
    fn one() -> Self {
        Self { re: 1.0, im: 0.0 }
    }
    fn norm_sq(self) -> f64 {
        self.re * self.re + self.im * self.im
    }
    fn abs(self) -> f64 {
        self.norm_sq().sqrt()
    }
    fn conj(self) -> Self {
        Self {
            re: self.re,
            im: -self.im,
        }
    }
}

impl std::ops::Add for C64 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self {
            re: self.re + rhs.re,
            im: self.im + rhs.im,
        }
    }
}

impl std::ops::Sub for C64 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self {
            re: self.re - rhs.re,
            im: self.im - rhs.im,
        }
    }
}

impl std::ops::Mul for C64 {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Self {
            re: self.re * rhs.re - self.im * rhs.im,
            im: self.re * rhs.im + self.im * rhs.re,
        }
    }
}

impl std::ops::Mul<f64> for C64 {
    type Output = Self;
    fn mul(self, rhs: f64) -> Self {
        Self {
            re: self.re * rhs,
            im: self.im * rhs,
        }
    }
}

impl std::ops::AddAssign for C64 {
    fn add_assign(&mut self, rhs: Self) {
        self.re += rhs.re;
        self.im += rhs.im;
    }
}

impl std::ops::SubAssign for C64 {
    fn sub_assign(&mut self, rhs: Self) {
        self.re -= rhs.re;
        self.im -= rhs.im;
    }
}

/// Complex matrix multiply: C = A * B, all d×d row-major.
fn cmat_mul(a: &[C64], b: &[C64], d: usize) -> Vec<C64> {
    let mut c = vec![C64::zero(); d * d];
    for i in 0..d {
        for j in 0..d {
            let mut s = C64::zero();
            for k in 0..d {
                s += a[i * d + k] * b[k * d + j];
            }
            c[i * d + j] = s;
        }
    }
    c
}

/// Complex matrix inverse via Gauss-Jordan, d×d row-major.
fn cmat_inv(a: &[C64], d: usize) -> Option<Vec<C64>> {
    let mut aug = vec![C64::zero(); d * 2 * d];
    for i in 0..d {
        for j in 0..d {
            aug[i * 2 * d + j] = a[i * d + j];
        }
        aug[i * 2 * d + d + i] = C64::one();
    }
    let w = 2 * d;
    for col in 0..d {
        // Pivot
        let mut max_row = col;
        let mut max_val = aug[col * w + col].abs();
        for row in (col + 1)..d {
            let v = aug[row * w + col].abs();
            if v > max_val {
                max_val = v;
                max_row = row;
            }
        }
        if max_val < 1e-30 {
            return None;
        }
        if max_row != col {
            for j in 0..w {
                aug.swap(col * w + j, max_row * w + j);
            }
        }
        let pivot = aug[col * w + col];
        let inv_pivot = pivot.conj() * (1.0 / pivot.norm_sq());
        for j in 0..w {
            aug[col * w + j] = aug[col * w + j] * inv_pivot;
        }
        for row in 0..d {
            if row == col {
                continue;
            }
            let factor = aug[row * w + col];
            for j in 0..w {
                let sub = factor * aug[col * w + j];
                aug[row * w + j] -= sub;
            }
        }
    }
    let mut result = vec![C64::zero(); d * d];
    for i in 0..d {
        for j in 0..d {
            result[i * d + j] = aug[i * w + d + j];
        }
    }
    Some(result)
}

/// Complex matrix determinant, d×d.
fn cmat_det(a: &[C64], d: usize) -> C64 {
    if d == 1 {
        return a[0];
    }
    if d == 2 {
        return a[0] * a[3] - a[1] * a[2];
    }
    // LU-based via Gaussian elimination
    let mut m = a.to_vec();
    let mut det = C64::one();
    for col in 0..d {
        let mut max_row = col;
        let mut max_val = m[col * d + col].abs();
        for row in (col + 1)..d {
            let v = m[row * d + col].abs();
            if v > max_val {
                max_val = v;
                max_row = row;
            }
        }
        if max_val < 1e-30 {
            return C64::zero();
        }
        if max_row != col {
            for j in 0..d {
                m.swap(col * d + j, max_row * d + j);
            }
            det = det * (-1.0);
        }
        det = det * m[col * d + col];
        let pivot = m[col * d + col];
        let inv_pivot = pivot.conj() * (1.0 / pivot.norm_sq());
        for row in (col + 1)..d {
            let factor = m[row * d + col] * inv_pivot;
            for j in col..d {
                let sub = factor * m[col * d + j];
                m[row * d + j] -= sub;
            }
        }
    }
    det
}

/// Conjugate transpose of d×d complex matrix.
fn cmat_conj_t(a: &[C64], d: usize) -> Vec<C64> {
    let mut r = vec![C64::zero(); d * d];
    for i in 0..d {
        for j in 0..d {
            r[j * d + i] = a[i * d + j].conj();
        }
    }
    r
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

    // beta = (X^T X)^{-1} X^T Y
    let beta = solve_matrix(&xtx, &xty, x_cols, d);

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
    let beta = solve_linear(&xtx, &xty, x_cols);
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

        // A(f) = I - sum_k coeff_k * exp(-2πi f (k+1))
        let mut a_f = vec![C64::zero(); d * d];
        for i in 0..d {
            a_f[i * d + i] = C64::one();
        }
        for k in 0..order {
            let angle = -2.0 * PI * f * (k + 1) as f64;
            let exp_val = C64::new(angle.cos(), angle.sin());
            for i in 0..d {
                for j in 0..d {
                    // beta is (order*d × d), block k is rows [k*d..(k+1)*d], transposed
                    let coeff = beta[(k * d + j) * d + i]; // beta[k*d+j, i] → coeff_block.T[i,j]
                    a_f[i * d + j] -= C64::new(coeff, 0.0) * exp_val;
                }
            }
        }

        let det = cmat_det(&a_f, d);
        if det.abs() < 1e-30 {
            continue;
        }
        let h = match cmat_inv(&a_f, d) {
            Some(inv) => inv,
            None => continue,
        };

        // S = H Σ H*
        let sigma_c: Vec<C64> = sigma.iter().map(|&v| C64::new(v, 0.0)).collect();
        let h_conj_t = cmat_conj_t(&h, d);
        let tmp = cmat_mul(&h, &sigma_c, d);
        let s = cmat_mul(&tmp, &h_conj_t, d);

        for i in 0..d {
            for j in 0..d {
                if i == j {
                    continue;
                }
                let s_ii = s[i * d + i].abs();
                if s_ii > 1e-30 {
                    let h_ij_sq = h[i * d + j].norm_sq();
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

        let mut a_f = vec![C64::zero(); d * d];
        for i in 0..d {
            a_f[i * d + i] = C64::one();
        }
        for k in 0..order {
            let angle = -2.0 * PI * f * (k + 1) as f64;
            let exp_val = C64::new(angle.cos(), angle.sin());
            for i in 0..d {
                for j in 0..d {
                    let coeff = beta[(k * d + j) * d + i];
                    a_f[i * d + j] -= C64::new(coeff, 0.0) * exp_val;
                }
            }
        }

        for j in 0..d {
            let norm: f64 = (0..d).map(|i| a_f[i * d + j].norm_sq()).sum::<f64>().sqrt();
            if norm > 0.0 {
                for i in 0..d {
                    pdc[(i * d + j) * n_freqs + fi] = a_f[i * d + j].abs() / norm;
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

        let mut a_f = vec![C64::zero(); d * d];
        for i in 0..d {
            a_f[i * d + i] = C64::one();
        }
        for k in 0..order {
            let angle = -2.0 * PI * f * (k + 1) as f64;
            let exp_val = C64::new(angle.cos(), angle.sin());
            for i in 0..d {
                for j in 0..d {
                    let coeff = beta[(k * d + j) * d + i];
                    a_f[i * d + j] -= C64::new(coeff, 0.0) * exp_val;
                }
            }
        }

        let det = cmat_det(&a_f, d);
        if det.abs() < 1e-30 {
            continue;
        }
        let h = match cmat_inv(&a_f, d) {
            Some(inv) => inv,
            None => continue,
        };

        for i in 0..d {
            let norm: f64 = (0..d).map(|j| h[i * d + j].norm_sq()).sum::<f64>().sqrt();
            if norm > 0.0 {
                for j in 0..d {
                    dtf[(i * d + j) * n_freqs + fi] = h[i * d + j].abs() / norm;
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

    // ── linear algebra helpers ──────────────────────────────────────

    #[test]
    fn test_solve_linear_identity() {
        // I x = b → x = b
        let a = vec![1.0, 0.0, 0.0, 1.0];
        let b = vec![3.0, 7.0];
        let x = solve_linear(&a, &b, 2);
        assert!((x[0] - 3.0).abs() < 1e-10);
        assert!((x[1] - 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_solve_linear_2x2() {
        // [2 1; 1 3] x = [5; 10] → x = [1, 3]
        let a = vec![2.0, 1.0, 1.0, 3.0];
        let b = vec![5.0, 10.0];
        let x = solve_linear(&a, &b, 2);
        assert!((x[0] - 1.0).abs() < 1e-10);
        assert!((x[1] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_cmat_det_2x2() {
        let a = vec![
            C64::new(1.0, 0.0),
            C64::new(2.0, 0.0),
            C64::new(3.0, 0.0),
            C64::new(4.0, 0.0),
        ];
        let det = cmat_det(&a, 2);
        assert!((det.re - (-2.0)).abs() < 1e-10);
        assert!(det.im.abs() < 1e-10);
    }

    #[test]
    fn test_cmat_inv_identity() {
        let a = vec![C64::one(), C64::zero(), C64::zero(), C64::one()];
        let inv = cmat_inv(&a, 2).unwrap();
        assert!((inv[0].re - 1.0).abs() < 1e-10);
        assert!((inv[3].re - 1.0).abs() < 1e-10);
        assert!(inv[1].abs() < 1e-10);
        assert!(inv[2].abs() < 1e-10);
    }

    #[test]
    fn test_cmat_inv_roundtrip() {
        let a = vec![
            C64::new(2.0, 1.0),
            C64::new(1.0, 0.0),
            C64::new(0.0, 1.0),
            C64::new(3.0, 0.0),
        ];
        let inv = cmat_inv(&a, 2).unwrap();
        let prod = cmat_mul(&a, &inv, 2);
        // Should be identity
        assert!((prod[0].re - 1.0).abs() < 1e-8);
        assert!((prod[3].re - 1.0).abs() < 1e-8);
        assert!(prod[1].abs() < 1e-8);
        assert!(prod[2].abs() < 1e-8);
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
            assert_eq!(gc[0 * 16 + fi], 0.0, "GC[0,0] should be 0");
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
                v >= 0.0 && v <= 1.0 + 1e-10,
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
                v >= 0.0 && v <= 1.0 + 1e-10,
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
