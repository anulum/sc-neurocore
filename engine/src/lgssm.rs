// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust LGSSM Kalman filter (parity with src/sc_neurocore/world_model/predictive_model.py)

//! Rust implementation of the Kalman filter forward pass for a
//! linear Gaussian state-space model.
//!
//! Match for `KalmanFilter.filter()` in
//! `src/sc_neurocore/world_model/predictive_model.py` so that the
//! Python and Rust paths return identical (within float64
//! round-off) means, covariances, and log-likelihood.
//!
//! References match the Python module:
//!   Kalman 1960; Bishop 2006 §13.3.1.
//!
//! Algorithm per timestep t:
//!   x_pred = A x_filt + B u
//!   P_pred = A P_filt A^T + Q
//!   e_t = y_t - C x_pred - D u
//!   S = C P_pred C^T + R
//!   K = P_pred C^T S^{-1}
//!   x_filt = x_pred + K e_t
//!   P_filt = (I - K C) P_pred (I - K C)^T + K R K^T   (Joseph form)
//!   log-lik += -0.5 * (p log 2pi + log |S| + e_t^T S^{-1} e_t)

use ndarray::{s, Array1, Array2, Array3, ArrayView1, ArrayView2};

/// Result of the Kalman filter forward pass.
pub struct KalmanResult {
    pub means: Array2<f64>,           // (T, d) filtered means
    pub covariances: Array3<f64>,     // (T, d, d) filtered covariances
    pub pred_means: Array2<f64>,      // (T, d) predicted means
    pub pred_covariances: Array3<f64>,// (T, d, d) predicted covariances
    pub log_likelihood: f64,
}

/// Run the forward Kalman filter on a sequence of observations.
///
/// All input matrices are dense `Array2<f64>`. Controls may be
/// empty (shape (T, 0)) when the model has no control input.
///
/// # Panics
///
/// Panics if dimensions are inconsistent. The caller (PyO3
/// wrapper) validates shapes before invoking.
pub fn kalman_filter(
    observations: ArrayView2<f64>,        // (T, p)
    controls: ArrayView2<f64>,            // (T, m) or (T, 0)
    a: ArrayView2<f64>,                   // (d, d)
    b: ArrayView2<f64>,                   // (d, m) or (d, 0)
    c: ArrayView2<f64>,                   // (p, d)
    d: ArrayView2<f64>,                   // (p, m) or (p, 0)
    q: ArrayView2<f64>,                   // (d, d)
    r: ArrayView2<f64>,                   // (p, p)
    mu_0: ArrayView1<f64>,                // (d,)
    sigma_0: ArrayView2<f64>,             // (d, d)
) -> KalmanResult {
    let t_len = observations.nrows();
    let p_dim = observations.ncols();
    let d_dim = a.nrows();
    let m_dim = b.ncols();

    let has_control = m_dim > 0;

    let mut means = Array2::<f64>::zeros((t_len, d_dim));
    let mut covs = Array3::<f64>::zeros((t_len, d_dim, d_dim));
    let mut pred_means = Array2::<f64>::zeros((t_len, d_dim));
    let mut pred_covs = Array3::<f64>::zeros((t_len, d_dim, d_dim));

    let mut x_pred: Array1<f64> = mu_0.to_owned();
    let mut p_pred: Array2<f64> = sigma_0.to_owned();

    let mut log_lik = 0.0_f64;
    let two_pi_log = (2.0 * std::f64::consts::PI).ln();
    let i_d = Array2::<f64>::eye(d_dim);

    for t in 0..t_len {
        // Record predicted state for this step
        pred_means.slice_mut(s![t, ..]).assign(&x_pred);
        pred_covs.slice_mut(s![t, .., ..]).assign(&p_pred);

        let y_t = observations.slice(s![t, ..]);

        // Innovation: e = y - C x_pred - D u
        let mut y_hat = c.dot(&x_pred);
        if has_control {
            let u_t = controls.slice(s![t, ..]);
            y_hat = y_hat + d.dot(&u_t);
        }
        let innov = &y_t - &y_hat;

        // Innovation covariance: S = C P_pred C^T + R
        let s_mat = c.dot(&p_pred).dot(&c.t()) + &r;

        // Solve S z = innov using Gaussian elimination (small p)
        let s_inv = invert_psd_matrix(&s_mat);
        let s_inv_innov = s_inv.dot(&innov);

        // Log-determinant of S via LU/Cholesky-style recursion
        let logdet_s = log_det_psd(&s_mat);
        let quad_form = innov.dot(&s_inv_innov);
        log_lik +=
            -0.5 * (p_dim as f64 * two_pi_log + logdet_s + quad_form);

        // Kalman gain: K = P_pred C^T S^{-1}
        let k_gain = p_pred.dot(&c.t()).dot(&s_inv);

        // Filtered state: x_filt = x_pred + K e
        let x_filt = &x_pred + &k_gain.dot(&innov);

        // Joseph form for filtered covariance:
        //   P_filt = (I - K C) P_pred (I - K C)^T + K R K^T
        let i_minus_kc = &i_d - &k_gain.dot(&c);
        let p_filt =
            i_minus_kc.dot(&p_pred).dot(&i_minus_kc.t()) + k_gain.dot(&r).dot(&k_gain.t());

        means.slice_mut(s![t, ..]).assign(&x_filt);
        covs.slice_mut(s![t, .., ..]).assign(&p_filt);

        // Predict next state
        let mut x_next = a.dot(&x_filt);
        if has_control {
            let u_t = controls.slice(s![t, ..]);
            x_next = x_next + b.dot(&u_t);
        }
        let p_next = a.dot(&p_filt).dot(&a.t()) + &q;
        x_pred = x_next;
        p_pred = p_next;
    }

    KalmanResult {
        means,
        covariances: covs,
        pred_means,
        pred_covariances: pred_covs,
        log_likelihood: log_lik,
    }
}

/// Cholesky-decomposition log-determinant of a symmetric PSD matrix.
///
/// Returns 2 * sum(log diag(L)) where M = L L^T. Returns NaN if
/// the matrix is not positive definite.
fn log_det_psd(m: &Array2<f64>) -> f64 {
    let l = cholesky(m);
    let mut acc = 0.0_f64;
    for i in 0..l.nrows() {
        let d = l[(i, i)];
        if d <= 0.0 {
            return f64::NAN;
        }
        acc += d.ln();
    }
    2.0 * acc
}

/// Cholesky-based inversion of a symmetric PSD matrix.
///
/// Computes `M^{-1}` via L L^T = M then inverting via forward +
/// backward substitution on the identity.
fn invert_psd_matrix(m: &Array2<f64>) -> Array2<f64> {
    let n = m.nrows();
    let l = cholesky(m);
    let mut inv = Array2::<f64>::eye(n);
    // Solve L Y = I
    for k in 0..n {
        for i in 0..n {
            let mut sum = inv[(i, k)];
            for j in 0..i {
                sum -= l[(i, j)] * inv[(j, k)];
            }
            inv[(i, k)] = sum / l[(i, i)];
        }
    }
    // Solve L^T X = Y
    let mut out = Array2::<f64>::zeros((n, n));
    for k in 0..n {
        for i in (0..n).rev() {
            let mut sum = inv[(i, k)];
            for j in (i + 1)..n {
                sum -= l[(j, i)] * out[(j, k)];
            }
            out[(i, k)] = sum / l[(i, i)];
        }
    }
    out
}

/// Cholesky decomposition: returns L lower-triangular such that
/// L L^T = M for symmetric PSD M. No checks on PSD-ness; if M is
/// not PSD, the diagonal of L will contain a NaN.
fn cholesky(m: &Array2<f64>) -> Array2<f64> {
    let n = m.nrows();
    let mut l = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..=i {
            let mut sum = m[(i, j)];
            for k in 0..j {
                sum -= l[(i, k)] * l[(j, k)];
            }
            if i == j {
                l[(i, j)] = sum.sqrt();
            } else {
                l[(i, j)] = sum / l[(j, j)];
            }
        }
    }
    l
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn cholesky_identity() {
        let i = Array2::<f64>::eye(3);
        let l = cholesky(&i);
        for r in 0..3 {
            for c in 0..3 {
                let expected = if r == c { 1.0 } else { 0.0 };
                assert!((l[(r, c)] - expected).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn invert_psd_matrix_identity() {
        let i = Array2::<f64>::eye(3);
        let inv = invert_psd_matrix(&i);
        for r in 0..3 {
            for c in 0..3 {
                let expected = if r == c { 1.0 } else { 0.0 };
                assert!((inv[(r, c)] - expected).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn log_det_identity_is_zero() {
        let i = Array2::<f64>::eye(4);
        assert!(log_det_psd(&i).abs() < 1e-12);
    }

    #[test]
    fn kalman_scalar_random_walk_matches_analytic() {
        // 1-D random walk: A=1, C=1, Q=0.1, R=1.
        // First-step prediction: mu_0 = 0, Sigma_0 = 1.
        // After observing y_0 = 1:
        //   S = 1 + 1 = 2
        //   K = 1 / 2 = 0.5
        //   x_filt = 0 + 0.5 * (1 - 0) = 0.5
        //   P_filt = (1 - 0.5) * 1 * (1 - 0.5) + 0.5 * 1 * 0.5 = 0.25 + 0.25 = 0.5
        let a = array![[1.0]];
        let b = array![[]];
        let c = array![[1.0]];
        let d = array![[]];
        let q = array![[0.1]];
        let r_mat = array![[1.0]];
        let mu_0 = array![0.0];
        let sigma_0 = array![[1.0]];

        let obs = array![[1.0_f64]];
        let controls = Array2::<f64>::zeros((1, 0));

        let result = kalman_filter(
            obs.view(), controls.view(),
            a.view(), b.view(), c.view(), d.view(),
            q.view(), r_mat.view(),
            mu_0.view(), sigma_0.view(),
        );

        assert!((result.means[(0, 0)] - 0.5).abs() < 1e-12);
        assert!((result.covariances[(0, 0, 0)] - 0.5).abs() < 1e-12);
    }

    #[test]
    fn kalman_log_likelihood_finite() {
        // 2-D state, 1-D obs, T=10 random sequence — log-lik must be finite.
        let a = array![[0.9, 0.1], [0.0, 0.95]];
        let b = array![[], []];
        let c = array![[1.0, 0.0]];
        let d = array![[]];
        let q = array![[0.01, 0.0], [0.0, 0.01]];
        let r_mat = array![[0.1]];
        let mu_0 = array![0.0, 0.0];
        let sigma_0 = array![[1.0, 0.0], [0.0, 1.0]];

        let obs = array![
            [0.1], [0.2], [0.15], [0.18], [0.22],
            [0.25], [0.21], [0.24], [0.27], [0.26],
        ];
        let controls = Array2::<f64>::zeros((10, 0));

        let result = kalman_filter(
            obs.view(), controls.view(),
            a.view(), b.view(), c.view(), d.view(),
            q.view(), r_mat.view(),
            mu_0.view(), sigma_0.view(),
        );

        assert!(result.log_likelihood.is_finite());
    }
}
