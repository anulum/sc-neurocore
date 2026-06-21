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

use nalgebra::{Cholesky as NaCholesky, DMatrix, DVector};
use ndarray::{s, Array1, Array2, Array3, ArrayView1, ArrayView2};

/// Result of the Kalman filter forward pass.
pub struct KalmanResult {
    pub means: Array2<f64>,            // (T, d) filtered means
    pub covariances: Array3<f64>,      // (T, d, d) filtered covariances
    pub pred_means: Array2<f64>,       // (T, d) predicted means
    pub pred_covariances: Array3<f64>, // (T, d, d) predicted covariances
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
    observations: ArrayView2<f64>, // (T, p)
    controls: ArrayView2<f64>,     // (T, m) or (T, 0)
    a: ArrayView2<f64>,            // (d, d)
    b: ArrayView2<f64>,            // (d, m) or (d, 0)
    c: ArrayView2<f64>,            // (p, d)
    d: ArrayView2<f64>,            // (p, m) or (p, 0)
    q: ArrayView2<f64>,            // (d, d)
    r: ArrayView2<f64>,            // (p, p)
    mu_0: ArrayView1<f64>,         // (d,)
    sigma_0: ArrayView2<f64>,      // (d, d)
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
        let s_mat = c.dot(&p_pred).dot(&c.t()) + r;

        // S is symmetric positive-definite; factor it once with a Cholesky
        // decomposition (nalgebra, LAPACK-grade) and reuse the single factor for the
        // log-determinant, the innovation quadratic form, and the Kalman gain —
        // never forming S^{-1} explicitly.
        let s_na = DMatrix::<f64>::from_fn(p_dim, p_dim, |i, j| s_mat[(i, j)]);
        let (logdet_s, s_inv_innov, k_gain) = match NaCholesky::new(s_na) {
            Some(chol) => {
                // log|S| = 2 Σ ln L_ii — the stable sum-of-logs form.
                let l = chol.l();
                let logdet = 2.0 * (0..p_dim).map(|i| l[(i, i)].ln()).sum::<f64>();
                // S^{-1} innov for the quadratic form, via the triangular solves.
                let innov_na = DVector::<f64>::from_fn(p_dim, |i, _| innov[i]);
                let z = chol.solve(&innov_na);
                let s_inv_innov = Array1::<f64>::from_iter((0..p_dim).map(|i| z[i]));
                // Kalman gain K = P_pred C^T S^{-1}. With S and P_pred symmetric,
                // K^T = S^{-1} (C P_pred), so solve S X = C P_pred and transpose —
                // no explicit inverse.
                let cp = c.dot(&p_pred); // (p × d)
                let cp_na = DMatrix::<f64>::from_fn(p_dim, d_dim, |i, j| cp[(i, j)]);
                let x = chol.solve(&cp_na); // S^{-1} (C P_pred), (p × d)
                let k = Array2::<f64>::from_shape_fn((d_dim, p_dim), |(i, j)| x[(j, i)]);
                (logdet, s_inv_innov, k)
            }
            None => {
                // Defensive: a non-positive-definite innovation covariance cannot
                // occur while R is positive-definite. Mirror the prior NaN
                // propagation rather than panicking.
                (
                    f64::NAN,
                    Array1::<f64>::zeros(p_dim),
                    Array2::<f64>::zeros((d_dim, p_dim)),
                )
            }
        };

        let quad_form = innov.dot(&s_inv_innov);
        log_lik += -0.5 * (p_dim as f64 * two_pi_log + logdet_s + quad_form);

        // Filtered state: x_filt = x_pred + K e
        let x_filt = &x_pred + &k_gain.dot(&innov);

        // Joseph form for filtered covariance:
        //   P_filt = (I - K C) P_pred (I - K C)^T + K R K^T
        let i_minus_kc = &i_d - &k_gain.dot(&c);
        let p_filt = i_minus_kc.dot(&p_pred).dot(&i_minus_kc.t()) + k_gain.dot(&r).dot(&k_gain.t());

        means.slice_mut(s![t, ..]).assign(&x_filt);
        covs.slice_mut(s![t, .., ..]).assign(&p_filt);

        // Predict next state
        let mut x_next = a.dot(&x_filt);
        if has_control {
            let u_t = controls.slice(s![t, ..]);
            x_next = x_next + b.dot(&u_t);
        }
        let p_next = a.dot(&p_filt).dot(&a.t()) + q;
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

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

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
            obs.view(),
            controls.view(),
            a.view(),
            b.view(),
            c.view(),
            d.view(),
            q.view(),
            r_mat.view(),
            mu_0.view(),
            sigma_0.view(),
        );

        assert!((result.means[(0, 0)] - 0.5).abs() < 1e-12);
        assert!((result.covariances[(0, 0, 0)] - 0.5).abs() < 1e-12);

        // Exact single-step Gaussian log-likelihood exercises the Cholesky
        // log-determinant and quadratic-form path: S = 2, innov = 1, p = 1.
        //   log N(y | y_hat, S) = -0.5 (log 2π + log|S| + innovᵀ S⁻¹ innov)
        //                       = -0.5 (log 2π + log 2 + 0.5)
        let expected_ll = -0.5 * ((2.0 * std::f64::consts::PI).ln() + 2.0_f64.ln() + 0.5);
        assert!((result.log_likelihood - expected_ll).abs() < 1e-12);
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
            [0.1],
            [0.2],
            [0.15],
            [0.18],
            [0.22],
            [0.25],
            [0.21],
            [0.24],
            [0.27],
            [0.26],
        ];
        let controls = Array2::<f64>::zeros((10, 0));

        let result = kalman_filter(
            obs.view(),
            controls.view(),
            a.view(),
            b.view(),
            c.view(),
            d.view(),
            q.view(),
            r_mat.view(),
            mu_0.view(),
            sigma_0.view(),
        );

        assert!(result.log_likelihood.is_finite());
    }

    #[test]
    fn kalman_two_dim_obs_symmetric_psd_and_finite() {
        // 2-D state, 2-D observation with a non-diagonal C and a non-diagonal R so
        // the gain transpose and the 2×2 Cholesky solve are exercised (p = d = 2).
        let a = array![[0.95, 0.0], [0.1, 0.9]];
        let b = array![[], []];
        let c = array![[1.0, 0.2], [0.0, 1.0]];
        let d = array![[], []];
        let q = array![[0.02, 0.0], [0.0, 0.02]];
        let r_mat = array![[0.15, 0.05], [0.05, 0.2]];
        let mu_0 = array![0.0, 0.0];
        let sigma_0 = array![[1.0, 0.0], [0.0, 1.0]];

        let obs = array![
            [0.10, 0.05],
            [0.20, 0.12],
            [0.18, 0.09],
            [0.25, 0.15],
            [0.30, 0.20],
        ];
        let controls = Array2::<f64>::zeros((5, 0));

        let result = kalman_filter(
            obs.view(),
            controls.view(),
            a.view(),
            b.view(),
            c.view(),
            d.view(),
            q.view(),
            r_mat.view(),
            mu_0.view(),
            sigma_0.view(),
        );

        assert!(result.log_likelihood.is_finite());
        for t in 0..obs.nrows() {
            assert!(result.means[(t, 0)].is_finite());
            assert!(result.means[(t, 1)].is_finite());
            // Filtered covariance must stay symmetric (Joseph form) and PSD.
            let p00 = result.covariances[(t, 0, 0)];
            let p11 = result.covariances[(t, 1, 1)];
            let p01 = result.covariances[(t, 0, 1)];
            let p10 = result.covariances[(t, 1, 0)];
            assert!(
                (p01 - p10).abs() < 1e-12,
                "covariance not symmetric at t={t}"
            );
            assert!(p00 >= 0.0 && p11 >= 0.0, "negative variance at t={t}");
            assert!(
                p00 * p11 - p01 * p10 >= -1e-12,
                "covariance not PSD at t={t}"
            );
        }
    }
}
