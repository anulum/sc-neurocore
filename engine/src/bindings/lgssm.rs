// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — LGSSM Kalman filter PyO3 binding

//! Python binding for the Linear Gaussian state-space model Kalman filter.

use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::lgssm;

/// Register the LGSSM Kalman filter with the extension module.
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(py_lgssm_kalman_filter, module)?)?;
    Ok(())
}

/// Forward Kalman filter for a Linear Gaussian State-Space Model.
///
/// Parity contract with `sc_neurocore.world_model.predictive_model.KalmanFilter`:
/// for the same model parameters and observation sequence, the
/// returned (means, covariances, log_likelihood) must agree with
/// the Python implementation to within float64 round-off.
///
/// All matrices are passed as flat row-major Vec<f64>; the caller
/// supplies their shapes explicitly. Returns a dict with keys:
///   - "means": Vec<Vec<f64>> shape (T, d)
///   - "covariances": Vec<Vec<Vec<f64>>> shape (T, d, d)
///   - "pred_means": Vec<Vec<f64>> shape (T, d)
///   - "pred_covariances": Vec<Vec<Vec<f64>>> shape (T, d, d)
///   - "log_likelihood": f64
///   - "backend": "rust"
#[pyfunction]
#[pyo3(signature = (
    obs_flat, controls_flat, t_len, p_dim, m_dim,
    a_flat, b_flat, c_flat, d_flat, q_flat, r_flat,
    mu_0, sigma_0_flat, d_dim,
))]
#[allow(clippy::too_many_arguments)]
fn py_lgssm_kalman_filter<'py>(
    py: Python<'py>,
    obs_flat: Vec<f64>,
    controls_flat: Vec<f64>,
    t_len: usize,
    p_dim: usize,
    m_dim: usize,
    a_flat: Vec<f64>,
    b_flat: Vec<f64>,
    c_flat: Vec<f64>,
    d_flat: Vec<f64>,
    q_flat: Vec<f64>,
    r_flat: Vec<f64>,
    mu_0: Vec<f64>,
    sigma_0_flat: Vec<f64>,
    d_dim: usize,
) -> PyResult<Py<PyAny>> {
    use ndarray::Array1;
    use ndarray::Array2;

    let to_2d = |flat: &[f64], rows: usize, cols: usize| -> Array2<f64> {
        Array2::from_shape_vec((rows, cols), flat.to_vec()).expect("shape")
    };
    let obs = to_2d(&obs_flat, t_len, p_dim);
    let controls = to_2d(&controls_flat, t_len, m_dim);
    let a = to_2d(&a_flat, d_dim, d_dim);
    let b = to_2d(&b_flat, d_dim, m_dim);
    let c = to_2d(&c_flat, p_dim, d_dim);
    let d = to_2d(&d_flat, p_dim, m_dim);
    let q = to_2d(&q_flat, d_dim, d_dim);
    let r = to_2d(&r_flat, p_dim, p_dim);
    let mu_0_arr = Array1::from(mu_0);
    let sigma_0 = to_2d(&sigma_0_flat, d_dim, d_dim);

    let result = lgssm::kalman_filter(
        obs.view(),
        controls.view(),
        a.view(),
        b.view(),
        c.view(),
        d.view(),
        q.view(),
        r.view(),
        mu_0_arr.view(),
        sigma_0.view(),
    );

    // Convert to Python-friendly nested Vec
    let means: Vec<Vec<f64>> = (0..t_len)
        .map(|t| (0..d_dim).map(|i| result.means[(t, i)]).collect())
        .collect();
    let covs: Vec<Vec<Vec<f64>>> = (0..t_len)
        .map(|t| {
            (0..d_dim)
                .map(|i| (0..d_dim).map(|j| result.covariances[(t, i, j)]).collect())
                .collect()
        })
        .collect();
    let pred_means: Vec<Vec<f64>> = (0..t_len)
        .map(|t| (0..d_dim).map(|i| result.pred_means[(t, i)]).collect())
        .collect();
    let pred_covs: Vec<Vec<Vec<f64>>> = (0..t_len)
        .map(|t| {
            (0..d_dim)
                .map(|i| {
                    (0..d_dim)
                        .map(|j| result.pred_covariances[(t, i, j)])
                        .collect()
                })
                .collect()
        })
        .collect();

    let dict = PyDict::new(py);
    dict.set_item("means", means)?;
    dict.set_item("covariances", covs)?;
    dict.set_item("pred_means", pred_means)?;
    dict.set_item("pred_covariances", pred_covs)?;
    dict.set_item("log_likelihood", result.log_likelihood)?;
    dict.set_item("backend", "rust")?;
    Ok(dict.into_any().unbind())
}
