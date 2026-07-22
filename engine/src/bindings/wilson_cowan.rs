// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Wilson-Cowan PyO3 binding

//! Python binding for the Wilson-Cowan 1972 excitatory/inhibitory rate model.

use crate::wilson_cowan;
use numpy::{IntoPyArray, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Register the Wilson-Cowan batch simulator with the extension module.
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(py_wilson_cowan_simulate, module)?)?;
    Ok(())
}

/// Simulate a single Wilson-Cowan E/I unit for `ext_input.len()` steps
/// and return per-step `e`, `i` traces plus final scalars.
///
/// Returns a dict with keys: `e`, `i` (1-D float64 arrays of length
/// `n_steps`) + the final scalars `e_final`, `i_final`.
#[pyfunction]
#[pyo3(signature = (
    e_init, i_init,
    w_ee, w_ei, w_ie, w_ii,
    tau_e, tau_i,
    a, theta, dt,
    ext_input,
))]
#[allow(clippy::too_many_arguments)]
fn py_wilson_cowan_simulate<'py>(
    py: Python<'py>,
    e_init: f64,
    i_init: f64,
    w_ee: f64,
    w_ei: f64,
    w_ie: f64,
    w_ii: f64,
    tau_e: f64,
    tau_i: f64,
    a: f64,
    theta: f64,
    dt: f64,
    ext_input: PyReadonlyArray1<'py, f64>,
) -> PyResult<Py<PyAny>> {
    let ext = ext_input.as_slice()?;
    let n = ext.len();
    let mut e_out = vec![0.0_f64; n];
    let mut i_out = vec![0.0_f64; n];
    let (e_final, i_final) = wilson_cowan::simulate(
        e_init, i_init, w_ee, w_ei, w_ie, w_ii, tau_e, tau_i, a, theta, dt, ext, &mut e_out,
        &mut i_out,
    )
    .map_err(PyValueError::new_err)?;
    let d = PyDict::new(py);
    d.set_item("e", e_out.into_pyarray(py))?;
    d.set_item("i", i_out.into_pyarray(py))?;
    d.set_item("e_final", e_final)?;
    d.set_item("i_final", i_final)?;
    Ok(d.into_any().unbind())
}
