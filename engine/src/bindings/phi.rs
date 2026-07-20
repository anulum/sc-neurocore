// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Phi-star PyO3 binding

//! Python binding for Gaussian integrated-information estimation.

use numpy::{PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Register the Phi-star estimator with the extension module.
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(py_phi_star, module)?)?;
    Ok(())
}

/// Estimate integrated information from a channel-major time series.
#[pyfunction]
fn py_phi_star(data: PyReadonlyArray2<'_, f64>, tau: usize) -> PyResult<f64> {
    if !data.is_c_contiguous() {
        return Err(PyValueError::new_err(
            "py_phi_star requires C-contiguous array input",
        ));
    }
    let shape = data.shape();
    let n_channels = shape[0];
    let n_timesteps = shape[1];
    let flat = data.as_slice().map_err(|error| {
        PyValueError::new_err(format!("py_phi_star requires C-contiguous array: {error}"))
    })?;
    let channels: Vec<Vec<f64>> = (0..n_channels)
        .map(|index| flat[index * n_timesteps..(index + 1) * n_timesteps].to_vec())
        .collect();
    Ok(crate::phi::phi_star(&channels, tau))
}
