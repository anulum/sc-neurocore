// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Wong-Wang PyO3 batch binding

//! Python boundary for the deterministic-sample Euler/OU batch contract.

use numpy::{IntoPyArray, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Register the batch binding beside the scalar neuron class.
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(py_wong_wang_simulate, module)?)?;
    Ok(())
}

/// Simulate a Wong-Wang batch from caller-owned standard-normal samples.
#[pyfunction]
#[pyo3(signature = (
    s1_init, s2_init, noise1_init, noise2_init,
    tau_s, tau_ampa, gamma, j_n, j_cross, i_0, sigma, dt,
    stim1, stim2, xi,
))]
#[allow(clippy::too_many_arguments)]
fn py_wong_wang_simulate<'py>(
    py: Python<'py>,
    s1_init: f64,
    s2_init: f64,
    noise1_init: f64,
    noise2_init: f64,
    tau_s: f64,
    tau_ampa: f64,
    gamma: f64,
    j_n: f64,
    j_cross: f64,
    i_0: f64,
    sigma: f64,
    dt: f64,
    stim1: PyReadonlyArray1<'py, f64>,
    stim2: PyReadonlyArray1<'py, f64>,
    xi: PyReadonlyArray1<'py, f64>,
) -> PyResult<Py<PyAny>> {
    let result = crate::wong_wang::simulate(
        s1_init,
        s2_init,
        noise1_init,
        noise2_init,
        tau_s,
        tau_ampa,
        gamma,
        j_n,
        j_cross,
        i_0,
        sigma,
        dt,
        stim1.as_slice()?,
        stim2.as_slice()?,
        xi.as_slice()?,
    )
    .map_err(PyValueError::new_err)?;
    let mapping = PyDict::new(py);
    mapping.set_item("s1", result.s1.into_pyarray(py))?;
    mapping.set_item("s2", result.s2.into_pyarray(py))?;
    mapping.set_item("noise1", result.noise1.into_pyarray(py))?;
    mapping.set_item("noise2", result.noise2.into_pyarray(py))?;
    mapping.set_item("r1", result.rate1.into_pyarray(py))?;
    mapping.set_item("r2", result.rate2.into_pyarray(py))?;
    mapping.set_item("s1_final", result.final_s1)?;
    mapping.set_item("s2_final", result.final_s2)?;
    mapping.set_item("noise1_final", result.final_noise1)?;
    mapping.set_item("noise2_final", result.final_noise2)?;
    Ok(mapping.into_any().unbind())
}

#[cfg(test)]
mod tests {
    #[test]
    fn engine_batch_rejects_mismatched_input_lengths() {
        let result = crate::wong_wang::simulate(
            0.1,
            0.1,
            0.0,
            0.0,
            0.1,
            0.002,
            0.641,
            0.2609,
            0.0497,
            0.3255,
            0.02,
            0.0001,
            &[0.0],
            &[],
            &[0.0, 0.0],
        );
        assert!(result.is_err());
    }
}
