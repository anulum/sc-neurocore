// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Montbrió population PyO3 scalar and batch binding

use numpy::{IntoPyArray, PyReadonlyArray1};
use pyo3::exceptions::{PyFloatingPointError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::neurons::ermentrout_kopell_pop::ErmentroutKopellPopulationError;
use crate::neurons::ErmentroutKopellPopulation;

fn map_mpr_error(error: ErmentroutKopellPopulationError) -> PyErr {
    match error {
        ErmentroutKopellPopulationError::NonFiniteCandidate
        | ErmentroutKopellPopulationError::NegativeCandidateRate => {
            PyFloatingPointError::new_err(error.to_string())
        }
        _ => PyValueError::new_err(error.to_string()),
    }
}

/// Register the scalar class and batch function outside the crate root.
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyErmentroutKopellPopulation>()?;
    module.add_function(wrap_pyfunction!(py_ermentrout_kopell_pop_simulate, module)?)?;
    Ok(())
}

#[pyclass(
    name = "ErmentroutKopellPopulation",
    module = "sc_neurocore_engine.sc_neurocore_engine"
)]
#[derive(Clone)]
pub struct PyErmentroutKopellPopulation {
    inner: ErmentroutKopellPopulation,
}

#[pymethods]
impl PyErmentroutKopellPopulation {
    /// Construct and validate a complete scalar MPR configuration.
    #[new]
    #[pyo3(signature = (
        r=0.1, v=-2.0, tau=1.0, delta=1.0, eta_bar=-5.0,
        coupling=15.0, dt=0.01,
    ))]
    fn new(
        r: f64,
        v: f64,
        tau: f64,
        delta: f64,
        eta_bar: f64,
        coupling: f64,
        dt: f64,
    ) -> PyResult<Self> {
        let inner =
            ErmentroutKopellPopulation::with_parameters(r, v, tau, delta, eta_bar, coupling, dt)
                .map_err(map_mpr_error)?;
        Ok(Self { inner })
    }

    /// Apply one atomic simultaneous Euler step and return the new rate.
    #[pyo3(signature = (ext_input=0.0))]
    fn step(&mut self, ext_input: f64) -> PyResult<f64> {
        self.inner.try_step(ext_input).map_err(map_mpr_error)
    }

    /// Restore both dynamic states while preserving all parameters.
    fn reset(&mut self) {
        self.inner.reset();
    }

    /// Return the two current dynamic states as a Python mapping.
    fn get_state(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let mapping = PyDict::new(py);
        mapping.set_item("r", self.inner.r)?;
        mapping.set_item("v", self.inner.v)?;
        Ok(mapping.into_any().unbind())
    }
}

/// Simulate one complete MPR external-drive batch.
#[pyfunction]
#[pyo3(signature = (r, v, tau, delta, eta_bar, coupling, dt, ext_input))]
#[expect(
    clippy::too_many_arguments,
    reason = "Python extension parity surface carries the complete configuration"
)]
fn py_ermentrout_kopell_pop_simulate<'py>(
    py: Python<'py>,
    r: f64,
    v: f64,
    tau: f64,
    delta: f64,
    eta_bar: f64,
    coupling: f64,
    dt: f64,
    ext_input: PyReadonlyArray1<'py, f64>,
) -> PyResult<Py<PyAny>> {
    let result = crate::neurons::ermentrout_kopell_pop::simulate(
        r,
        v,
        tau,
        delta,
        eta_bar,
        coupling,
        dt,
        ext_input.as_slice()?,
    )
    .map_err(map_mpr_error)?;
    let mapping = PyDict::new(py);
    mapping.set_item("r", result.r.into_pyarray(py))?;
    mapping.set_item("v", result.v.into_pyarray(py))?;
    mapping.set_item("r_final", result.final_state[0])?;
    mapping.set_item("v_final", result.final_state[1])?;
    Ok(mapping.into_any().unbind())
}

#[cfg(test)]
mod tests {
    #[test]
    fn engine_batch_rejects_nonfinite_drive_without_partial_result() {
        let result = crate::neurons::ermentrout_kopell_pop::simulate(
            0.1,
            -2.0,
            1.0,
            1.0,
            -5.0,
            15.0,
            0.01,
            &[0.0, f64::NAN],
        );
        assert!(result.is_err());
    }
}
