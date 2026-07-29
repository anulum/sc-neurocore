// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — project adaptive-threshold-map PyO3 binding

use numpy::{IntoPyArray, PyReadonlyArray1};
use pyo3::exceptions::{PyFloatingPointError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::neurons::{self, SCAdaptiveThresholdMapError};

fn map_error(error: SCAdaptiveThresholdMapError) -> PyErr {
    match error {
        SCAdaptiveThresholdMapError::NonFiniteCandidate => {
            PyFloatingPointError::new_err(error.to_string())
        }
        _ => PyValueError::new_err(error.to_string()),
    }
}

/// Python-owned checked SC adaptive-threshold-map state object.
#[pyclass(
    name = "SCAdaptiveThresholdMapNeuron",
    module = "sc_neurocore_engine.sc_neurocore_engine"
)]
#[derive(Clone)]
pub struct PySCAdaptiveThresholdMapNeuron {
    inner: neurons::SCAdaptiveThresholdMapNeuron,
}

#[pymethods]
impl PySCAdaptiveThresholdMapNeuron {
    #[new]
    fn new() -> Self {
        Self {
            inner: neurons::SCAdaptiveThresholdMapNeuron::default(),
        }
    }

    fn step(&mut self, current: f64) -> PyResult<i32> {
        self.inner.try_step(current).map_err(map_error)
    }

    fn reset(&mut self) {
        self.inner.reset();
    }

    fn get_state(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let state = PyDict::new(py);
        state.set_item("x", self.inner.x)?;
        state.set_item("theta", self.inner.theta)?;
        Ok(state.into_any().unbind())
    }
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PySCAdaptiveThresholdMapNeuron>()?;
    module.add_function(wrap_pyfunction!(
        py_sc_adaptive_threshold_map_simulate,
        module
    )?)?;
    Ok(())
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (x, theta, k, beta, gamma, theta_spike, x_threshold, current))]
fn py_sc_adaptive_threshold_map_simulate<'py>(
    py: Python<'py>,
    x: f64,
    theta: f64,
    k: f64,
    beta: f64,
    gamma: f64,
    theta_spike: f64,
    x_threshold: f64,
    current: PyReadonlyArray1<'py, f64>,
) -> PyResult<Py<PyAny>> {
    let result = neurons::simulate_sc_adaptive_threshold_map(
        x,
        theta,
        k,
        beta,
        gamma,
        theta_spike,
        x_threshold,
        current.as_slice()?,
    )
    .map_err(map_error)?;
    let mapping = PyDict::new(py);
    mapping.set_item("x", result.x.into_pyarray(py))?;
    mapping.set_item("theta", result.theta.into_pyarray(py))?;
    mapping.set_item("spikes", result.spikes.into_pyarray(py))?;
    mapping.set_item("x_final", result.x_final)?;
    mapping.set_item("theta_final", result.theta_final)?;
    mapping.set_item("spike_count", result.spike_count)?;
    Ok(mapping.into_any().unbind())
}
