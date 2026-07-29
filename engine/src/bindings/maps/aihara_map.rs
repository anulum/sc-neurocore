// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — source-faithful Aihara PyO3 binding

use numpy::{IntoPyArray, PyReadonlyArray1};
use pyo3::exceptions::{PyFloatingPointError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::neurons::{self, AiharaMapError};

fn map_error(error: AiharaMapError) -> PyErr {
    match error {
        AiharaMapError::NonFiniteCandidate => PyFloatingPointError::new_err(error.to_string()),
        _ => PyValueError::new_err(error.to_string()),
    }
}

#[pyclass(
    name = "AiharaMapNeuron",
    module = "sc_neurocore_engine.sc_neurocore_engine"
)]
#[derive(Clone)]
pub struct PyAiharaMapNeuron {
    inner: neurons::AiharaMapNeuron,
}

#[pymethods]
impl PyAiharaMapNeuron {
    #[new]
    fn new() -> Self {
        Self {
            inner: neurons::AiharaMapNeuron::default(),
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
        state.set_item("y", self.inner.y)?;
        state.set_item("x", self.inner.output())?;
        Ok(state.into_any().unbind())
    }
}

/// Register the checked class and complete batch function.
pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyAiharaMapNeuron>()?;
    module.add_function(wrap_pyfunction!(py_aihara_map_simulate, module)?)?;
    Ok(())
}

#[pyfunction]
#[pyo3(signature = (y, k, alpha, bias, epsilon, current))]
fn py_aihara_map_simulate<'py>(
    py: Python<'py>,
    y: f64,
    k: f64,
    alpha: f64,
    bias: f64,
    epsilon: f64,
    current: PyReadonlyArray1<'py, f64>,
) -> PyResult<Py<PyAny>> {
    let result = neurons::simulate_aihara_map(y, k, alpha, bias, epsilon, current.as_slice()?)
        .map_err(map_error)?;
    let mapping = PyDict::new(py);
    mapping.set_item("y", result.y.into_pyarray(py))?;
    mapping.set_item("x", result.x.into_pyarray(py))?;
    mapping.set_item("spikes", result.spikes.into_pyarray(py))?;
    mapping.set_item("y_final", result.y_final)?;
    mapping.set_item("x_final", result.x_final)?;
    mapping.set_item("spike_count", result.spike_count)?;
    Ok(mapping.into_any().unbind())
}
