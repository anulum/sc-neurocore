// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — preserved SC chaotic-map PyO3 binding

use numpy::{IntoPyArray, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::neurons::{simulate_sc_chaotic_map, SCChaoticMapNeuron};

#[pyclass(
    name = "SCChaoticMapNeuron",
    module = "sc_neurocore_engine.sc_neurocore_engine"
)]
#[derive(Clone)]
pub struct PySCChaoticMapNeuron {
    inner: SCChaoticMapNeuron,
}

#[pymethods]
impl PySCChaoticMapNeuron {
    #[new]
    fn new() -> Self {
        Self {
            inner: SCChaoticMapNeuron::new(),
        }
    }

    fn step(&mut self, current: f64) -> PyResult<i32> {
        self.inner.try_step(current).map_err(PyValueError::new_err)
    }

    fn reset(&mut self) {
        self.inner.reset();
    }

    fn get_state(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let state = PyDict::new(py);
        state.set_item("x", self.inner.x)?;
        state.set_item("y", self.inner.y)?;
        Ok(state.into_any().unbind())
    }
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PySCChaoticMapNeuron>()?;
    module.add_function(wrap_pyfunction!(py_sc_chaotic_map_simulate, module)?)?;
    Ok(())
}

#[pyfunction]
#[pyo3(signature = (x, y, k_f, k_s, alpha, delta, x_threshold, current))]
fn py_sc_chaotic_map_simulate<'py>(
    py: Python<'py>,
    x: f64,
    y: f64,
    k_f: f64,
    k_s: f64,
    alpha: f64,
    delta: f64,
    x_threshold: f64,
    current: PyReadonlyArray1<'py, f64>,
) -> PyResult<Py<PyAny>> {
    let result = simulate_sc_chaotic_map(
        x,
        y,
        k_f,
        k_s,
        alpha,
        delta,
        x_threshold,
        current.as_slice()?,
    )
    .map_err(PyValueError::new_err)?;
    let mapping = PyDict::new(py);
    mapping.set_item("x", result.x.into_pyarray(py))?;
    mapping.set_item("y", result.y.into_pyarray(py))?;
    mapping.set_item("spikes", result.spikes.into_pyarray(py))?;
    mapping.set_item("x_final", result.x_final)?;
    mapping.set_item("y_final", result.y_final)?;
    mapping.set_item("spike_count", result.spike_count)?;
    Ok(mapping.into_any().unbind())
}
