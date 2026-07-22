// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Lapicque neuron PyO3 binding

use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::neuron;

#[pyclass(
    name = "LapicqueNeuron",
    module = "sc_neurocore_engine.sc_neurocore_engine"
)]
#[derive(Clone)]
pub struct PyLapicqueNeuron {
    inner: neuron::LapicqueNeuron,
}

#[pymethods]
impl PyLapicqueNeuron {
    #[new]
    #[pyo3(signature = (tau=20.0, resistance=1.0, threshold=1.0, dt=1.0))]
    fn new(tau: f64, resistance: f64, threshold: f64, dt: f64) -> Self {
        Self {
            inner: neuron::LapicqueNeuron::new(tau, resistance, threshold, dt),
        }
    }
    fn step(&mut self, current: f64) -> i32 {
        self.inner.step(current)
    }
    fn reset(&mut self) {
        self.inner.reset();
    }
    fn get_state(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let d = PyDict::new(py);
        d.set_item("v", self.inner.v)?;
        Ok(d.into_any().unbind())
    }
}

/// Register the Lapicque neuron class.
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyLapicqueNeuron>()?;
    Ok(())
}
