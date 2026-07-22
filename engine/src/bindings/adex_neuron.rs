// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Adaptive exponential neuron PyO3 binding

use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::neuron;

#[pyclass(
    name = "AdExNeuron",
    module = "sc_neurocore_engine.sc_neurocore_engine"
)]
#[derive(Clone)]
pub struct PyAdExNeuron {
    inner: neuron::AdExNeuron,
}

#[pymethods]
impl PyAdExNeuron {
    #[new]
    fn new() -> Self {
        Self {
            inner: neuron::AdExNeuron::new(),
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
        d.set_item("w", self.inner.w)?;
        Ok(d.into_any().unbind())
    }
}

/// Register the adaptive exponential neuron class.
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyAdExNeuron>()?;
    Ok(())
}
