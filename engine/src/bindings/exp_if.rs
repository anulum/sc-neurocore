// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Exponential integrate-and-fire PyO3 binding

//! Python binding for the exponential integrate-and-fire neuron.

use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::neuron;

/// Register the exponential integrate-and-fire neuron with the extension module.
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyExpIFNeuron>()?;
    Ok(())
}

#[pyclass(
    name = "ExpIFNeuron",
    module = "sc_neurocore_engine.sc_neurocore_engine"
)]
#[derive(Clone)]
pub struct PyExpIFNeuron {
    inner: neuron::ExpIfNeuron,
}

#[pymethods]
impl PyExpIFNeuron {
    #[new]
    fn new() -> Self {
        Self {
            inner: neuron::ExpIfNeuron::new(),
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
        d.set_item("refractory_remaining", self.inner.refractory_remaining)?;
        Ok(d.into_any().unbind())
    }
}
