// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Leaky-compete-and-fire neuron PyO3 binding

use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::neurons;

#[pyclass(
    name = "LeakyCompeteFireNeuron",
    module = "sc_neurocore_engine.sc_neurocore_engine"
)]
#[derive(Clone)]
pub struct PyLeakyCompeteFireNeuron {
    inner: neurons::LeakyCompeteFireNeuron,
}

#[pymethods]
impl PyLeakyCompeteFireNeuron {
    #[new]
    #[pyo3(signature = (n_units=4))]
    fn new(n_units: usize) -> Self {
        Self {
            inner: neurons::LeakyCompeteFireNeuron::new(n_units),
        }
    }

    fn step(&mut self, currents: Vec<f64>) -> Vec<i32> {
        self.inner.step(&currents)
    }

    fn reset(&mut self) {
        self.inner.reset();
    }

    fn get_state(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let d = PyDict::new(py);
        d.set_item("v", self.inner.v.clone())?;
        Ok(d.into_any().unbind())
    }
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyLeakyCompeteFireNeuron>()?;
    Ok(())
}
