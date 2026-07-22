// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — TrueNorth neuron PyO3 binding

use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::neurons;

#[pyclass(
    name = "TrueNorthNeuron",
    module = "sc_neurocore_engine.sc_neurocore_engine"
)]
#[derive(Clone)]
pub struct PyTrueNorthNeuron {
    inner: neurons::TrueNorthNeuron,
}

#[pymethods]
impl PyTrueNorthNeuron {
    #[new]
    #[pyo3(signature = (threshold=100))]
    fn new(threshold: i32) -> Self {
        Self {
            inner: neurons::TrueNorthNeuron::new(threshold),
        }
    }

    fn step(&mut self, weighted_input: i32) -> i32 {
        self.inner.step(weighted_input)
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

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyTrueNorthNeuron>()?;
    Ok(())
}
