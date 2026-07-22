// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Loihi 2 neuron PyO3 binding

use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::neurons;

#[pyclass(
    name = "Loihi2Neuron",
    module = "sc_neurocore_engine.sc_neurocore_engine"
)]
#[derive(Clone)]
pub struct PyLoihi2Neuron {
    inner: neurons::Loihi2Neuron,
}

#[pymethods]
impl PyLoihi2Neuron {
    #[new]
    fn new() -> Self {
        Self {
            inner: neurons::Loihi2Neuron::new(),
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
        d.set_item("s1", self.inner.s1)?;
        d.set_item("s2", self.inner.s2)?;
        d.set_item("s3", self.inner.s3)?;
        Ok(d.into_any().unbind())
    }
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyLoihi2Neuron>()?;
    Ok(())
}
