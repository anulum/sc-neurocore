// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Tsodyks-Markram neuron PyO3 binding

use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::neurons;

#[pyclass(
    name = "TsodyksMarkramNeuron",
    module = "sc_neurocore_engine.sc_neurocore_engine"
)]
#[derive(Clone)]
pub struct PyTsodyksMarkramNeuron {
    inner: neurons::TsodyksMarkramNeuron,
}

#[pymethods]
impl PyTsodyksMarkramNeuron {
    #[new]
    fn new() -> Self {
        Self {
            inner: neurons::TsodyksMarkramNeuron::new(),
        }
    }

    #[pyo3(signature = (current, presynaptic_spike=false))]
    fn step(&mut self, current: f64, presynaptic_spike: bool) -> i32 {
        self.inner.step(current, presynaptic_spike)
    }

    fn reset(&mut self) {
        self.inner.reset();
    }

    fn get_state(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let d = PyDict::new(py);
        d.set_item("v", self.inner.v)?;
        d.set_item("x", self.inner.x)?;
        d.set_item("u", self.inner.u)?;
        Ok(d.into_any().unbind())
    }
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyTsodyksMarkramNeuron>()?;
    Ok(())
}
