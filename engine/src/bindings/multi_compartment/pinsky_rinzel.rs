// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Pinsky-Rinzel neuron PyO3 binding

use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::neurons;

#[pyclass(
    name = "PinskyRinzelNeuron",
    module = "sc_neurocore_engine.sc_neurocore_engine"
)]
#[derive(Clone)]
pub struct PyPinskyRinzelNeuron {
    inner: neurons::PinskyRinzelNeuron,
}

#[pymethods]
impl PyPinskyRinzelNeuron {
    #[new]
    fn new() -> Self {
        Self {
            inner: neurons::PinskyRinzelNeuron::new(),
        }
    }

    #[pyo3(signature = (current_soma, current_dend=0.0))]
    fn step(&mut self, current_soma: f64, current_dend: f64) -> i32 {
        self.inner.step(current_soma, current_dend)
    }

    fn reset(&mut self) {
        self.inner.reset();
    }

    fn get_state(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let d = PyDict::new(py);
        d.set_item("v_s", self.inner.v_s)?;
        d.set_item("v_d", self.inner.v_d)?;
        Ok(d.into_any().unbind())
    }
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyPinskyRinzelNeuron>()?;
    Ok(())
}
