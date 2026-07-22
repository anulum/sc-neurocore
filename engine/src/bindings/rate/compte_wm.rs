// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Compte working-memory neuron PyO3 binding

use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::neurons;

#[pyclass(
    name = "CompteWMNeuron",
    module = "sc_neurocore_engine.sc_neurocore_engine"
)]
#[derive(Clone)]
pub struct PyCompteWMNeuron {
    inner: neurons::CompteWMNeuron,
}

#[pymethods]
impl PyCompteWMNeuron {
    #[new]
    fn new() -> Self {
        Self {
            inner: neurons::CompteWMNeuron::new(),
        }
    }

    #[pyo3(signature = (current, spike_in=false))]
    fn step(&mut self, current: f64, spike_in: bool) -> i32 {
        self.inner.step(current, spike_in)
    }

    fn reset(&mut self) {
        self.inner.reset();
    }

    fn get_state(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let d = PyDict::new(py);
        d.set_item("v", self.inner.v)?;
        d.set_item("s_nmda", self.inner.s_nmda)?;
        Ok(d.into_any().unbind())
    }
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyCompteWMNeuron>()?;
    Ok(())
}
