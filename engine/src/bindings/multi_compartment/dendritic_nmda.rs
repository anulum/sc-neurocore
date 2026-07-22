// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Dendritic NMDA neuron PyO3 binding

use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::neurons;

#[pyclass(
    name = "RustDendriticNMDANeuron",
    module = "sc_neurocore_engine.sc_neurocore_engine"
)]
#[derive(Clone)]
pub struct PyDendriticNMDANeuron {
    inner: neurons::DendriticNMDANeuron,
}

#[pymethods]
impl PyDendriticNMDANeuron {
    #[new]
    fn new() -> Self {
        Self {
            inner: neurons::DendriticNMDANeuron::new(),
        }
    }

    fn step(&mut self, i_soma: f64, glutamate: f64) -> i32 {
        self.inner.step(i_soma, glutamate)
    }

    fn reset(&mut self) {
        self.inner.reset();
    }

    fn get_state(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let d = PyDict::new(py);
        d.set_item("v_soma", self.inner.v_soma)?;
        d.set_item("v_dend", self.inner.v_dend)?;
        Ok(d.into_any().unbind())
    }
}

/// Register the dendritic NMDA neuron class.
pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyDendriticNMDANeuron>()?;
    Ok(())
}
