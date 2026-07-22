// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — EProp ALIF neuron PyO3 binding

use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::neurons;

#[pyclass(
    name = "EPropALIFNeuron",
    module = "sc_neurocore_engine.sc_neurocore_engine"
)]
#[derive(Clone)]
pub struct PyEPropALIFNeuron {
    inner: neurons::EPropALIFNeuron,
}

#[pymethods]
impl PyEPropALIFNeuron {
    #[new]
    #[pyo3(signature = (tau_m=20.0, tau_a=200.0, dt=1.0))]
    fn new(tau_m: f64, tau_a: f64, dt: f64) -> Self {
        Self {
            inner: neurons::EPropALIFNeuron::new(tau_m, tau_a, dt),
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
        d.set_item("a", self.inner.a)?;
        d.set_item("e_trace", self.inner.e_trace)?;
        Ok(d.into_any().unbind())
    }
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyEPropALIFNeuron>()?;
    Ok(())
}
