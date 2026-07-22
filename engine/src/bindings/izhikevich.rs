// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Izhikevich PyO3 binding

//! Python binding for the floating-point Izhikevich neuron.

use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::neuron;

/// Register the floating-point Izhikevich neuron with the extension module.
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyIzhikevich>()?;
    Ok(())
}

#[pyclass(
    name = "Izhikevich",
    module = "sc_neurocore_engine.sc_neurocore_engine"
)]
pub struct PyIzhikevich {
    inner: neuron::Izhikevich,
}

#[pymethods]
impl PyIzhikevich {
    #[new]
    #[pyo3(signature = (a=0.02, b=0.2, c=-65.0, d=8.0, dt=1.0))]
    fn new(a: f64, b: f64, c: f64, d: f64, dt: f64) -> Self {
        Self {
            inner: neuron::Izhikevich::new(a, b, c, d, dt),
        }
    }

    fn step(&mut self, current: f64) -> i32 {
        self.inner.step(current)
    }

    fn reset(&mut self) {
        self.inner.reset();
    }

    fn reset_state(&mut self) {
        self.reset();
    }

    fn get_state(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let dict = PyDict::new(py);
        dict.set_item("v", self.inner.v)?;
        dict.set_item("u", self.inner.u)?;
        Ok(dict.into_any().unbind())
    }
}
