// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Larter-Breakspear neural-mass PyO3 binding

use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::neurons;

#[pyclass(
    name = "LarterBreakspearNeuron",
    module = "sc_neurocore_engine.sc_neurocore_engine"
)]
#[derive(Clone)]
pub struct PyLarterBreakspearNeuron {
    inner: neurons::LarterBreakspearNeuron,
}

#[pymethods]
impl PyLarterBreakspearNeuron {
    #[new]
    fn new() -> Self {
        Self {
            inner: neurons::LarterBreakspearNeuron::new(),
        }
    }

    #[pyo3(signature = (coupling=0.0))]
    fn step(&mut self, coupling: f64) -> f64 {
        self.inner.step(coupling)
    }

    fn reset(&mut self) {
        self.inner.reset();
    }

    fn get_state(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let d = PyDict::new(py);
        d.set_item("v", self.inner.v)?;
        d.set_item("w", self.inner.w)?;
        d.set_item("z", self.inner.z)?;
        Ok(d.into_any().unbind())
    }
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyLarterBreakspearNeuron>()?;
    Ok(())
}
