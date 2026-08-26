// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — T-type calcium neuron PyO3 binding

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::neurons;

#[pyclass(
    name = "TTypeCaNeuron",
    module = "sc_neurocore_engine.sc_neurocore_engine"
)]
#[derive(Clone)]
pub struct PyTTypeCaNeuron {
    inner: neurons::TTypeCaNeuron,
}

#[pymethods]
impl PyTTypeCaNeuron {
    #[new]
    fn new() -> Self {
        Self {
            inner: neurons::TTypeCaNeuron::default(),
        }
    }

    fn step(&mut self, current: f64) -> PyResult<i32> {
        self.inner.try_step(current).map_err(PyValueError::new_err)
    }

    fn reset(&mut self) {
        self.inner.reset();
    }

    fn get_state(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let state = PyDict::new(py);
        state.set_item("v", self.inner.v)?;
        state.set_item("h", self.inner.h)?;
        state.set_item("n", self.inner.n)?;
        state.set_item("s", self.inner.s)?;
        Ok(state.into_any().unbind())
    }
}

/// Register the T-type calcium neuron class.
pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyTTypeCaNeuron>()?;
    Ok(())
}
