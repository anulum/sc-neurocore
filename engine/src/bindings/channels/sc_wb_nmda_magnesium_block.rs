// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Retained SC WB plus NMDA PyO3 binding

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::neurons;

#[pyclass(
    name = "SCWBNMDAMagnesiumBlockNeuron",
    module = "sc_neurocore_engine.sc_neurocore_engine"
)]
#[derive(Clone)]
pub struct PySCWBNMDAMagnesiumBlockNeuron {
    inner: neurons::SCWBNMDAMagnesiumBlockNeuron,
}

#[pymethods]
impl PySCWBNMDAMagnesiumBlockNeuron {
    #[new]
    fn new() -> Self {
        Self {
            inner: neurons::SCWBNMDAMagnesiumBlockNeuron::new(),
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
        state.set_item("s_nmda", self.inner.s_nmda)?;
        Ok(state.into_any().unbind())
    }
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PySCWBNMDAMagnesiumBlockNeuron>()?;
    Ok(())
}
