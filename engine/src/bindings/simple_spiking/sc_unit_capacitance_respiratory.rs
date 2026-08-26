// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — retained unit-capacitance respiratory PyO3 binding

use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::neurons;

#[pyclass(
    name = "SCUnitCapacitanceRespiratoryNeuron",
    module = "sc_neurocore_engine.sc_neurocore_engine"
)]
#[derive(Clone)]
pub struct PySCUnitCapacitanceRespiratoryNeuron {
    inner: neurons::SCUnitCapacitanceRespiratoryNeuron,
}

#[pymethods]
impl PySCUnitCapacitanceRespiratoryNeuron {
    #[new]
    fn new() -> Self {
        Self {
            inner: neurons::SCUnitCapacitanceRespiratoryNeuron::default(),
        }
    }

    fn step(&mut self, current: f64) -> i32 {
        self.inner.step(current)
    }

    fn reset(&mut self) {
        self.inner.reset();
    }

    fn get_state(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let state = PyDict::new(py);
        state.set_item("v", self.inner.inner.v)?;
        state.set_item("n", self.inner.inner.n)?;
        state.set_item("h_nap", self.inner.inner.h_nap)?;
        Ok(state.into_any().unbind())
    }
}

/// Register the retained unit-capacitance respiratory class.
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PySCUnitCapacitanceRespiratoryNeuron>()?;
    Ok(())
}
