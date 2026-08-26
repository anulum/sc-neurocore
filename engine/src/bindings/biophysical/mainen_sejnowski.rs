// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Mainen-Sejnowski neuron PyO3 binding

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::neurons;

#[pyclass(
    name = "MainenSejnowskiNeuron",
    module = "sc_neurocore_engine.sc_neurocore_engine"
)]
#[derive(Clone)]
pub struct PyMainenSejnowskiNeuron {
    inner: neurons::MainenSejnowskiNeuron,
}

#[pymethods]
impl PyMainenSejnowskiNeuron {
    #[new]
    fn new() -> Self {
        Self {
            inner: neurons::MainenSejnowskiNeuron::default(),
        }
    }

    /// Reconstruct the original engine configuration (Gauss-Seidel
    /// compartment ordering with the historical rate handling) without
    /// adding a second catalogue identity.
    #[staticmethod]
    fn legacy_sequential() -> Self {
        Self {
            inner: neurons::MainenSejnowskiNeuron::new_legacy_sequential(),
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
        state.set_item("vs", self.inner.vs)?;
        state.set_item("va", self.inner.va)?;
        state.set_item("m", self.inner.m)?;
        state.set_item("h", self.inner.h)?;
        state.set_item("n", self.inner.n)?;
        Ok(state.into_any().unbind())
    }
}

/// Register the Mainen-Sejnowski neuron class.
pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyMainenSejnowskiNeuron>()?;
    Ok(())
}
