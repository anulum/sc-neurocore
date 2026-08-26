// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — NMDA-channel neuron PyO3 binding

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::neurons;

#[pyclass(
    name = "NMDANeuron",
    module = "sc_neurocore_engine.sc_neurocore_engine"
)]
#[derive(Clone)]
pub struct PyNMDANeuron {
    inner: neurons::NMDANeuron,
}

#[pymethods]
impl PyNMDANeuron {
    #[new]
    fn new() -> Self {
        Self {
            inner: neurons::NMDANeuron::default(),
        }
    }

    /// Advance one step; raises `ValueError` with the state unchanged for
    /// a non-finite drive, an invalid configuration, or a non-finite
    /// candidate.
    fn step(&mut self, current: f64) -> PyResult<i32> {
        self.inner.try_step(current).map_err(PyValueError::new_err)
    }

    /// Restore dynamic state to the initial values, preserving parameters.
    fn reset(&mut self) {
        self.inner.reset();
    }

    /// Return the complete dynamic state as a Python dictionary.
    fn get_state(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let state = PyDict::new(py);
        state.set_item("v", self.inner.v)?;
        state.set_item("h", self.inner.h)?;
        state.set_item("n", self.inner.n)?;
        state.set_item("s_nmda", self.inner.s_nmda)?;
        Ok(state.into_any().unbind())
    }
}

/// Register the NMDA-channel neuron class.
pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyNMDANeuron>()?;
    Ok(())
}
