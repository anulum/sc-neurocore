// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Generalised linear model neuron PyO3 binding

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::neurons;

#[pyclass(name = "GLMNeuron", module = "sc_neurocore_engine.sc_neurocore_engine")]
#[derive(Clone)]
pub struct PyGLMNeuron {
    inner: neurons::GLMNeuron,
}

#[pymethods]
impl PyGLMNeuron {
    #[new]
    #[pyo3(signature = (n_k=10, n_h=20, seed=42))]
    fn new(n_k: usize, n_h: usize, seed: u64) -> Self {
        Self {
            inner: neurons::GLMNeuron::new(n_k, n_h, seed),
        }
    }

    #[pyo3(signature = (stimulus, uniform=None))]
    fn step(&mut self, stimulus: f64, uniform: Option<f64>) -> PyResult<i32> {
        self.inner
            .try_step(stimulus, uniform)
            .map_err(PyValueError::new_err)
    }

    fn reset(&mut self) {
        self.inner.reset();
    }

    fn get_state(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let state = PyDict::new(py);
        state.set_item("mu", self.inner.mu)?;
        state.set_item("dt_ms", self.inner.dt_ms)?;
        state.set_item("k", self.inner.k.clone())?;
        state.set_item("h", self.inner.h.clone())?;
        state.set_item("stim_buf", self.inner.stim_buf_view())?;
        state.set_item("spike_buf", self.inner.spike_buf_view())?;
        Ok(state.into_any().unbind())
    }
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyGLMNeuron>()?;
    Ok(())
}
