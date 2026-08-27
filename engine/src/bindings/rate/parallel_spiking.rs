// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Parallel-spiking neuron PyO3 binding

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::neurons;

#[pyclass(
    name = "ParallelSpikingNeuron",
    module = "sc_neurocore_engine.sc_neurocore_engine"
)]
#[derive(Clone)]
pub struct PyParallelSpikingNeuron {
    inner: neurons::ParallelSpikingNeuron,
}

#[pymethods]
impl PyParallelSpikingNeuron {
    #[new]
    #[pyo3(signature = (kernel_size=8, v_threshold=1.0))]
    fn new(kernel_size: usize, v_threshold: f64) -> PyResult<Self> {
        if kernel_size < 1 {
            return Err(PyValueError::new_err(
                "kernel_size must be a positive integer",
            ));
        }
        Ok(Self {
            inner: neurons::ParallelSpikingNeuron::new(kernel_size, v_threshold),
        })
    }

    /// Advance one step with the newest input; raises `ValueError` with
    /// the state unchanged on any invalid input.
    fn step(&mut self, current: f64) -> PyResult<i32> {
        self.inner.try_step(current).map_err(PyValueError::new_err)
    }

    /// Replace the learnable weight vector `W` (length must stay k).
    fn set_weights(&mut self, weights: Vec<f64>) -> PyResult<()> {
        if weights.len() != self.inner.weights.len() {
            return Err(PyValueError::new_err(
                "weights must have exactly kernel_size entries",
            ));
        }
        if !weights.iter().all(|w| w.is_finite()) {
            return Err(PyValueError::new_err("weights must be finite"));
        }
        self.inner.weights = weights;
        Ok(())
    }

    /// Clear the retained inputs, preserving weights and threshold.
    fn reset(&mut self) {
        self.inner.reset();
    }

    /// Return the complete dynamic state as a Python dictionary.
    fn get_state(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let state = PyDict::new(py);
        state.set_item("hidden", self.inner.hidden)?;
        state.set_item("history", self.inner.history.clone())?;
        Ok(state.into_any().unbind())
    }
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyParallelSpikingNeuron>()?;
    Ok(())
}
