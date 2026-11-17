// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Differentiable dense-layer PyO3 binding

//! Python binding for stochastic differentiable dense-layer training.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::surrogate_binding::parse_surrogate;
use crate::grad;

/// Register differentiable dense-layer training with the extension module.
pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyDifferentiableDenseLayer>()?;
    Ok(())
}

#[pyclass(
    name = "DifferentiableDenseLayer",
    module = "sc_neurocore_engine.sc_neurocore_engine"
)]
pub struct PyDifferentiableDenseLayer {
    inner: grad::DifferentiableDenseLayer,
}

#[pymethods]
impl PyDifferentiableDenseLayer {
    #[new]
    #[pyo3(signature = (
        n_inputs,
        n_neurons,
        length=1024,
        seed=24301,
        surrogate="fast_sigmoid",
        k=None
    ))]
    fn new(
        n_inputs: usize,
        n_neurons: usize,
        length: usize,
        seed: u64,
        surrogate: &str,
        k: Option<f32>,
    ) -> PyResult<Self> {
        let surrogate = parse_surrogate(surrogate, k)?;
        if length == 0 {
            return Err(PyValueError::new_err("bitstream length must be > 0"));
        }
        if n_inputs == 0 {
            return Err(PyValueError::new_err("n_inputs must be > 0"));
        }
        if n_neurons == 0 {
            return Err(PyValueError::new_err("n_neurons must be > 0"));
        }
        crate::runtime_control_binding::ensure_parallel_runtime()?;
        Ok(Self {
            inner: grad::DifferentiableDenseLayer::try_new(
                n_inputs, n_neurons, length, seed, surrogate,
            )
            .map_err(PyValueError::new_err)?,
        })
    }

    fn get_weights(&self) -> Vec<Vec<f64>> {
        self.inner.layer.get_weights()
    }

    #[pyo3(signature = (input_values, seed=44257))]
    fn forward(&mut self, input_values: Vec<f64>, seed: u64) -> PyResult<Vec<f64>> {
        self.inner
            .forward(&input_values, seed)
            .map_err(PyValueError::new_err)
    }

    fn backward(&self, grad_output: Vec<f64>) -> PyResult<(Vec<f64>, Vec<Vec<f64>>)> {
        self.inner
            .backward(&grad_output)
            .map_err(PyValueError::new_err)
    }

    fn update_weights(&mut self, weight_grads: Vec<Vec<f64>>, lr: f64) -> PyResult<()> {
        self.inner
            .try_update_weights(&weight_grads, lr)
            .map_err(PyValueError::new_err)
    }

    fn clear_cache(&mut self) {
        self.inner.clear_cache();
    }
}
