// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Generalised linear model neuron PyO3 binding

use pyo3::prelude::*;

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

    fn step(&mut self, stimulus: f64) -> i32 {
        self.inner.step(stimulus)
    }

    fn reset(&mut self) {
        self.inner.reset();
    }
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyGLMNeuron>()?;
    Ok(())
}
