// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Hybrid linear-attention neuron PyO3 binding

use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::neurons;

#[pyclass(
    name = "RustHybridLinearAttentionNeuron",
    module = "sc_neurocore_engine.sc_neurocore_engine"
)]
#[derive(Clone)]
pub struct PyHybridLinearAttentionNeuron {
    inner: neurons::HybridLinearAttentionNeuron,
}

#[pymethods]
impl PyHybridLinearAttentionNeuron {
    #[new]
    #[pyo3(signature = (dim=16))]
    fn new(dim: usize) -> Self {
        Self {
            inner: neurons::HybridLinearAttentionNeuron::new(dim),
        }
    }
    fn step(&mut self, current: f64) -> i32 {
        self.inner.step(current)
    }
    fn step_qkv(&mut self, query: f64, key: f64, value: f64) -> f64 {
        self.inner.step_qkv(query, key, value)
    }
    fn reset(&mut self) {
        self.inner.reset();
    }
    fn get_state(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let d = PyDict::new(py);
        d.set_item("v", self.inner.v)?;
        Ok(d.into_any().unbind())
    }
}

/// Register the hybrid linear-attention neuron class.
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyHybridLinearAttentionNeuron>()?;
    Ok(())
}
