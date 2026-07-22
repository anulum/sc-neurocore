// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Astrocyte LIF neuron PyO3 binding

use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::neurons;

#[pyclass(
    name = "RustAstrocyteLIFNeuron",
    module = "sc_neurocore_engine.sc_neurocore_engine"
)]
#[derive(Clone)]
pub struct PyAstrocyteLIFNeuron {
    inner: neurons::AstrocyteLIFNeuron,
}

#[pymethods]
impl PyAstrocyteLIFNeuron {
    #[new]
    fn new() -> Self {
        Self {
            inner: neurons::AstrocyteLIFNeuron::new(),
        }
    }

    fn step(&mut self, current: f64) -> i32 {
        self.inner.step(current)
    }

    fn step_with_pre(&mut self, i_ext: f64, pre_spike: bool) -> i32 {
        self.inner.step_with_pre(i_ext, pre_spike)
    }

    fn reset(&mut self) {
        self.inner.reset();
    }

    fn get_state(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let d = PyDict::new(py);
        d.set_item("v", self.inner.v)?;
        d.set_item("ca", self.inner.ca)?;
        Ok(d.into_any().unbind())
    }
}

/// Register the astrocyte LIF neuron class.
pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyAstrocyteLIFNeuron>()?;
    Ok(())
}
