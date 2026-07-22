// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Short-term-plasticity synapse PyO3 binding

use pyo3::prelude::*;
use pyo3::types::PyDict;

#[pyclass(
    name = "RustShortTermPlasticitySynapse",
    module = "sc_neurocore_engine.sc_neurocore_engine"
)]
#[derive(Clone)]
pub struct PyShortTermPlasticitySynapse {
    inner: crate::synapses::ShortTermPlasticitySynapse,
}

#[pymethods]
impl PyShortTermPlasticitySynapse {
    #[new]
    fn new() -> Self {
        Self {
            inner: crate::synapses::ShortTermPlasticitySynapse::new_depressing(),
        }
    }

    #[staticmethod]
    fn depressing() -> Self {
        Self {
            inner: crate::synapses::ShortTermPlasticitySynapse::new_depressing(),
        }
    }

    #[staticmethod]
    fn facilitating() -> Self {
        Self {
            inner: crate::synapses::ShortTermPlasticitySynapse::new_facilitating(),
        }
    }

    fn step(&mut self, pre_spike: bool) -> f64 {
        self.inner.step(pre_spike)
    }

    fn reset(&mut self) {
        self.inner.reset();
    }

    fn get_state(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let d = PyDict::new(py);
        d.set_item("x", self.inner.x)?;
        d.set_item("u", self.inner.u)?;
        Ok(d.into_any().unbind())
    }
}

/// Register the short-term-plasticity synapse class.
pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyShortTermPlasticitySynapse>()?;
    Ok(())
}
