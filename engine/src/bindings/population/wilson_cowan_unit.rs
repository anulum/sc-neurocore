// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Wilson-Cowan population unit PyO3 binding

use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::neurons;

#[pyclass(
    name = "WilsonCowanUnit",
    module = "sc_neurocore_engine.sc_neurocore_engine"
)]
#[derive(Clone)]
pub struct PyWilsonCowanUnit {
    inner: neurons::WilsonCowanUnit,
}

#[pymethods]
impl PyWilsonCowanUnit {
    #[new]
    fn new() -> Self {
        Self {
            inner: neurons::WilsonCowanUnit::new(),
        }
    }

    #[pyo3(signature = (ext_input=0.0))]
    fn step(&mut self, ext_input: f64) -> f64 {
        self.inner.step(ext_input)
    }

    fn reset(&mut self) {
        self.inner.reset();
    }

    fn get_state(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let d = PyDict::new(py);
        d.set_item("e", self.inner.e)?;
        d.set_item("i", self.inner.i)?;
        Ok(d.into_any().unbind())
    }
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyWilsonCowanUnit>()?;
    Ok(())
}
