// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Wong-Wang decision unit PyO3 binding

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::neurons;

#[pyclass(
    name = "WongWangUnit",
    module = "sc_neurocore_engine.sc_neurocore_engine"
)]
#[derive(Clone)]
pub struct PyWongWangUnit {
    inner: neurons::WongWangUnit,
}

#[pymethods]
impl PyWongWangUnit {
    #[new]
    #[pyo3(signature = (seed=42))]
    fn new(seed: u64) -> Self {
        Self {
            inner: neurons::WongWangUnit::new(seed),
        }
    }

    #[pyo3(signature = (stim1=0.0, stim2=0.0))]
    fn step(&mut self, stim1: f64, stim2: f64) -> PyResult<(f64, f64)> {
        self.inner.step(stim1, stim2).map_err(PyValueError::new_err)
    }

    #[pyo3(signature = (stim1=0.0, stim2=0.0, xi1=0.0, xi2=0.0))]
    fn step_with_gaussian_samples(
        &mut self,
        stim1: f64,
        stim2: f64,
        xi1: f64,
        xi2: f64,
    ) -> PyResult<(f64, f64)> {
        self.inner
            .step_with_gaussian_samples(stim1, stim2, xi1, xi2)
            .map_err(PyValueError::new_err)
    }

    fn reset(&mut self) {
        self.inner.reset();
    }

    fn get_state(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let d = PyDict::new(py);
        d.set_item("s1", self.inner.s1)?;
        d.set_item("s2", self.inner.s2)?;
        d.set_item("noise1", self.inner.noise1)?;
        d.set_item("noise2", self.inner.noise2)?;
        Ok(d.into_any().unbind())
    }
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyWongWangUnit>()?;
    Ok(())
}
