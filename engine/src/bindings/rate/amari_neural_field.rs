// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Amari neural-field PyO3 binding

use numpy::{IntoPyArray, PyArray1};
use pyo3::prelude::*;

use crate::neurons;

#[pyclass(
    name = "AmariNeuralField",
    module = "sc_neurocore_engine.sc_neurocore_engine"
)]
#[derive(Clone)]
pub struct PyAmariNeuralField {
    inner: neurons::AmariNeuralField,
}

#[pymethods]
impl PyAmariNeuralField {
    #[new]
    #[pyo3(signature = (n=64))]
    fn new(n: usize) -> Self {
        Self {
            inner: neurons::AmariNeuralField::new(n),
        }
    }

    fn step(&mut self, input: Vec<f64>) -> f64 {
        self.inner.step(&input)
    }

    fn reset(&mut self) {
        self.inner.reset();
    }

    fn get_state<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.u.clone().into_pyarray(py)
    }
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyAmariNeuralField>()?;
    Ok(())
}
