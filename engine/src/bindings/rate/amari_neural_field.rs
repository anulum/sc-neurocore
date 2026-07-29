// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Amari neural-field PyO3 binding

use numpy::{IntoPyArray, PyArray1};
use pyo3::exceptions::PyValueError;
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
    #[pyo3(signature = (n=64, tau=10.0, a_exc=1.5, a_width=2.0, b_inh=0.75, b_width=1.0, dx=0.5, dt=0.5, u=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        n: usize,
        tau: f64,
        a_exc: f64,
        a_width: f64,
        b_inh: f64,
        b_width: f64,
        dx: f64,
        dt: f64,
        u: Option<Vec<f64>>,
    ) -> PyResult<Self> {
        let state = u.unwrap_or_else(|| vec![0.0; n]);
        let inner = neurons::AmariNeuralField::with_config(
            n, tau, a_exc, a_width, b_inh, b_width, dx, dt, state,
        )
        .map_err(PyValueError::new_err)?;
        Ok(Self { inner })
    }

    /// Advance an exact-length vector stimulus atomically.
    fn step(&mut self, input: Vec<f64>) -> PyResult<f64> {
        self.inner.step(&input).map_err(PyValueError::new_err)
    }

    /// Zero the dynamic field while preserving configuration.
    fn reset(&mut self) {
        self.inner.reset();
    }

    /// Return a copy of all periodic field potentials.
    fn get_state<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.u.clone().into_pyarray(py)
    }
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyAmariNeuralField>()?;
    Ok(())
}
