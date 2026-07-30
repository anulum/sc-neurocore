// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li

//! PyO3 exposure for the source-bound McKean Heaviside system.

use crate::neurons::McKeanNeuron;
use numpy::{IntoPyArray, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

#[pyclass(
    name = "McKeanNeuron",
    module = "sc_neurocore_engine.sc_neurocore_engine"
)]
#[derive(Clone)]
pub struct PyMcKeanNeuron {
    inner: McKeanNeuron,
}

#[pymethods]
impl PyMcKeanNeuron {
    #[new]
    #[pyo3(signature=(v=0.0,w=0.0,a=0.25,lambda_=1.0,mu=1.0,b=0.01,dt=0.1))]
    fn new(v: f64, w: f64, a: f64, lambda_: f64, mu: f64, b: f64, dt: f64) -> PyResult<Self> {
        let inner = McKeanNeuron {
            v,
            w,
            a,
            lambda: lambda_,
            mu,
            b,
            dt,
        };
        if !inner.valid() {
            return Err(PyValueError::new_err(
                "invalid McKean state or configuration",
            ));
        }
        Ok(Self { inner })
    }
    fn step(&mut self, current: f64) -> PyResult<i32> {
        self.inner.try_step(current).map_err(PyValueError::new_err)
    }
    fn reset(&mut self) {
        self.inner.reset();
    }
    fn get_state(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let d = PyDict::new(py);
        d.set_item("v", self.inner.v)?;
        d.set_item("w", self.inner.w)?;
        Ok(d.into_any().unbind())
    }
}

#[pyfunction]
#[pyo3(signature=(v,w,a,lambda_,mu,b,dt,currents))]
#[allow(clippy::too_many_arguments)]
fn py_mckean_simulate<'py>(
    py: Python<'py>,
    v: f64,
    w: f64,
    a: f64,
    lambda_: f64,
    mu: f64,
    b: f64,
    dt: f64,
    currents: PyReadonlyArray1<'py, f64>,
) -> PyResult<Py<PyAny>> {
    let mut n = McKeanNeuron {
        v,
        w,
        a,
        lambda: lambda_,
        mu,
        b,
        dt,
    };
    if !n.valid() {
        return Err(PyValueError::new_err(
            "invalid McKean state or configuration",
        ));
    }
    let mut voltages = Vec::with_capacity(currents.len()?);
    let mut recovery = Vec::with_capacity(currents.len()?);
    let mut events = Vec::with_capacity(currents.len()?);
    for &current in currents.as_slice()? {
        events.push(n.try_step(current).map_err(PyValueError::new_err)?);
        voltages.push(n.v);
        recovery.push(n.w);
    }
    let d = PyDict::new(py);
    d.set_item("voltages", voltages.into_pyarray(py))?;
    d.set_item("recovery", recovery.into_pyarray(py))?;
    d.set_item("events", events.into_pyarray(py))?;
    d.set_item("v_final", n.v)?;
    d.set_item("w_final", n.w)?;
    Ok(d.into_any().unbind())
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyMcKeanNeuron>()?;
    module.add_function(wrap_pyfunction!(py_mckean_simulate, module)?)?;
    Ok(())
}
