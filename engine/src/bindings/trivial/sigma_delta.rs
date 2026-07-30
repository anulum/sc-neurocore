// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li

//! PyO3 exposure for the sampled APSDM contract.

use crate::neurons;
use numpy::{IntoPyArray, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

#[pyclass(
    name = "SigmaDeltaNeuron",
    module = "sc_neurocore_engine.sc_neurocore_engine"
)]
#[derive(Clone)]
pub struct PySigmaDeltaNeuron {
    inner: neurons::SigmaDeltaNeuron,
}

#[pymethods]
impl PySigmaDeltaNeuron {
    #[new]
    #[pyo3(signature=(sigma=0.0,reconstruction=0.0,delta=1.0,tau_reconstruction=10.0,dt=0.1))]
    fn new(
        sigma: f64,
        reconstruction: f64,
        delta: f64,
        tau_reconstruction: f64,
        dt: f64,
    ) -> PyResult<Self> {
        let inner = neurons::SigmaDeltaNeuron {
            sigma,
            reconstruction,
            delta,
            tau_reconstruction,
            dt,
        };
        if !inner.validate() {
            return Err(PyValueError::new_err(
                "invalid SigmaDelta state or configuration",
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
        d.set_item("sigma", self.inner.sigma)?;
        d.set_item("reconstruction", self.inner.reconstruction)?;
        Ok(d.into_any().unbind())
    }
}

#[pyfunction]
#[pyo3(signature=(sigma,reconstruction,delta,tau_reconstruction,dt,currents))]
fn py_sigma_delta_simulate<'py>(
    py: Python<'py>,
    sigma: f64,
    reconstruction: f64,
    delta: f64,
    tau_reconstruction: f64,
    dt: f64,
    currents: PyReadonlyArray1<'py, f64>,
) -> PyResult<Py<PyAny>> {
    let mut n = neurons::SigmaDeltaNeuron {
        sigma,
        reconstruction,
        delta,
        tau_reconstruction,
        dt,
    };
    if !n.validate() {
        return Err(PyValueError::new_err(
            "invalid SigmaDelta state or configuration",
        ));
    }
    let mut sigmas = Vec::with_capacity(currents.len()?);
    let mut reconstructions = Vec::with_capacity(currents.len()?);
    let mut events = Vec::with_capacity(currents.len()?);
    for &current in currents.as_slice()? {
        events.push(n.try_step(current).map_err(PyValueError::new_err)?);
        sigmas.push(n.sigma);
        reconstructions.push(n.reconstruction);
    }
    let d = PyDict::new(py);
    d.set_item("sigma", sigmas.into_pyarray(py))?;
    d.set_item("reconstruction", reconstructions.into_pyarray(py))?;
    d.set_item("events", events.into_pyarray(py))?;
    d.set_item("sigma_final", n.sigma)?;
    d.set_item("reconstruction_final", n.reconstruction)?;
    Ok(d.into_any().unbind())
}
pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PySigmaDeltaNeuron>()?;
    module.add_function(wrap_pyfunction!(py_sigma_delta_simulate, module)?)?;
    Ok(())
}
