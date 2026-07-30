// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li

//! PyO3 exposure for the retained SC bipolar accumulator.

use crate::neurons;
use numpy::{IntoPyArray, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

#[pyclass(
    name = "SCSigmaDeltaAccumulatorNeuron",
    module = "sc_neurocore_engine.sc_neurocore_engine"
)]
#[derive(Clone)]
pub struct PySCSigmaDeltaAccumulatorNeuron {
    inner: neurons::SCSigmaDeltaAccumulatorNeuron,
}
#[pymethods]
impl PySCSigmaDeltaAccumulatorNeuron {
    #[new]
    #[pyo3(signature=(sigma=0.0,v_threshold=1.0))]
    fn new(sigma: f64, v_threshold: f64) -> PyResult<Self> {
        let inner = neurons::SCSigmaDeltaAccumulatorNeuron { sigma, v_threshold };
        if !inner.validate() {
            return Err(PyValueError::new_err("invalid SC SigmaDelta accumulator"));
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
        Ok(d.into_any().unbind())
    }
}
#[pyfunction]
#[pyo3(signature=(sigma,v_threshold,currents))]
fn py_sc_sigma_delta_accumulator_simulate<'py>(
    py: Python<'py>,
    sigma: f64,
    v_threshold: f64,
    currents: PyReadonlyArray1<'py, f64>,
) -> PyResult<Py<PyAny>> {
    let mut n = neurons::SCSigmaDeltaAccumulatorNeuron { sigma, v_threshold };
    if !n.validate() {
        return Err(PyValueError::new_err("invalid SC SigmaDelta accumulator"));
    }
    let mut trace = Vec::with_capacity(currents.len()?);
    let mut events = Vec::with_capacity(currents.len()?);
    for &current in currents.as_slice()? {
        events.push(n.try_step(current).map_err(PyValueError::new_err)?);
        trace.push(n.sigma);
    }
    let d = PyDict::new(py);
    d.set_item("sigma", trace.into_pyarray(py))?;
    d.set_item("events", events.into_pyarray(py))?;
    d.set_item("sigma_final", n.sigma)?;
    Ok(d.into_any().unbind())
}
pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PySCSigmaDeltaAccumulatorNeuron>()?;
    module.add_function(wrap_pyfunction!(
        py_sc_sigma_delta_accumulator_simulate,
        module
    )?)?;
    Ok(())
}
