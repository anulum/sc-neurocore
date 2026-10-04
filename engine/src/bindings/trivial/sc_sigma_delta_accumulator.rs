// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Retained bipolar accumulator native Python binding

//! PyO3 exposure for the retained SC bipolar accumulator.

use numpy::{PyArrayMethods, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::fixed_point_lif_binding::array_allocation::zeros_1d;
use crate::neurons;

/// Python-owned retained accumulator state and threshold.
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
    /// Construct a finite residual with a finite positive threshold.
    #[new]
    #[pyo3(signature=(sigma=0.0,v_threshold=1.0))]
    fn new(sigma: f64, v_threshold: f64) -> PyResult<Self> {
        let inner = neurons::SCSigmaDeltaAccumulatorNeuron { sigma, v_threshold };
        if !inner.validate() {
            return Err(PyValueError::new_err("invalid SC SigmaDelta accumulator"));
        }
        Ok(Self { inner })
    }
    /// Advance one atomic signed transition with one event at most.
    fn step(&mut self, current: f64) -> PyResult<i32> {
        self.inner.try_step(current).map_err(PyValueError::new_err)
    }
    /// Clear the residual while retaining the configured threshold.
    fn reset(&mut self) {
        self.inner.reset();
    }
    /// Return a detached dictionary containing the residual state.
    fn get_state(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let d = PyDict::new(py);
        d.set_item("sigma", self.inner.sigma)?;
        Ok(d.into_any().unbind())
    }
}
/// Simulate into owning arrays, propagating input and allocation exceptions.
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
    let inputs = currents.as_slice()?;
    let trace = zeros_1d::<f64>(py, inputs.len())?;
    let events = zeros_1d::<i32>(py, inputs.len())?;
    // SAFETY: These fresh contiguous arrays own disjoint writable storage.
    // Their owners outlive the slices and no external aliases are exposed.
    let (sigma_values, event_values) = unsafe { (trace.as_slice_mut()?, events.as_slice_mut()?) };
    for (index, &current) in inputs.iter().enumerate() {
        event_values[index] = n.try_step(current).map_err(PyValueError::new_err)?;
        sigma_values[index] = n.sigma;
    }
    let d = PyDict::new(py);
    d.set_item("sigma", trace)?;
    d.set_item("events", events)?;
    d.set_item("sigma_final", n.sigma)?;
    Ok(d.into_any().unbind())
}
/// Register the distinct retained project class and batch function.
pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PySCSigmaDeltaAccumulatorNeuron>()?;
    module.add_function(wrap_pyfunction!(
        py_sc_sigma_delta_accumulator_simulate,
        module
    )?)?;
    Ok(())
}
