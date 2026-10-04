// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Source MAT(1) native Python binding

//! PyO3 exposure for the complete source MAT(1) contract.

use numpy::{PyArrayMethods, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::fixed_point_lif_binding::array_allocation::zeros_1d;
use crate::neurons;

/// Python-owned complete source MAT(1) neuron.
#[pyclass(
    name = "NonResettingLIFNeuron",
    module = "sc_neurocore_engine.sc_neurocore_engine"
)]
#[derive(Clone)]
pub struct PyNonResettingLIFNeuron {
    inner: neurons::NonResettingLIFNeuron,
}

#[pymethods]
impl PyNonResettingLIFNeuron {
    /// Construct a configured MAT(1) neuron.
    #[new]
    #[pyo3(signature = (v=0.0, theta=0.0, refractory_remaining=0.0, omega=19.0, tau_m=5.0, tau_theta=50.0, alpha=37.0, resistance=50.0, refractory_period=2.0, dt=0.001))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        v: f64,
        theta: f64,
        refractory_remaining: f64,
        omega: f64,
        tau_m: f64,
        tau_theta: f64,
        alpha: f64,
        resistance: f64,
        refractory_period: f64,
        dt: f64,
    ) -> PyResult<Self> {
        let inner = neurons::NonResettingLIFNeuron {
            v,
            theta,
            refractory_remaining,
            omega,
            tau_m,
            tau_theta,
            alpha,
            resistance,
            refractory_period,
            dt,
        };
        if !inner.validate() {
            return Err(PyValueError::new_err(
                "invalid MAT(1) state or configuration",
            ));
        }
        Ok(Self { inner })
    }
    /// Advance one atomic source-model step.
    fn step(&mut self, current: f64) -> PyResult<i32> {
        self.inner.try_step(current).map_err(PyValueError::new_err)
    }
    /// Reset dynamic state while retaining configuration.
    fn reset(&mut self) {
        self.inner.reset();
    }
    /// Return all three dynamic state fields.
    fn get_state(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let state = PyDict::new(py);
        state.set_item("v", self.inner.v)?;
        state.set_item("theta", self.inner.theta)?;
        state.set_item("refractory_remaining", self.inner.refractory_remaining)?;
        Ok(state.into_any().unbind())
    }
}

/// Simulate MAT(1) into owned arrays, propagating input and allocation errors.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature=(v,theta,refractory_remaining,omega,tau_m,tau_theta,alpha,resistance,refractory_period,dt,currents))]
fn py_non_resetting_lif_simulate<'py>(
    py: Python<'py>,
    v: f64,
    theta: f64,
    refractory_remaining: f64,
    omega: f64,
    tau_m: f64,
    tau_theta: f64,
    alpha: f64,
    resistance: f64,
    refractory_period: f64,
    dt: f64,
    currents: PyReadonlyArray1<'py, f64>,
) -> PyResult<Py<PyAny>> {
    let mut neuron = neurons::NonResettingLIFNeuron {
        v,
        theta,
        refractory_remaining,
        omega,
        tau_m,
        tau_theta,
        alpha,
        resistance,
        refractory_period,
        dt,
    };
    if !neuron.validate() {
        return Err(PyValueError::new_err(
            "invalid MAT(1) state or configuration",
        ));
    }
    let inputs = currents.as_slice()?;
    let voltages = zeros_1d::<f64>(py, inputs.len())?;
    let thresholds = zeros_1d::<f64>(py, inputs.len())?;
    let refractory = zeros_1d::<f64>(py, inputs.len())?;
    let events = zeros_1d::<i32>(py, inputs.len())?;
    // SAFETY: The four freshly allocated arrays are contiguous and have no
    // external aliases. Their owners outlive these disjoint writable slices.
    let (voltage_values, threshold_values, refractory_values, event_values) = unsafe {
        (
            voltages.as_slice_mut()?,
            thresholds.as_slice_mut()?,
            refractory.as_slice_mut()?,
            events.as_slice_mut()?,
        )
    };
    for (index, &current) in inputs.iter().enumerate() {
        event_values[index] = neuron.try_step(current).map_err(PyValueError::new_err)?;
        voltage_values[index] = neuron.v;
        threshold_values[index] = neuron.theta;
        refractory_values[index] = neuron.refractory_remaining;
    }
    let result = PyDict::new(py);
    result.set_item("voltages", voltages)?;
    result.set_item("theta", thresholds)?;
    result.set_item("refractory", refractory)?;
    result.set_item("events", events)?;
    result.set_item("v_final", neuron.v)?;
    result.set_item("theta_final", neuron.theta)?;
    result.set_item("refractory_final", neuron.refractory_remaining)?;
    Ok(result.into_any().unbind())
}

/// Register the source MAT(1) class and batch simulator.
pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyNonResettingLIFNeuron>()?;
    module.add_function(wrap_pyfunction!(py_non_resetting_lif_simulate, module)?)?;
    Ok(())
}
