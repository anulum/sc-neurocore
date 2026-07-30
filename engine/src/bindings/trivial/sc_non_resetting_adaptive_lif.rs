// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li

//! PyO3 exposure for the retained SC adaptive-LIF contract.

use crate::neurons;
use numpy::{IntoPyArray, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Python-owned retained project neuron.
#[pyclass(
    name = "SCNonResettingAdaptiveLIFNeuron",
    module = "sc_neurocore_engine.sc_neurocore_engine"
)]
#[derive(Clone)]
pub struct PySCNonResettingAdaptiveLIFNeuron {
    inner: neurons::SCNonResettingAdaptiveLIFNeuron,
}

#[pymethods]
impl PySCNonResettingAdaptiveLIFNeuron {
    /// Construct a configured retained-project neuron.
    #[new]
    #[pyo3(signature=(v=-65.0,theta=-50.0,v_rest=-65.0,theta_rest=-50.0,delta_theta=5.0,tau_m=10.0,tau_theta=50.0,r_m=1.0,dt=0.1))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        v: f64,
        theta: f64,
        v_rest: f64,
        theta_rest: f64,
        delta_theta: f64,
        tau_m: f64,
        tau_theta: f64,
        r_m: f64,
        dt: f64,
    ) -> PyResult<Self> {
        let inner = neurons::SCNonResettingAdaptiveLIFNeuron {
            v,
            theta,
            v_rest,
            theta_rest,
            delta_theta,
            tau_m,
            tau_theta,
            r_m,
            dt,
        };
        if !inner.validate() {
            return Err(PyValueError::new_err(
                "invalid SC adaptive LIF state or configuration",
            ));
        }
        Ok(Self { inner })
    }
    /// Advance one atomic project step.
    fn step(&mut self, current: f64) -> PyResult<i32> {
        self.inner.try_step(current).map_err(PyValueError::new_err)
    }
    /// Reset dynamic state while retaining configuration.
    fn reset(&mut self) {
        self.inner.reset();
    }
    /// Return both dynamic state fields.
    fn get_state(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let state = PyDict::new(py);
        state.set_item("v", self.inner.v)?;
        state.set_item("theta", self.inner.theta)?;
        Ok(state.into_any().unbind())
    }
}

/// Simulate one configured retained-project trace.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature=(v,theta,v_rest,theta_rest,delta_theta,tau_m,tau_theta,r_m,dt,currents))]
fn py_sc_non_resetting_adaptive_lif_simulate<'py>(
    py: Python<'py>,
    v: f64,
    theta: f64,
    v_rest: f64,
    theta_rest: f64,
    delta_theta: f64,
    tau_m: f64,
    tau_theta: f64,
    r_m: f64,
    dt: f64,
    currents: PyReadonlyArray1<'py, f64>,
) -> PyResult<Py<PyAny>> {
    let mut neuron = neurons::SCNonResettingAdaptiveLIFNeuron {
        v,
        theta,
        v_rest,
        theta_rest,
        delta_theta,
        tau_m,
        tau_theta,
        r_m,
        dt,
    };
    if !neuron.validate() {
        return Err(PyValueError::new_err(
            "invalid SC adaptive LIF state or configuration",
        ));
    }
    let inputs = currents.as_slice()?;
    let mut voltages = Vec::with_capacity(inputs.len());
    let mut thresholds = Vec::with_capacity(inputs.len());
    let mut events = Vec::with_capacity(inputs.len());
    for &current in inputs {
        events.push(neuron.try_step(current).map_err(PyValueError::new_err)?);
        voltages.push(neuron.v);
        thresholds.push(neuron.theta);
    }
    let result = PyDict::new(py);
    result.set_item("voltages", voltages.into_pyarray(py))?;
    result.set_item("theta", thresholds.into_pyarray(py))?;
    result.set_item("events", events.into_pyarray(py))?;
    result.set_item("v_final", neuron.v)?;
    result.set_item("theta_final", neuron.theta)?;
    Ok(result.into_any().unbind())
}

/// Register the retained project class and batch simulator.
pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PySCNonResettingAdaptiveLIFNeuron>()?;
    module.add_function(wrap_pyfunction!(
        py_sc_non_resetting_adaptive_lif_simulate,
        module
    )?)?;
    Ok(())
}
