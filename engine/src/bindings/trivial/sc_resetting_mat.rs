// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — SC resetting-MAT PyO3 binding

use numpy::{PyArrayMethods, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::fixed_point_lif_binding::array_allocation::zeros_1d;
use crate::neurons;

/// Python-owned complete SC resetting-MAT neuron.
#[pyclass(
    name = "SCResettingMATNeuron",
    module = "sc_neurocore_engine.sc_neurocore_engine"
)]
#[derive(Clone)]
pub struct PySCResettingMATNeuron {
    inner: neurons::SCResettingMATNeuron,
}

#[pymethods]
impl PySCResettingMATNeuron {
    /// Construct a configured SC resetting-MAT neuron.
    #[new]
    #[pyo3(signature = (
        v=-70.0, theta1=0.0, theta2=0.0, v_rest=-70.0, v_reset=-70.0,
        v_threshold_base=-50.0, tau_m=10.0, tau_1=10.0, tau_2=200.0,
        h1=5.0, h2=3.0, resistance=1.0, dt=1.0
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        v: f64,
        theta1: f64,
        theta2: f64,
        v_rest: f64,
        v_reset: f64,
        v_threshold_base: f64,
        tau_m: f64,
        tau_1: f64,
        tau_2: f64,
        h1: f64,
        h2: f64,
        resistance: f64,
        dt: f64,
    ) -> PyResult<Self> {
        let inner = neurons::SCResettingMATNeuron {
            v,
            theta1,
            theta2,
            v_rest,
            v_reset,
            v_threshold_base,
            tau_m,
            tau_1,
            tau_2,
            h1,
            h2,
            resistance,
            dt,
        };
        if !inner.validate() {
            return Err(PyValueError::new_err(
                "invalid SC resetting-MAT state or configuration",
            ));
        }
        Ok(Self { inner })
    }

    /// Advance one atomic candidate-first RK4/reset step.
    fn step(&mut self, current: f64) -> PyResult<i32> {
        self.inner.try_step(current).map_err(PyValueError::new_err)
    }

    /// Reset dynamic state while retaining configuration.
    fn reset(&mut self) -> PyResult<()> {
        self.inner.try_reset().map_err(PyValueError::new_err)
    }

    /// Return all three dynamic state fields.
    fn get_state(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let state = PyDict::new(py);
        state.set_item("v", self.inner.v)?;
        state.set_item("theta1", self.inner.theta1)?;
        state.set_item("theta2", self.inner.theta2)?;
        Ok(state.into_any().unbind())
    }
}

/// Simulate one configured SC resetting-MAT trace natively.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (
    v, theta1, theta2, v_rest, v_reset, v_threshold_base, tau_m, tau_1,
    tau_2, h1, h2, resistance, dt, currents
))]
fn py_sc_resetting_mat_simulate<'py>(
    py: Python<'py>,
    v: f64,
    theta1: f64,
    theta2: f64,
    v_rest: f64,
    v_reset: f64,
    v_threshold_base: f64,
    tau_m: f64,
    tau_1: f64,
    tau_2: f64,
    h1: f64,
    h2: f64,
    resistance: f64,
    dt: f64,
    currents: PyReadonlyArray1<'py, f64>,
) -> PyResult<Py<PyAny>> {
    let mut neuron = neurons::SCResettingMATNeuron {
        v,
        theta1,
        theta2,
        v_rest,
        v_reset,
        v_threshold_base,
        tau_m,
        tau_1,
        tau_2,
        h1,
        h2,
        resistance,
        dt,
    };
    if !neuron.validate() {
        return Err(PyValueError::new_err(
            "invalid SC resetting-MAT state or configuration",
        ));
    }
    let inputs = currents.as_slice()?;
    let voltages = zeros_1d::<f64>(py, inputs.len())?;
    let theta1_trace = zeros_1d::<f64>(py, inputs.len())?;
    let theta2_trace = zeros_1d::<f64>(py, inputs.len())?;
    let events = zeros_1d::<i32>(py, inputs.len())?;
    // SAFETY: These fresh contiguous arrays own disjoint writable storage.
    // The owners outlive the slices and no external aliases are exposed.
    let (voltages_values, theta1_trace_values, theta2_trace_values, events_values) = unsafe {
        (
            voltages.as_slice_mut()?,
            theta1_trace.as_slice_mut()?,
            theta2_trace.as_slice_mut()?,
            events.as_slice_mut()?,
        )
    };
    for (index, &current) in inputs.iter().enumerate() {
        events_values[index] = neuron.try_step(current).map_err(PyValueError::new_err)?;
        voltages_values[index] = neuron.v;
        theta1_trace_values[index] = neuron.theta1;
        theta2_trace_values[index] = neuron.theta2;
    }
    let result = PyDict::new(py);
    result.set_item("voltages", voltages)?;
    result.set_item("theta1", theta1_trace)?;
    result.set_item("theta2", theta2_trace)?;
    result.set_item("events", events)?;
    result.set_item("v_final", neuron.v)?;
    result.set_item("theta1_final", neuron.theta1)?;
    result.set_item("theta2_final", neuron.theta2)?;
    Ok(result.into_any().unbind())
}

/// Register the SC resetting-MAT class and native batch simulator.
pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PySCResettingMATNeuron>()?;
    module.add_function(wrap_pyfunction!(py_sc_resetting_mat_simulate, module)?)?;
    Ok(())
}
