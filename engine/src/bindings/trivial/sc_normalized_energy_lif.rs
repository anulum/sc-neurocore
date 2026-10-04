// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Retained normalized energy-LIF Python binding

//! PyO3 exposure for retained normalized energy-gated SC LIF.

use crate::neurons;
use numpy::{PyArrayMethods, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::fixed_point_lif_binding::array_allocation::zeros_1d;

/// Python-owned normalized energy state and its complete configuration.
#[pyclass(
    name = "SCNormalizedEnergyLIFNeuron",
    module = "sc_neurocore_engine.sc_neurocore_engine"
)]
#[derive(Clone)]
pub struct PySCNormalizedEnergyLIFNeuron {
    inner: neurons::SCNormalizedEnergyLIFNeuron,
}
#[pymethods]
impl PySCNormalizedEnergyLIFNeuron {
    /// Construct a state whose configured reset remains inside its envelope.
    #[new]
    #[pyo3(signature=(v=-70.0,epsilon=1.0,v_rest=-70.0,v_reset=-70.0,v_threshold=-50.0,tau_m=10.0,tau_e=500.0,alpha=0.1,epsilon_0=1.0,resistance=1.0,dt=1.0))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        v: f64,
        epsilon: f64,
        v_rest: f64,
        v_reset: f64,
        v_threshold: f64,
        tau_m: f64,
        tau_e: f64,
        alpha: f64,
        epsilon_0: f64,
        resistance: f64,
        dt: f64,
    ) -> PyResult<Self> {
        let inner = neurons::SCNormalizedEnergyLIFNeuron {
            v,
            epsilon,
            v_rest,
            v_reset,
            v_threshold,
            tau_m,
            tau_e,
            alpha,
            epsilon_0,
            resistance,
            dt,
        };
        if !inner.valid() {
            return Err(PyValueError::new_err("invalid SC normalized EnergyLIF"));
        }
        Ok(Self { inner })
    }
    /// Advance one atomic exact-flow sample with a finite current.
    fn step(&mut self, current: f64) -> PyResult<i32> {
        self.inner.try_step(current).map_err(PyValueError::new_err)
    }
    /// Restore a validated resting state without partial mutation.
    fn reset(&mut self) -> PyResult<()> {
        self.inner.try_reset().map_err(PyValueError::new_err)
    }
    /// Return an independent dictionary containing both dynamic values.
    fn get_state(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let d = PyDict::new(py);
        d.set_item("v", self.inner.v)?;
        d.set_item("epsilon", self.inner.epsilon)?;
        Ok(d.into_any().unbind())
    }
}
/// Simulate into owning arrays and propagate layout or allocation refusal.
#[pyfunction]
#[pyo3(signature=(v,epsilon,v_rest,v_reset,v_threshold,tau_m,tau_e,alpha,epsilon_0,resistance,dt,currents))]
#[allow(clippy::too_many_arguments)]
fn py_sc_normalized_energy_lif_simulate<'py>(
    py: Python<'py>,
    v: f64,
    epsilon: f64,
    v_rest: f64,
    v_reset: f64,
    v_threshold: f64,
    tau_m: f64,
    tau_e: f64,
    alpha: f64,
    epsilon_0: f64,
    resistance: f64,
    dt: f64,
    currents: PyReadonlyArray1<'py, f64>,
) -> PyResult<Py<PyAny>> {
    let mut n = neurons::SCNormalizedEnergyLIFNeuron {
        v,
        epsilon,
        v_rest,
        v_reset,
        v_threshold,
        tau_m,
        tau_e,
        alpha,
        epsilon_0,
        resistance,
        dt,
    };
    if !n.valid() {
        return Err(PyValueError::new_err("invalid SC normalized EnergyLIF"));
    }
    let inputs = currents.as_slice()?;
    let voltages = zeros_1d::<f64>(py, inputs.len())?;
    let energies = zeros_1d::<f64>(py, inputs.len())?;
    let events = zeros_1d::<i32>(py, inputs.len())?;
    // SAFETY: These fresh contiguous arrays own disjoint writable storage.
    // Their owners outlive the slices and no external aliases are exposed.
    let (voltage_values, energy_values, event_values) = unsafe {
        (
            voltages.as_slice_mut()?,
            energies.as_slice_mut()?,
            events.as_slice_mut()?,
        )
    };
    for (index, &current) in inputs.iter().enumerate() {
        event_values[index] = n.try_step(current).map_err(PyValueError::new_err)?;
        voltage_values[index] = n.v;
        energy_values[index] = n.epsilon;
    }
    let d = PyDict::new(py);
    d.set_item("voltages", voltages)?;
    d.set_item("epsilon", energies)?;
    d.set_item("events", events)?;
    d.set_item("v_final", n.v)?;
    d.set_item("epsilon_final", n.epsilon)?;
    Ok(d.into_any().unbind())
}
/// Register the retained SC class and complete batch function.
pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PySCNormalizedEnergyLIFNeuron>()?;
    module.add_function(wrap_pyfunction!(
        py_sc_normalized_energy_lif_simulate,
        module
    )?)?;
    Ok(())
}
