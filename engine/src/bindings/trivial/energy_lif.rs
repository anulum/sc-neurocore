// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li

//! PyO3 exposure for source-faithful Fardet-Levina eLIF.

use crate::neurons;
use numpy::{IntoPyArray, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

#[pyclass(
    name = "EnergyLIFNeuron",
    module = "sc_neurocore_engine.sc_neurocore_engine"
)]
#[derive(Clone)]
pub struct PyEnergyLIFNeuron {
    inner: neurons::EnergyLIFNeuron,
}

#[pymethods]
impl PyEnergyLIFNeuron {
    #[new]
    #[pyo3(signature=(v=-61.0,epsilon=0.32,capacitance=100.0,g_leak=9.0,e_0=-62.5,e_u=-58.5,e_d=-40.0,e_f=-62.0,v_threshold=-59.0,v_reset=-62.0,alpha=1.0,epsilon_0=0.5,epsilon_c=0.18,delta=0.01,tau_e=200.0,dt=0.1))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        v: f64,
        epsilon: f64,
        capacitance: f64,
        g_leak: f64,
        e_0: f64,
        e_u: f64,
        e_d: f64,
        e_f: f64,
        v_threshold: f64,
        v_reset: f64,
        alpha: f64,
        epsilon_0: f64,
        epsilon_c: f64,
        delta: f64,
        tau_e: f64,
        dt: f64,
    ) -> PyResult<Self> {
        let inner = neurons::EnergyLIFNeuron {
            v,
            epsilon,
            capacitance,
            g_leak,
            e_0,
            e_u,
            e_d,
            e_f,
            v_threshold,
            v_reset,
            alpha,
            epsilon_0,
            epsilon_c,
            delta,
            tau_e,
            dt,
        };
        if !inner.valid() {
            return Err(PyValueError::new_err(
                "invalid EnergyLIF state or configuration",
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
        d.set_item("epsilon", self.inner.epsilon)?;
        Ok(d.into_any().unbind())
    }
}

#[pyfunction]
#[pyo3(signature=(v,epsilon,capacitance,g_leak,e_0,e_u,e_d,e_f,v_threshold,v_reset,alpha,epsilon_0,epsilon_c,delta,tau_e,dt,currents))]
#[allow(clippy::too_many_arguments)]
fn py_energy_lif_simulate<'py>(
    py: Python<'py>,
    v: f64,
    epsilon: f64,
    capacitance: f64,
    g_leak: f64,
    e_0: f64,
    e_u: f64,
    e_d: f64,
    e_f: f64,
    v_threshold: f64,
    v_reset: f64,
    alpha: f64,
    epsilon_0: f64,
    epsilon_c: f64,
    delta: f64,
    tau_e: f64,
    dt: f64,
    currents: PyReadonlyArray1<'py, f64>,
) -> PyResult<Py<PyAny>> {
    let mut n = neurons::EnergyLIFNeuron {
        v,
        epsilon,
        capacitance,
        g_leak,
        e_0,
        e_u,
        e_d,
        e_f,
        v_threshold,
        v_reset,
        alpha,
        epsilon_0,
        epsilon_c,
        delta,
        tau_e,
        dt,
    };
    if !n.valid() {
        return Err(PyValueError::new_err(
            "invalid EnergyLIF state or configuration",
        ));
    }
    let mut voltages = Vec::with_capacity(currents.len()?);
    let mut energies = Vec::with_capacity(currents.len()?);
    let mut events = Vec::with_capacity(currents.len()?);
    for &current in currents.as_slice()? {
        events.push(n.try_step(current).map_err(PyValueError::new_err)?);
        voltages.push(n.v);
        energies.push(n.epsilon);
    }
    let d = PyDict::new(py);
    d.set_item("voltages", voltages.into_pyarray(py))?;
    d.set_item("epsilon", energies.into_pyarray(py))?;
    d.set_item("events", events.into_pyarray(py))?;
    d.set_item("v_final", n.v)?;
    d.set_item("epsilon_final", n.epsilon)?;
    Ok(d.into_any().unbind())
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyEnergyLIFNeuron>()?;
    module.add_function(wrap_pyfunction!(py_energy_lif_simulate, module)?)?;
    Ok(())
}
