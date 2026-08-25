// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — PyO3 binding for SC stochastic rate adaptation

use numpy::{IntoPyArray, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::neurons;

#[pyclass(
    name = "SCStochasticRateAdaptationNeuron",
    module = "sc_neurocore_engine.sc_neurocore_engine"
)]
#[derive(Clone)]
pub struct PySCStochasticRateAdaptationNeuron {
    inner: neurons::SCStochasticRateAdaptationNeuron,
}

#[pymethods]
impl PySCStochasticRateAdaptationNeuron {
    #[new]
    #[pyo3(signature = (seed=42))]
    fn new(seed: u64) -> Self {
        Self {
            inner: neurons::SCStochasticRateAdaptationNeuron::new(seed),
        }
    }

    fn step(&mut self, current: f64) -> i32 {
        self.inner.step(current)
    }
    fn step_with_uniform(&mut self, current: f64, uniform: f64) -> i32 {
        self.inner.step_with_uniform(current, uniform)
    }
    fn reset(&mut self) {
        self.inner.reset();
    }
    fn get_state(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let d = PyDict::new(py);
        d.set_item("a", self.inner.a)?;
        Ok(d.into_any().unbind())
    }
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PySCStochasticRateAdaptationNeuron>()?;
    module.add_function(wrap_pyfunction!(
        py_sc_stochastic_rate_adaptation_simulate,
        module
    )?)?;
    Ok(())
}

#[pyfunction]
#[pyo3(signature=(a,f_max,beta,i_half,tau_a,delta_a,dt,currents,uniforms))]
#[allow(clippy::too_many_arguments)]
fn py_sc_stochastic_rate_adaptation_simulate<'py>(
    py: Python<'py>,
    a: f64,
    f_max: f64,
    beta: f64,
    i_half: f64,
    tau_a: f64,
    delta_a: f64,
    dt: f64,
    currents: PyReadonlyArray1<'py, f64>,
    uniforms: PyReadonlyArray1<'py, f64>,
) -> PyResult<Py<PyAny>> {
    if currents.len()? != uniforms.len()? {
        return Err(PyValueError::new_err("current/uniform length mismatch"));
    }
    let mut n = neurons::SCStochasticRateAdaptationNeuron::new(0);
    n.a = a;
    n.f_max = f_max;
    n.beta = beta;
    n.i_half = i_half;
    n.tau_a = tau_a;
    n.delta_a = delta_a;
    n.dt = dt;
    if !n.valid() {
        return Err(PyValueError::new_err(
            "invalid SC stochastic rate-adaptation configuration",
        ));
    }
    let mut adaptation = Vec::with_capacity(currents.len()?);
    let mut events = Vec::with_capacity(currents.len()?);
    for (&current, &uniform) in currents.as_slice()?.iter().zip(uniforms.as_slice()?) {
        let event = n.step_with_uniform(current, uniform);
        if event < 0 {
            return Err(PyValueError::new_err("invalid SC stochastic transition"));
        }
        adaptation.push(n.a);
        events.push(event);
    }
    let d = PyDict::new(py);
    d.set_item("adaptation", adaptation.into_pyarray(py))?;
    d.set_item("events", events.into_pyarray(py))?;
    d.set_item("a_final", n.a)?;
    Ok(d.into_any().unbind())
}
