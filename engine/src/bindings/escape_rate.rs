// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — EscapeRate PyO3 binding

//! Python binding for the Gerstner 2000 stochastic-threshold cell.

use numpy::{IntoPyArray, PyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::neurons;

/// Register the EscapeRate simulator with the extension module.
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyEscapeRateNeuron>()?;
    module.add_function(wrap_pyfunction!(py_escape_rate_simulate, module)?)?;
    Ok(())
}

// EscapeRateNeuron needs seed
#[pyclass(
    name = "EscapeRateNeuron",
    module = "sc_neurocore_engine.sc_neurocore_engine"
)]
#[derive(Clone)]
pub struct PyEscapeRateNeuron {
    inner: neurons::EscapeRateNeuron,
}

#[pymethods]
impl PyEscapeRateNeuron {
    #[new]
    #[pyo3(signature = (seed=0xACE1))]
    fn new(seed: u16) -> Self {
        Self {
            inner: neurons::EscapeRateNeuron::new(u64::from(seed)),
        }
    }
    fn step(&mut self, current: f64) -> i32 {
        self.inner.step(current)
    }
    fn reset(&mut self) {
        self.inner.reset();
    }
    fn get_state(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let d = PyDict::new(py);
        d.set_item("v", self.inner.v)?;
        d.set_item("rng_state", self.inner.rng_state)?;
        d.set_item("initial_seed", self.inner.initial_seed)?;
        Ok(d.into_any().unbind())
    }
}

/// Full-contract seeded EscapeRate batch with the canonical LFSR16 trial.
#[pyfunction]
#[pyo3(signature = (
    v0, v_rest, v_reset, v_threshold, tau_m, rho_0, delta_u, resistance,
    dt, rng_state, n_steps, current
))]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn py_escape_rate_simulate<'py>(
    py: Python<'py>,
    v0: f64,
    v_rest: f64,
    v_reset: f64,
    v_threshold: f64,
    tau_m: f64,
    rho_0: f64,
    delta_u: f64,
    resistance: f64,
    dt: f64,
    rng_state: u16,
    n_steps: usize,
    current: f64,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<u8>>,
    f64,
    u16,
)> {
    let mut neuron = crate::neurons::EscapeRateNeuron {
        v: v0,
        v_rest,
        v_reset,
        v_threshold,
        tau_m,
        rho_0,
        delta_u,
        resistance,
        dt,
        rng_state,
        initial_seed: rng_state,
    };
    if !neuron.valid() || !current.is_finite() {
        return Err(PyValueError::new_err(
            "invalid EscapeRate simulation state or input",
        ));
    }
    let mut trace = Vec::with_capacity(n_steps);
    let mut events = Vec::with_capacity(n_steps);
    for _ in 0..n_steps {
        let spike = neuron.try_step(current).map_err(PyValueError::new_err)?;
        trace.push(neuron.v);
        events.push(spike as u8);
    }
    Ok((
        trace.into_pyarray(py),
        events.into_pyarray(py),
        neuron.v,
        neuron.rng_state,
    ))
}
