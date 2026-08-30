// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Adaptive exponential neuron PyO3 binding

use numpy::{IntoPyArray, PyArray1};
use pyo3::exceptions::PyFloatingPointError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::neuron;

type CompleteTracePacket<'py> = (
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<u8>>,
    f64,
    f64,
);

#[pyclass(
    name = "AdExNeuron",
    module = "sc_neurocore_engine.sc_neurocore_engine"
)]
#[derive(Clone)]
pub struct PyAdExNeuron {
    inner: neuron::AdExNeuron,
}

#[pymethods]
impl PyAdExNeuron {
    #[new]
    fn new() -> Self {
        Self {
            inner: neuron::AdExNeuron::new(),
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
        d.set_item("w", self.inner.w)?;
        Ok(d.into_any().unbind())
    }
}

/// Register the adaptive exponential neuron class.
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyAdExNeuron>()?;
    module.add_function(wrap_pyfunction!(adex_simulate_complete, module)?)?;
    Ok(())
}

/// Run the checked full-parameter AdEx batch and return every state/event row.
#[pyfunction]
#[pyo3(signature = (
    v, w, v_rest, v_reset, v_threshold, v_rh, delta_t, tau, tau_w,
    a, b, c_m, dt, n_steps, current
))]
#[allow(clippy::too_many_arguments)]
fn adex_simulate_complete<'py>(
    py: Python<'py>,
    v: f64,
    w: f64,
    v_rest: f64,
    v_reset: f64,
    v_threshold: f64,
    v_rh: f64,
    delta_t: f64,
    tau: f64,
    tau_w: f64,
    a: f64,
    b: f64,
    c_m: f64,
    dt: f64,
    n_steps: usize,
    current: f64,
) -> PyResult<CompleteTracePacket<'py>> {
    let mut model = neuron::AdExNeuron {
        v,
        w,
        v_rest,
        v_reset,
        v_threshold,
        v_rh,
        delta_t,
        tau,
        tau_w,
        a,
        b,
        c_m,
        dt,
    };
    let (v_trace, w_trace, event_trace) = model
        .simulate_complete(n_steps, current)
        .map_err(PyFloatingPointError::new_err)?;
    Ok((
        v_trace.into_pyarray(py),
        w_trace.into_pyarray(py),
        event_trace.into_pyarray(py),
        model.v,
        model.w,
    ))
}
