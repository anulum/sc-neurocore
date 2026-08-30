// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Exponential integrate-and-fire PyO3 binding

//! Python binding for the exponential integrate-and-fire neuron.

use numpy::{IntoPyArray, PyArray1};
use pyo3::exceptions::PyFloatingPointError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::neuron;

/// Register the exponential integrate-and-fire neuron with the extension module.
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyExpIFNeuron>()?;
    module.add_function(wrap_pyfunction!(expif_simulate_complete, module)?)?;
    Ok(())
}

type CompleteTracePacket<'py> = (
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<u8>>,
    f64,
    f64,
);

/// Run the full-parameter checked ExpIF recurrence across one Rust boundary.
#[pyfunction]
#[pyo3(signature = (
    v, v_rest, v_reset, v_threshold, v_rh, delta_t, tau, dt,
    refractory_period, refractory_remaining, source_profile, n_steps, current
))]
#[allow(clippy::too_many_arguments)]
fn expif_simulate_complete<'py>(
    py: Python<'py>,
    v: f64,
    v_rest: f64,
    v_reset: f64,
    v_threshold: f64,
    v_rh: f64,
    delta_t: f64,
    tau: f64,
    dt: f64,
    refractory_period: f64,
    refractory_remaining: f64,
    source_profile: bool,
    n_steps: usize,
    current: f64,
) -> PyResult<CompleteTracePacket<'py>> {
    let mut model = neuron::ExpIfNeuron {
        v,
        v_rest,
        v_reset,
        v_threshold,
        v_rh,
        delta_t,
        tau,
        dt,
        refractory_period,
        refractory_remaining,
        source_profile,
        inv_delta_t: 1.0 / delta_t,
        dt_div_tau: dt / tau,
    };
    let (voltage, refractory, events) =
        model.simulate_complete(n_steps, current).map_err(|error| {
            PyFloatingPointError::new_err(format!("ExpIF batch rejected: {error:?}"))
        })?;
    Ok((
        voltage.into_pyarray(py),
        refractory.into_pyarray(py),
        events.into_pyarray(py),
        model.v,
        model.refractory_remaining,
    ))
}

/// Register the historical mixed-case class alias.
pub(crate) fn register_legacy_alias(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyExpIfNeuron>()?;
    Ok(())
}

#[pyclass(
    name = "ExpIFNeuron",
    module = "sc_neurocore_engine.sc_neurocore_engine"
)]
#[derive(Clone)]
pub struct PyExpIFNeuron {
    inner: neuron::ExpIfNeuron,
}

#[pymethods]
impl PyExpIFNeuron {
    #[new]
    fn new() -> Self {
        Self {
            inner: neuron::ExpIfNeuron::new(),
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
        d.set_item("refractory_remaining", self.inner.refractory_remaining)?;
        Ok(d.into_any().unbind())
    }
}

#[pyclass(
    name = "ExpIfNeuron",
    module = "sc_neurocore_engine.sc_neurocore_engine"
)]
#[derive(Clone)]
pub struct PyExpIfNeuron {
    inner: neuron::ExpIfNeuron,
}

#[pymethods]
impl PyExpIfNeuron {
    #[new]
    fn new() -> Self {
        Self {
            inner: neuron::ExpIfNeuron::new(),
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
        d.set_item("refractory_remaining", self.inner.refractory_remaining)?;
        Ok(d.into_any().unbind())
    }
}
