// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Profile-explicit perfect-integrator PyO3 boundary

use numpy::{IntoPyArray, PyArray1};
use pyo3::exceptions::{PyFloatingPointError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::neurons;

#[pyclass(
    name = "PerfectIntegratorNeuron",
    module = "sc_neurocore_engine.sc_neurocore_engine"
)]
#[derive(Clone)]
pub struct PyPerfectIntegratorNeuron {
    inner: neurons::PerfectIntegratorNeuron,
}

#[pymethods]
impl PyPerfectIntegratorNeuron {
    #[new]
    #[pyo3(signature = (c_m=1.0, v_threshold=1.0, dt=0.1))]
    fn new(c_m: f64, v_threshold: f64, dt: f64) -> PyResult<Self> {
        let inner = neurons::PerfectIntegratorNeuron::new(c_m, v_threshold, dt);
        if !inner.valid() {
            return Err(PyValueError::new_err(
                "PerfectIntegrator parameters violate the finite positive-capacitance contract",
            ));
        }
        Ok(Self { inner })
    }

    fn step(&mut self, current: f64) -> PyResult<i32> {
        self.inner
            .try_step(current)
            .map_err(|error| PyFloatingPointError::new_err(error.to_string()))
    }

    fn reset(&mut self) {
        self.inner.reset();
    }

    fn get_state(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let state = PyDict::new(py);
        state.set_item("v", self.inner.v)?;
        Ok(state.into_any().unbind())
    }
}

type PerfectIntegratorCompletePacket<'py> =
    (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<u8>>, f64);

/// Execute either the source or preserved SC profile in one checked boundary.
#[pyfunction]
#[pyo3(signature = (
    v, c_m, v_threshold, v_reset, dt, source_profile, n_steps, current
))]
#[allow(clippy::too_many_arguments)]
fn perfect_integrator_simulate_complete<'py>(
    py: Python<'py>,
    v: f64,
    c_m: f64,
    v_threshold: f64,
    v_reset: f64,
    dt: f64,
    source_profile: bool,
    n_steps: usize,
    current: f64,
) -> PyResult<PerfectIntegratorCompletePacket<'py>> {
    let model = neurons::PerfectIntegratorNeuron {
        v,
        c_m,
        v_threshold,
        v_reset,
        dt,
        source_profile,
    };
    let (voltage, events, final_v) =
        model.simulate_complete(n_steps, current).map_err(|error| {
            PyFloatingPointError::new_err(format!("PerfectIntegrator batch rejected: {error}"))
        })?;
    Ok((voltage.into_pyarray(py), events.into_pyarray(py), final_v))
}

/// Register the perfect-integrator class and complete-batch function.
pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyPerfectIntegratorNeuron>()?;
    module.add_function(wrap_pyfunction!(
        perfect_integrator_simulate_complete,
        module
    )?)?;
    Ok(())
}
