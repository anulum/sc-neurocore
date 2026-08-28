// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — source-faithful Mihalas-Niebur PyO3 binding

//! Python class and failure-signalling batch surface for Mihalaş-Niebur dynamics.

use numpy::{IntoPyArray, PyArray1};
use pyo3::exceptions::PyFloatingPointError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::neurons::MihalasNieburNeuron;

type MihalasNieburBatchOutput<'py> = (Bound<'py, PyArray1<f64>>, i64, f64, f64, f64, f64);

py_neuron_default!(
    "MihalasNieburNeuron",
    PyMihalasNieburNeuron,
    MihalasNieburNeuron,
    state v,
    state theta,
    state i1,
    state i2
);

/// Register the Mihalas-Niebur class and batch simulator.
pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyMihalasNieburNeuron>()?;
    module.add_function(wrap_pyfunction!(py_mihalas_niebur_simulate, module)?)?;
    Ok(())
}

/// Execute equations 2.1–2.2 with fixed-grid RK4 and the published reset map.
#[pyfunction]
#[pyo3(signature = (
    v0, theta0, i1_0, i2_0, v_rest, v_reset, theta_reset, theta_inf,
    leak_rate, threshold_voltage_coupling, threshold_decay_rate,
    current_decay_rate_1, current_decay_rate_2, current_retention_1,
    current_retention_2, current_jump_1, current_jump_2, dt, n_steps, current
))]
#[allow(clippy::too_many_arguments)]
fn py_mihalas_niebur_simulate<'py>(
    py: Python<'py>,
    v0: f64,
    theta0: f64,
    i1_0: f64,
    i2_0: f64,
    v_rest: f64,
    v_reset: f64,
    theta_reset: f64,
    theta_inf: f64,
    leak_rate: f64,
    threshold_voltage_coupling: f64,
    threshold_decay_rate: f64,
    current_decay_rate_1: f64,
    current_decay_rate_2: f64,
    current_retention_1: f64,
    current_retention_2: f64,
    current_jump_1: f64,
    current_jump_2: f64,
    dt: f64,
    n_steps: usize,
    current: f64,
) -> PyResult<MihalasNieburBatchOutput<'py>> {
    let mut neuron = MihalasNieburNeuron {
        v: v0,
        theta: theta0,
        i1: i1_0,
        i2: i2_0,
        v_rest,
        v_reset,
        theta_reset,
        theta_inf,
        leak_rate,
        threshold_voltage_coupling,
        threshold_decay_rate,
        current_decay_rate_1,
        current_decay_rate_2,
        current_retention_1,
        current_retention_2,
        current_jump_1,
        current_jump_2,
        dt,
    };
    let Some((trace, events)) = neuron.try_simulate(n_steps, current) else {
        return Err(PyFloatingPointError::new_err(
            "Mihalas-Niebur Rust batch rejected an invalid candidate",
        ));
    };
    Ok((
        trace.into_pyarray(py),
        events,
        neuron.v,
        neuron.theta,
        neuron.i1,
        neuron.i2,
    ))
}
