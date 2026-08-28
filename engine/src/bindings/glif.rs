// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Teeter GLIF5 PyO3 binding

//! Python class and failure-atomic batch surface for canonical GLIF5.

use numpy::{IntoPyArray, PyArray1};
use pyo3::exceptions::PyFloatingPointError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::neurons::GLIFNeuron;

type GLIFBatchOutput<'py> = (Bound<'py, PyArray1<f64>>, i64, f64, f64, f64, f64, f64, f64);

py_neuron_default!(
    "GLIFNeuron",
    PyGLIFNeuron,
    GLIFNeuron,
    state v,
    state theta_spike,
    state i_asc1,
    state i_asc2,
    state theta_voltage,
    state refractory_remaining
);

/// Register the canonical class and batch function.
pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyGLIFNeuron>()?;
    module.add_function(wrap_pyfunction!(py_glif_simulate, module)?)?;
    Ok(())
}

/// Execute the five-state exact-flow specialization under constant current.
#[pyfunction]
#[pyo3(signature = (
    v0, theta_spike0, i_asc1_0, i_asc2_0, theta_voltage0, refractory_remaining0,
    e_l, capacitance, resistance, theta_inf, b_spike, b_voltage, a_voltage,
    k_asc1, k_asc2, f_v, delta_v, delta_theta_spike, f_asc1, f_asc2,
    delta_i_asc1, delta_i_asc2, refractory_period, dt, n_steps, current
))]
#[allow(clippy::too_many_arguments)]
fn py_glif_simulate<'py>(
    py: Python<'py>,
    v0: f64,
    theta_spike0: f64,
    i_asc1_0: f64,
    i_asc2_0: f64,
    theta_voltage0: f64,
    refractory_remaining0: f64,
    e_l: f64,
    capacitance: f64,
    resistance: f64,
    theta_inf: f64,
    b_spike: f64,
    b_voltage: f64,
    a_voltage: f64,
    k_asc1: f64,
    k_asc2: f64,
    f_v: f64,
    delta_v: f64,
    delta_theta_spike: f64,
    f_asc1: f64,
    f_asc2: f64,
    delta_i_asc1: f64,
    delta_i_asc2: f64,
    refractory_period: f64,
    dt: f64,
    n_steps: usize,
    current: f64,
) -> PyResult<GLIFBatchOutput<'py>> {
    let mut neuron = GLIFNeuron {
        v: v0,
        theta_spike: theta_spike0,
        i_asc1: i_asc1_0,
        i_asc2: i_asc2_0,
        theta_voltage: theta_voltage0,
        refractory_remaining: refractory_remaining0,
        e_l,
        capacitance,
        resistance,
        theta_inf,
        b_spike,
        b_voltage,
        a_voltage,
        k_asc1,
        k_asc2,
        f_v,
        delta_v,
        delta_theta_spike,
        f_asc1,
        f_asc2,
        delta_i_asc1,
        delta_i_asc2,
        refractory_period,
        dt,
    };
    let Some((trace, events)) = neuron.try_simulate(n_steps, current) else {
        return Err(PyFloatingPointError::new_err(
            "GLIF5 Rust batch rejected an invalid candidate",
        ));
    };
    Ok((
        trace.into_pyarray(py),
        events,
        neuron.v,
        neuron.theta_spike,
        neuron.i_asc1,
        neuron.i_asc2,
        neuron.theta_voltage,
        neuron.refractory_remaining,
    ))
}
