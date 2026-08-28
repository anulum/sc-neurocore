// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — retained four-state GLIF PyO3 binding

//! Python class and batch surface for the count-neutral project recurrence.

use numpy::{IntoPyArray, PyArray1};
use pyo3::exceptions::PyFloatingPointError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::neurons::SCFourStateGLIFNeuron;

type SCFourStateGLIFBatchOutput<'py> = (Bound<'py, PyArray1<f64>>, i64, f64, f64, f64, f64);

py_neuron_default!(
    "SCFourStateGLIFNeuron",
    PySCFourStateGLIFNeuron,
    SCFourStateGLIFNeuron,
    state v,
    state theta,
    state i_asc1,
    state i_asc2
);

/// Register the retained class and batch function.
pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PySCFourStateGLIFNeuron>()?;
    module.add_function(wrap_pyfunction!(py_sc_four_state_glif_simulate, module)?)?;
    Ok(())
}

/// Execute the historical four-state candidate-first RK4 recurrence.
#[pyfunction]
#[pyo3(signature = (
    v0, theta0, theta_inf, i_asc1_0, i_asc2_0, v_rest, v_reset, tau_m, tau_theta,
    tau_asc1, tau_asc2, a_theta, delta_theta, r_asc1, r_asc2, resistance, dt,
    n_steps, current
))]
#[allow(clippy::too_many_arguments)]
fn py_sc_four_state_glif_simulate<'py>(
    py: Python<'py>,
    v0: f64,
    theta0: f64,
    theta_inf: f64,
    i_asc1_0: f64,
    i_asc2_0: f64,
    v_rest: f64,
    v_reset: f64,
    tau_m: f64,
    tau_theta: f64,
    tau_asc1: f64,
    tau_asc2: f64,
    a_theta: f64,
    delta_theta: f64,
    r_asc1: f64,
    r_asc2: f64,
    resistance: f64,
    dt: f64,
    n_steps: usize,
    current: f64,
) -> PyResult<SCFourStateGLIFBatchOutput<'py>> {
    let mut neuron = SCFourStateGLIFNeuron {
        v: v0,
        theta: theta0,
        theta_inf,
        i_asc1: i_asc1_0,
        i_asc2: i_asc2_0,
        v_rest,
        v_reset,
        tau_m,
        tau_theta,
        tau_asc1,
        tau_asc2,
        a_theta,
        delta_theta,
        r_asc1,
        r_asc2,
        resistance,
        dt,
    };
    let Some((trace, events)) = neuron.try_simulate(n_steps, current) else {
        return Err(PyFloatingPointError::new_err(
            "retained four-state GLIF Rust batch rejected an invalid candidate",
        ));
    };
    Ok((
        trace.into_pyarray(py),
        events,
        neuron.v,
        neuron.theta,
        neuron.i_asc1,
        neuron.i_asc2,
    ))
}
