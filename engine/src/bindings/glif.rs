// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — GLIF PyO3 binding

//! Python binding for the Allen GLIF5 generalised integrate-and-fire model.

use numpy::{IntoPyArray, PyArray1};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::neurons;

py_neuron_default!("GLIFNeuron", PyGLIFNeuron, neurons::GLIFNeuron, state v, state theta, state i_asc1, state i_asc2);

/// Register the GLIF class and batch simulator.
pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyGLIFNeuron>()?;
    module.add_function(wrap_pyfunction!(py_glif_simulate, module)?)?;
    Ok(())
}

/// Parity contract with
/// `sc_neurocore.neurons.models.glif.GLIFNeuron.simulate`: for the same
/// parameters and constant input the returned `v` trace, total spike count, and
/// final `(v, theta, i_asc1, i_asc2)` state are bit-identical to the Python RK4
/// reference. The Allen GLIF5 model has a purely linear right-hand side (no
/// transcendental functions), so every RK4 stage is exact arithmetic and the
/// trace reproduces to the last bit across Rust/Julia/Go (Mojo FMA-fuses,
/// validated non-amplifying).
#[pyfunction]
#[pyo3(signature = (
    v0, theta0, theta_inf, i_asc1_0, i_asc2_0, v_rest, v_reset, tau_m, tau_theta,
    tau_asc1, tau_asc2, a_theta, delta_theta, r_asc1, r_asc2, resistance, dt,
    n_steps, current
))]
#[allow(clippy::too_many_arguments)]
fn py_glif_simulate<'py>(
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
) -> (Bound<'py, PyArray1<f64>>, i64, f64, f64, f64, f64) {
    let mut neuron = crate::neurons::GLIFNeuron {
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
    let (trace, spikes) = neuron.simulate(n_steps, current);
    (
        trace.into_pyarray(py),
        spikes,
        neuron.v,
        neuron.theta,
        neuron.i_asc1,
        neuron.i_asc2,
    )
}
