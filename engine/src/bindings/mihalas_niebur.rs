// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Mihalas-Niebur PyO3 binding

//! Python binding for the Mihalas-Niebur 2009 generalised integrate-and-fire model.

use numpy::{IntoPyArray, PyArray1};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::neurons;

py_neuron_default!("MihalasNieburNeuron", PyMihalasNieburNeuron, neurons::MihalasNieburNeuron, state v, state theta, state i1, state i2);

/// Register the Mihalas-Niebur class and batch simulator.
pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyMihalasNieburNeuron>()?;
    module.add_function(wrap_pyfunction!(py_mihalas_niebur_simulate, module)?)?;
    Ok(())
}

/// Parity contract with
/// `sc_neurocore.neurons.models.mihalas_niebur.MihalasNieburNeuron.simulate`:
/// for the same parameters and constant input the returned `v` trace, total
/// spike count, and final `(v, theta, i1, i2)` state are bit-identical to the
/// Python RK4 reference. The Mihalas-Niebur 2009 generalised integrate-and-fire
/// model has a purely linear right-hand side (no transcendental functions), so
/// every RK4 stage is exact arithmetic and the trace reproduces to the last bit
/// across Rust/Julia/Go (Mojo FMA-fuses, validated non-amplifying).
#[pyfunction]
#[pyo3(signature = (
    v0, theta0, i1_0, i2_0, v_rest, v_reset, theta_reset, theta_inf,
    tau_v, tau_theta, tau_1, tau_2, a, b, r1, r2, dt, n_steps, current
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
    tau_v: f64,
    tau_theta: f64,
    tau_1: f64,
    tau_2: f64,
    a: f64,
    b: f64,
    r1: f64,
    r2: f64,
    dt: f64,
    n_steps: usize,
    current: f64,
) -> (Bound<'py, PyArray1<f64>>, i64, f64, f64, f64, f64) {
    let mut neuron = crate::neurons::MihalasNieburNeuron {
        v: v0,
        theta: theta0,
        i1: i1_0,
        i2: i2_0,
        v_rest,
        v_reset,
        theta_reset,
        theta_inf,
        tau_v,
        tau_theta,
        tau_1,
        tau_2,
        a,
        b,
        r1,
        r2,
        dt,
    };
    let (trace, spikes) = neuron.simulate(n_steps, current);
    (
        trace.into_pyarray(py),
        spikes,
        neuron.v,
        neuron.theta,
        neuron.i1,
        neuron.i2,
    )
}
