// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Pernarowski neuron PyO3 binding

//! Python binding for the Pernarowski autonomous beta-cell burster.

use numpy::{IntoPyArray, PyArray1};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::neurons::PernarowskiNeuron;

py_neuron_default!("PernarowskiNeuron", PyPernarowskiNeuron, PernarowskiNeuron, state v, state w, state z);

/// Register the Pernarowski class and simulator with the extension module.
pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyPernarowskiNeuron>()?;
    module.add_function(wrap_pyfunction!(py_pernarowski_simulate, module)?)?;
    Ok(())
}

/// N-step Pernarowski (1994) pancreatic beta-cell burster simulation.
///
/// Parity contract with
/// `sc_neurocore.neurons.models.pernarowski.PernarowskiNeuron.simulate`: for the
/// same parameters and constant input the returned `v` trace, upward-crossing
/// spike count, and final `(v, w, z)` state are bit-identical to the Python RK4
/// reference (the cubic uses `v.powi(3)` = `v*v*v`, matching the Python `v*v*v`;
/// no transcendental functions).
#[pyfunction]
#[pyo3(signature = (v0, w0, z0, alpha, beta, eps1, eps2, gamma, dt, v_threshold, n_steps, current))]
#[allow(clippy::too_many_arguments)]
fn py_pernarowski_simulate<'py>(
    py: Python<'py>,
    v0: f64,
    w0: f64,
    z0: f64,
    alpha: f64,
    beta: f64,
    eps1: f64,
    eps2: f64,
    gamma: f64,
    dt: f64,
    v_threshold: f64,
    n_steps: usize,
    current: f64,
) -> (Bound<'py, PyArray1<f64>>, i64, f64, f64, f64) {
    let mut neuron = PernarowskiNeuron {
        v: v0,
        w: w0,
        z: z0,
        alpha,
        beta,
        eps1,
        eps2,
        gamma,
        dt,
        v_threshold,
    };
    let (trace, spikes) = neuron.simulate(n_steps, current);
    (trace.into_pyarray(py), spikes, neuron.v, neuron.w, neuron.z)
}
