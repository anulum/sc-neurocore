// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — FitzHugh-Nagumo PyO3 binding

//! Python binding for the FitzHugh-Nagumo two-state excitable-system model.

use numpy::{IntoPyArray, PyArray1};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::neurons::FitzHughNagumoNeuron;

py_neuron_default!("FitzHughNagumoNeuron", PyFitzHughNagumoNeuron, FitzHughNagumoNeuron, state v, state w);

/// Register the FitzHugh-Nagumo class and batch simulator.
pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyFitzHughNagumoNeuron>()?;
    module.add_function(wrap_pyfunction!(py_fitzhugh_nagumo_simulate, module)?)?;
    Ok(())
}

/// Parity contract with
/// `sc_neurocore.neurons.models.fitzhugh_nagumo.FitzHughNagumoNeuron.simulate`:
/// for the same parameters and constant input the returned `v` trace, upward-
/// crossing spike count, and final `(v, w)` state are bit-identical to the
/// Python RK4 reference (the right-hand side is exact arithmetic — a cube
/// `v.powi(3)` = `v*v*v`, additions and multiplications, no transcendental
/// functions — and a two-dimensional flow cannot be chaotic).
#[pyfunction]
#[pyo3(signature = (v0, w0, a, b, epsilon, dt, v_threshold, n_steps, current))]
#[allow(clippy::too_many_arguments)]
fn py_fitzhugh_nagumo_simulate<'py>(
    py: Python<'py>,
    v0: f64,
    w0: f64,
    a: f64,
    b: f64,
    epsilon: f64,
    dt: f64,
    v_threshold: f64,
    n_steps: usize,
    current: f64,
) -> (Bound<'py, PyArray1<f64>>, i64, f64, f64) {
    let mut neuron = FitzHughNagumoNeuron {
        v: v0,
        w: w0,
        a,
        b,
        epsilon,
        dt,
        v_threshold,
    };
    let (trace, spikes) = neuron.simulate(n_steps, current);
    (trace.into_pyarray(py), spikes, neuron.v, neuron.w)
}
