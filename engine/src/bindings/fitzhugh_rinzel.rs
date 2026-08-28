// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — FitzHugh-Rinzel PyO3 binding

//! Python binding for the FitzHugh-Rinzel three-timescale bursting model.

use numpy::{IntoPyArray, PyArray1};
use pyo3::exceptions::PyFloatingPointError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::neurons::FitzHughRinzelNeuron;

type PyFitzHughRinzelBatch<'py> = (Bound<'py, PyArray1<f64>>, i64, f64, f64, f64);

py_neuron_default!("FitzHughRinzelNeuron", PyFitzHughRinzelNeuron, FitzHughRinzelNeuron, state v, state w, state y);

/// Register the FitzHugh-Rinzel class and batch simulator.
pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyFitzHughRinzelNeuron>()?;
    module.add_function(wrap_pyfunction!(py_fitzhugh_rinzel_simulate, module)?)?;
    Ok(())
}

/// Parity contract with
/// `sc_neurocore.neurons.models.fitzhugh_rinzel.FitzHughRinzelNeuron.simulate`:
/// for the same parameters and constant input the returned `v` trace, upward-
/// crossing spike count, and final `(v, w, y)` state are bit-identical to the
/// Python RK4 reference (the right-hand side is exact arithmetic — `v.powi(3)`
/// = `v*v*v`, additions and multiplications, no transcendental functions).
#[pyfunction]
#[pyo3(signature = (v0, w0, y0, a, b, c, d, delta, mu, dt, v_threshold, n_steps, current))]
#[allow(clippy::too_many_arguments)]
fn py_fitzhugh_rinzel_simulate<'py>(
    py: Python<'py>,
    v0: f64,
    w0: f64,
    y0: f64,
    a: f64,
    b: f64,
    c: f64,
    d: f64,
    delta: f64,
    mu: f64,
    dt: f64,
    v_threshold: f64,
    n_steps: usize,
    current: f64,
) -> PyResult<PyFitzHughRinzelBatch<'py>> {
    let mut neuron = FitzHughRinzelNeuron {
        v: v0,
        w: w0,
        y: y0,
        a,
        b,
        c,
        d,
        delta,
        mu,
        dt,
        v_threshold,
    };
    let Some((trace, spikes)) = neuron.try_simulate(n_steps, current) else {
        return Err(PyFloatingPointError::new_err(
            "FitzHugh-Rinzel Rust batch rejected an invalid candidate",
        ));
    };
    Ok((trace.into_pyarray(py), spikes, neuron.v, neuron.w, neuron.y))
}
