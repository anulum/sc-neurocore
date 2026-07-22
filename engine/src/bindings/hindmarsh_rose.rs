// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Hindmarsh-Rose PyO3 binding

//! Python binding for the Hindmarsh-Rose three-state bursting model.

use numpy::{IntoPyArray, PyArray1};
use pyo3::prelude::*;

/// Register the Hindmarsh-Rose batch simulator with the extension module.
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(py_hindmarsh_rose_simulate, module)?)?;
    Ok(())
}

/// Parity contract with
/// `sc_neurocore.neurons.models.hindmarsh_rose.HindmarshRoseNeuron.simulate`:
/// for the same parameters and constant input the returned `x` trace, upward-
/// crossing spike count, and final `(x, y, z)` state are bit-identical to the
/// Python RK4 reference (the right-hand side is exact arithmetic — `x.powi(3)`
/// = `x*x*x`, `x.powi(2)` = `x*x`, no transcendental functions — so even the
/// chaotic bursting trace reproduces exactly).
#[pyfunction]
#[pyo3(signature = (x0, y0, z0, b, r, s, x_rest, dt, x_threshold, n_steps, current))]
#[allow(clippy::too_many_arguments)]
fn py_hindmarsh_rose_simulate<'py>(
    py: Python<'py>,
    x0: f64,
    y0: f64,
    z0: f64,
    b: f64,
    r: f64,
    s: f64,
    x_rest: f64,
    dt: f64,
    x_threshold: f64,
    n_steps: usize,
    current: f64,
) -> (Bound<'py, PyArray1<f64>>, i64, f64, f64, f64) {
    let mut neuron = crate::neurons::HindmarshRoseNeuron {
        x: x0,
        y: y0,
        z: z0,
        b,
        r,
        s,
        x_rest,
        dt,
        x_threshold,
    };
    let (trace, spikes) = neuron.simulate(n_steps, current);
    (trace.into_pyarray(py), spikes, neuron.x, neuron.y, neuron.z)
}
