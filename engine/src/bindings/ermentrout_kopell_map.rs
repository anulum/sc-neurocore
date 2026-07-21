// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Ermentrout-Kopell map PyO3 binding

//! Python binding for the Ermentrout-Kopell Type-I theta map.

use numpy::{IntoPyArray, PyArray1};
use pyo3::prelude::*;

/// Register the Ermentrout-Kopell map simulator with the extension module.
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(py_ermentrout_kopell_map_simulate, module)?)?;
    Ok(())
}

/// Parity contract with
/// `sc_neurocore.neurons.models.ermentrout_kopell_map_neuron.ErmentroutKopellMapNeuron.simulate`:
/// for the same parameters and constant input the returned `theta` trace,
/// upward-crossing spike count, and final `theta` state match the Python
/// reference bit-for-bit on a shared libm (the only transcendental is `cos`,
/// and the non-chaotic phase flow does not amplify ULP differences). This is a
/// one-dimensional phase map, so there is no second state.
#[pyfunction]
#[pyo3(signature = (theta0, dt, gain, theta_threshold, n_steps, current))]
fn py_ermentrout_kopell_map_simulate<'py>(
    py: Python<'py>,
    theta0: f64,
    dt: f64,
    gain: f64,
    theta_threshold: f64,
    n_steps: usize,
    current: f64,
) -> (Bound<'py, PyArray1<f64>>, i64, f64) {
    let mut neuron = crate::neurons::ErmentroutKopellMapNeuron {
        theta: theta0,
        dt,
        gain,
        theta_threshold,
    };
    let (trace, spikes) = neuron.simulate(n_steps, current);
    (trace.into_pyarray(py), spikes, neuron.theta)
}
