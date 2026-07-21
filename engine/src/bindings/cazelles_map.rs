// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Cazelles-map PyO3 binding

//! Python binding for the Cazelles-Courbage-Rabinovich bursting map.

use numpy::{IntoPyArray, PyArray1};
use pyo3::prelude::*;

use crate::neurons::CazellesMapNeuron;

/// Register the Cazelles-map simulator with the extension module.
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(py_cazelles_map_simulate, module)?)?;
    Ok(())
}

/// N-step Cazelles-Courbage-Rabinovich (2001) bursting-map simulation.
///
/// Parity contract with `sc_neurocore.neurons.models.cazelles_map.CazellesMapNeuron.simulate`:
/// for the same parameters and constant input the returned `x` trace, spike
/// count, and final `(x, y)` state are bit-identical to the Python reference
/// (the map is exact floating-point arithmetic, no transcendental functions).
#[pyfunction]
#[pyo3(signature = (x0, y0, a, epsilon, sigma, x_threshold, n_steps, current))]
#[allow(clippy::too_many_arguments)]
fn py_cazelles_map_simulate<'py>(
    py: Python<'py>,
    x0: f64,
    y0: f64,
    a: f64,
    epsilon: f64,
    sigma: f64,
    x_threshold: f64,
    n_steps: usize,
    current: f64,
) -> (Bound<'py, PyArray1<f64>>, i64, f64, f64) {
    let mut neuron = CazellesMapNeuron {
        x: x0,
        y: y0,
        a,
        epsilon,
        sigma,
        x_threshold,
    };
    let (trace, spikes) = neuron.simulate(n_steps, current);
    // Return the trace as a NumPy array directly; marshalling a multi-million
    // element Vec<f64> into a Python list would dominate the wall-clock.
    (trace.into_pyarray(py), spikes, neuron.x, neuron.y)
}
