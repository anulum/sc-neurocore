// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Retained triangular SCTriangularMcKean-like PyO3 binding

//! Python binding for the SCTriangularMcKean piecewise-linear neuron.

use numpy::{IntoPyArray, PyArray1};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::neurons::SCTriangularMcKeanNeuron;

py_neuron_default!("SCTriangularMcKeanNeuron", PySCTriangularMcKeanNeuron, SCTriangularMcKeanNeuron, state v, state w);

/// Register the SCTriangularMcKean class and simulator with the extension module.
pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PySCTriangularMcKeanNeuron>()?;
    module.add_function(wrap_pyfunction!(py_sc_triangular_mckean_simulate, module)?)?;
    Ok(())
}

/// N-step retained SC triangular piecewise-linear recurrence.
///
/// Parity contract with
/// `sc_neurocore.neurons.models.sc_triangular_mckean.SCTriangularMcKeanNeuron.simulate`: for the same
/// parameters and constant input the returned `v` trace, upward-`v_peak`-crossing
/// spike count, and final `(v, w)` state are bit-identical to the Python RK4
/// reference (the piecewise-linear right-hand side is exact arithmetic —
/// additions, multiplications and branch selection, no transcendental functions —
/// and a two-dimensional autonomous flow cannot be chaotic).
#[pyfunction]
#[pyo3(signature = (v0, w0, a, epsilon, gamma, dt, v_peak, n_steps, current))]
#[allow(clippy::too_many_arguments)]
fn py_sc_triangular_mckean_simulate<'py>(
    py: Python<'py>,
    v0: f64,
    w0: f64,
    a: f64,
    epsilon: f64,
    gamma: f64,
    dt: f64,
    v_peak: f64,
    n_steps: usize,
    current: f64,
) -> (Bound<'py, PyArray1<f64>>, i64, f64, f64) {
    let mut neuron = SCTriangularMcKeanNeuron {
        v: v0,
        w: w0,
        a,
        epsilon,
        gamma,
        dt,
        v_peak,
    };
    let (trace, spikes) = neuron.simulate(n_steps, current);
    (trace.into_pyarray(py), spikes, neuron.v, neuron.w)
}
