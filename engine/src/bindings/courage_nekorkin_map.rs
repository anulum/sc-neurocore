// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Courbage-Nekorkin-Vdovin map PyO3 binding

//! Python binding for the Courbage-Nekorkin-Vdovin spiking map.

use numpy::{IntoPyArray, PyArray1};
use pyo3::prelude::*;

use crate::neurons::CourageNekorkinMapNeuron;

/// Register the Courbage-Nekorkin-Vdovin map simulator with the extension module.
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(py_courage_nekorkin_map_simulate, module)?)?;
    Ok(())
}

/// N-step Courbage-Nekorkin-Vdovin (2007) discontinuous spiking-map simulation.
///
/// Parity contract with
/// `sc_neurocore.neurons.models.courage_nekorkin_map.CourageNekorkinMapNeuron.simulate`:
/// for the same parameters and constant input the returned `x` trace,
/// upward-crossing spike count, and final `(x, y)` state are bit-identical to
/// the Python reference (the map is exact floating-point arithmetic — additions,
/// multiplications, one division for the breakpoints, and a piecewise/Heaviside
/// branch, no transcendental functions).
#[pyfunction]
#[pyo3(signature = (x0, y0, m0, m1, a, d, j, beta, eps, x_threshold, n_steps, current))]
#[allow(clippy::too_many_arguments)]
fn py_courage_nekorkin_map_simulate<'py>(
    py: Python<'py>,
    x0: f64,
    y0: f64,
    m0: f64,
    m1: f64,
    a: f64,
    d: f64,
    j: f64,
    beta: f64,
    eps: f64,
    x_threshold: f64,
    n_steps: usize,
    current: f64,
) -> (Bound<'py, PyArray1<f64>>, i64, f64, f64) {
    let mut neuron = CourageNekorkinMapNeuron {
        x: x0,
        y: y0,
        m0,
        m1,
        a,
        d,
        j,
        beta,
        eps,
        x_threshold,
    };
    let (trace, spikes) = neuron.simulate(n_steps, current);
    (trace.into_pyarray(py), spikes, neuron.x, neuron.y)
}
