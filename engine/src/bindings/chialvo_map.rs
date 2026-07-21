// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Chialvo-map PyO3 binding

//! Python binding for the checked Chialvo two-dimensional map.

use numpy::{IntoPyArray, PyArray1};
use pyo3::exceptions::PyFloatingPointError;
use pyo3::prelude::*;

use crate::neurons::ChialvoMapNeuron;

/// Register the Chialvo-map simulator with the extension module.
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(py_chialvo_map_simulate, module)?)?;
    Ok(())
}

/// N-step Chialvo (1995) two-dimensional-map simulation.
///
/// The recurrence matches
/// `sc_neurocore.neurons.models.chialvo_map.ChialvoMapNeuron.simulate`. The
/// returned trace records `x` after every simultaneous map update; the event
/// count uses the maintained upward `x_threshold` crossing convention. The
/// checked path rejects non-finite state, parameters, input, or candidates
/// without committing a corrupt state.
#[pyfunction]
#[pyo3(signature = (x0, y0, a, b, c, k, x_threshold, n_steps, current))]
#[allow(clippy::too_many_arguments)]
fn py_chialvo_map_simulate<'py>(
    py: Python<'py>,
    x0: f64,
    y0: f64,
    a: f64,
    b: f64,
    c: f64,
    k: f64,
    x_threshold: f64,
    n_steps: usize,
    current: f64,
) -> PyResult<(Bound<'py, PyArray1<f64>>, i64, f64, f64)> {
    let mut neuron = ChialvoMapNeuron {
        x: x0,
        y: y0,
        a,
        b,
        c,
        k,
        x_threshold,
    };
    let (trace, spikes) = neuron
        .simulate(n_steps, current)
        .map_err(PyFloatingPointError::new_err)?;
    Ok((trace.into_pyarray(py), spikes, neuron.x, neuron.y))
}
