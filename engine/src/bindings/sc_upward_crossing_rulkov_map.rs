// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — retained upward-crossing Rulkov PyO3 binding

//! Python binding for the retained upward-crossing Rulkov map.

use numpy::{IntoPyArray, PyArray1};
use pyo3::exceptions::PyFloatingPointError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::neurons::SCUpwardCrossingRulkovMapNeuron;

py_neuron_default!(
    "SCUpwardCrossingRulkovMapNeuron",
    PySCUpwardCrossingRulkovMapNeuron,
    SCUpwardCrossingRulkovMapNeuron,
    state x,
    state y
);

/// Register the retained class and batch simulator.
pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PySCUpwardCrossingRulkovMapNeuron>()?;
    module.add_function(wrap_pyfunction!(
        py_sc_upward_crossing_rulkov_map_simulate,
        module
    )?)?;
    Ok(())
}

/// Execute the retained configurable upward-crossing recurrence.
#[pyfunction]
#[pyo3(signature = (x0, y0, alpha, sigma, mu, x_threshold, n_steps, current))]
#[allow(clippy::too_many_arguments)]
fn py_sc_upward_crossing_rulkov_map_simulate<'py>(
    py: Python<'py>,
    x0: f64,
    y0: f64,
    alpha: f64,
    sigma: f64,
    mu: f64,
    x_threshold: f64,
    n_steps: usize,
    current: f64,
) -> PyResult<(Bound<'py, PyArray1<f64>>, i64, f64, f64)> {
    let mut neuron = SCUpwardCrossingRulkovMapNeuron {
        x: x0,
        y: y0,
        alpha,
        sigma,
        mu,
        x_threshold,
    };
    let Some((trace, events)) = neuron.try_simulate(n_steps, current) else {
        return Err(PyFloatingPointError::new_err(
            "retained Rulkov Rust batch rejected an invalid candidate",
        ));
    };
    Ok((trace.into_pyarray(py), events, neuron.x, neuron.y))
}
