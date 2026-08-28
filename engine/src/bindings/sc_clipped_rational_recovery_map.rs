// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Retained rational-recovery map PyO3 binding

//! Checked Python binding for the count-neutral retained project map.

use numpy::{IntoPyArray, PyArray1};
use pyo3::exceptions::PyFloatingPointError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::neurons::SCClippedRationalRecoveryMapNeuron;

py_neuron_default!(
    "SCClippedRationalRecoveryMapNeuron",
    PySCClippedRationalRecoveryMapNeuron,
    SCClippedRationalRecoveryMapNeuron,
    state x,
    state y
);

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PySCClippedRationalRecoveryMapNeuron>()?;
    module.add_function(wrap_pyfunction!(
        py_sc_clipped_rational_recovery_map_simulate,
        module
    )?)?;
    Ok(())
}

#[pyfunction]
#[pyo3(signature = (x0, y0, alpha, beta, j, x_threshold, clip_bound, n_steps, current))]
#[allow(clippy::too_many_arguments)]
fn py_sc_clipped_rational_recovery_map_simulate<'py>(
    py: Python<'py>,
    x0: f64,
    y0: f64,
    alpha: f64,
    beta: f64,
    j: f64,
    x_threshold: f64,
    clip_bound: f64,
    n_steps: usize,
    current: f64,
) -> PyResult<(Bound<'py, PyArray1<f64>>, i64, f64, f64)> {
    let mut neuron = SCClippedRationalRecoveryMapNeuron {
        x: x0,
        y: y0,
        alpha,
        beta,
        j,
        x_threshold,
        clip_bound,
    };
    let (trace, events) = neuron
        .simulate(n_steps, current)
        .map_err(PyFloatingPointError::new_err)?;
    Ok((trace.into_pyarray(py), events, neuron.x, neuron.y))
}
