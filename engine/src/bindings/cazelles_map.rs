// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Cazelles map PyO3 binding

//! Checked Python binding for the source-faithful scalar map.

use numpy::{IntoPyArray, PyArray1};
use pyo3::exceptions::PyFloatingPointError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::neurons::CazellesMapNeuron;

py_neuron_default!("CazellesMapNeuron", PyCazellesMapNeuron, CazellesMapNeuron, state x);

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyCazellesMapNeuron>()?;
    module.add_function(wrap_pyfunction!(py_cazelles_map_simulate, module)?)?;
    Ok(())
}

/// Run the Cazelles et al. (2001) four-branch scalar map.
#[pyfunction]
#[pyo3(signature = (x, alpha, x0, x1, x2, x3, x4, a1, a2, a3, a4, b1, b2, b3, b4, exponent, n_steps, current))]
#[allow(clippy::too_many_arguments)]
fn py_cazelles_map_simulate<'py>(
    py: Python<'py>,
    x: f64,
    alpha: f64,
    x0: f64,
    x1: f64,
    x2: f64,
    x3: f64,
    x4: f64,
    a1: f64,
    a2: f64,
    a3: f64,
    a4: f64,
    b1: f64,
    b2: f64,
    b3: f64,
    b4: f64,
    exponent: u8,
    n_steps: usize,
    current: f64,
) -> PyResult<(Bound<'py, PyArray1<f64>>, i64, f64)> {
    let mut neuron = CazellesMapNeuron {
        x,
        alpha,
        exponent,
        x0,
        x1,
        x2,
        x3,
        x4,
        a1,
        a2,
        a3,
        a4,
        b1,
        b2,
        b3,
        b4,
    };
    let (trace, events) = neuron
        .simulate(n_steps, current)
        .map_err(PyFloatingPointError::new_err)?;
    Ok((trace.into_pyarray(py), events, neuron.x))
}
