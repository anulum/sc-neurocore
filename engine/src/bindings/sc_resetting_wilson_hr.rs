// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — retained resetting Wilson-HR PyO3 binding

//! Python binding for the retained resetting Wilson-HR project recurrence.

use numpy::{IntoPyArray, PyArray1};
use pyo3::exceptions::PyFloatingPointError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::neurons::SCResettingWilsonHRNeuron;

type PySCResettingWilsonHRBatch<'py> = (Bound<'py, PyArray1<f64>>, i64, f64, f64);

py_neuron_default!(
    "SCResettingWilsonHRNeuron",
    PySCResettingWilsonHRNeuron,
    SCResettingWilsonHRNeuron,
    state v,
    state r
);

/// Register the retained project class and native batch simulator.
pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PySCResettingWilsonHRNeuron>()?;
    module.add_function(wrap_pyfunction!(
        py_sc_resetting_wilson_hr_simulate,
        module
    )?)?;
    Ok(())
}

/// Run the historical unit-capacitance RK4 recurrence with hard reset.
#[pyfunction]
#[pyo3(signature = (v0, r0, tau_r, v_peak, dt, n_steps, current))]
#[allow(clippy::too_many_arguments)]
fn py_sc_resetting_wilson_hr_simulate<'py>(
    py: Python<'py>,
    v0: f64,
    r0: f64,
    tau_r: f64,
    v_peak: f64,
    dt: f64,
    n_steps: usize,
    current: f64,
) -> PyResult<PySCResettingWilsonHRBatch<'py>> {
    let mut neuron = SCResettingWilsonHRNeuron {
        v: v0,
        r: r0,
        tau_r,
        v_peak,
        dt,
    };
    let Some((trace, events)) = neuron.try_simulate(n_steps, current) else {
        return Err(PyFloatingPointError::new_err(
            "SC resetting Wilson-HR Rust batch rejected an invalid candidate",
        ));
    };
    Ok((trace.into_pyarray(py), events, neuron.v, neuron.r))
}
