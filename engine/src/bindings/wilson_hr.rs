// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Wilson-HR neuron PyO3 binding

//! Python binding for the Wilson-HR polynomial cortical neuron.

use numpy::{IntoPyArray, PyArray1};
use pyo3::exceptions::PyFloatingPointError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::neurons::WilsonHRNeuron;

type PyWilsonHRBatch<'py> = (Bound<'py, PyArray1<f64>>, i64, f64, f64);

py_neuron_default!("WilsonHRNeuron", PyWilsonHRNeuron, WilsonHRNeuron, state v, state r);

/// Register the Wilson-HR class and simulator with the extension module.
pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyWilsonHRNeuron>()?;
    module.add_function(wrap_pyfunction!(py_wilson_hr_simulate, module)?)?;
    Ok(())
}

/// N-step Wilson (1999) polynomial cortical-neuron simulation.
///
/// Parity contract with
/// `sc_neurocore.neurons.models.wilson_hr.WilsonHRNeuron.simulate`: for the same
/// parameters and constant input the returned continuous `v` trace, sampled
/// upward-crossing count, and final `(v, r)` state are bit-identical to the
/// Python RK4 reference.
#[pyfunction]
#[pyo3(signature = (v0, r0, capacitance, tau_r, v_peak, dt, n_steps, current))]
#[allow(clippy::too_many_arguments)]
fn py_wilson_hr_simulate<'py>(
    py: Python<'py>,
    v0: f64,
    r0: f64,
    capacitance: f64,
    tau_r: f64,
    v_peak: f64,
    dt: f64,
    n_steps: usize,
    current: f64,
) -> PyResult<PyWilsonHRBatch<'py>> {
    let mut neuron = WilsonHRNeuron {
        v: v0,
        r: r0,
        capacitance,
        tau_r,
        v_peak,
        dt,
    };
    let Some((trace, spikes)) = neuron.try_simulate(n_steps, current) else {
        return Err(PyFloatingPointError::new_err(
            "Wilson-HR Rust batch rejected an invalid candidate",
        ));
    };
    Ok((trace.into_pyarray(py), spikes, neuron.v, neuron.r))
}
