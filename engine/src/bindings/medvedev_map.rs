// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Medvedev map PyO3 binding

//! Python binding for the Medvedev slow-calcium first-return map.

use numpy::{IntoPyArray, PyArray1};
use pyo3::exceptions::PyFloatingPointError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::neurons::MedvedevMapNeuron;

py_neuron_default!("MedvedevMapNeuron", PyMedvedevMapNeuron, MedvedevMapNeuron, state u);

/// Register the Medvedev map class and simulator with the extension module.
pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyMedvedevMapNeuron>()?;
    module.add_function(wrap_pyfunction!(py_medvedev_map_simulate, module)?)?;
    Ok(())
}

/// N-step Medvedev (2005) slow-calcium first-return simulation.
///
/// The recurrence matches the disclosed Section-4 calibration in
/// `sc_neurocore.neurons.models.medvedev_map.MedvedevMapNeuron`. The returned
/// trace records `u` after each map iteration and the event count identifies
/// pre-step states in the active fast-return region `u <= u_HC`. Non-finite or
/// topologically invalid inputs fail before a corrupt candidate is committed.
#[pyfunction]
#[pyo3(signature = (u0, beta_0, beta_hc, beta_sn, delta, decay_t0, alpha_t0, f_0, f_1, homoclinic_exponent, d, input_gain, n_steps, current))]
#[allow(clippy::too_many_arguments)]
fn py_medvedev_map_simulate<'py>(
    py: Python<'py>,
    u0: f64,
    beta_0: f64,
    beta_hc: f64,
    beta_sn: f64,
    delta: f64,
    decay_t0: f64,
    alpha_t0: f64,
    f_0: f64,
    f_1: f64,
    homoclinic_exponent: f64,
    d: f64,
    input_gain: f64,
    n_steps: usize,
    current: f64,
) -> PyResult<(Bound<'py, PyArray1<f64>>, i64, f64)> {
    let mut neuron = MedvedevMapNeuron {
        u: u0,
        beta_0,
        beta_hc,
        beta_sn,
        delta,
        decay_t0,
        alpha_t0,
        f_0,
        f_1,
        homoclinic_exponent,
        d,
        input_gain,
    };
    let (trace, events) = neuron
        .simulate(n_steps, current)
        .map_err(PyFloatingPointError::new_err)?;
    Ok((trace.into_pyarray(py), events, neuron.u))
}
