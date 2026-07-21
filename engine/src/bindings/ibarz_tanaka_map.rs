// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Ibarz-Tanaka map PyO3 binding

//! Python binding for the Ibarz-Tanaka spiking map.

use numpy::{IntoPyArray, PyArray1};
use pyo3::exceptions::PyFloatingPointError;
use pyo3::prelude::*;

/// Register the Ibarz-Tanaka map simulator with the extension module.
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(py_ibarz_tanaka_map_simulate, module)?)?;
    Ok(())
}

/// Parity contract with
/// `sc_neurocore.neurons.models.ibarz_tanaka_map.IbarzTanakaMapNeuron.simulate`:
/// for the same parameters and constant input the returned `v` trace, reset-
/// branch event count, and final `(v, u)` state are bit-identical to the Python
/// reference. The implementation follows Eqs. 2-3 of Ibarz et al. (2007).
#[pyfunction]
#[pyo3(signature = (v0, u0, alpha, mu, sigma, n_steps, current))]
fn py_ibarz_tanaka_map_simulate<'py>(
    py: Python<'py>,
    v0: f64,
    u0: f64,
    alpha: f64,
    mu: f64,
    sigma: f64,
    n_steps: usize,
    current: f64,
) -> PyResult<(Bound<'py, PyArray1<f64>>, i64, f64, f64)> {
    let mut neuron = crate::neurons::IbarzTanakaMapNeuron {
        v: v0,
        u: u0,
        alpha,
        mu,
        sigma,
    };
    let (trace, events) = neuron
        .simulate(n_steps, current)
        .map_err(PyFloatingPointError::new_err)?;
    Ok((trace.into_pyarray(py), events, neuron.v, neuron.u))
}
