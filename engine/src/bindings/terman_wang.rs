// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Terman-Wang oscillator PyO3 binding

//! Python binding for the Terman-Wang LEGION relaxation oscillator.

use numpy::{IntoPyArray, PyArray1};
use pyo3::exceptions::PyFloatingPointError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::neurons::TermanWangOscillator;

type PyTermanWangBatch<'py> = (Bound<'py, PyArray1<f64>>, i64, f64, f64);

py_neuron_default!("TermanWangOscillator", PyTermanWangOscillator, TermanWangOscillator, state v, state w);

/// Register the Terman-Wang class and simulator with the extension module.
pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyTermanWangOscillator>()?;
    module.add_function(wrap_pyfunction!(py_terman_wang_simulate, module)?)?;
    Ok(())
}

/// N-step Terman-Wang (LEGION) relaxation-oscillator simulation.
///
/// Parity contract with
/// `sc_neurocore.neurons.models.terman_wang.TermanWangOscillator.simulate`: for
/// the same parameters and constant input the returned `v` trace, upward-crossing
/// spike count, and final `(v, w)` state match the Python RK4 reference. The cubic
/// uses `v.powi(3)` = `v*v*v` (matching the Python `v*v*v`); the `tanh` gating
/// resolves to the same glibc symbol as Python on Linux, so this backend is
/// bit-identical there. Julia, Go, and Mojo use their own libm `tanh` and are
/// validated against bounded full traces with exact event counts on enrolled
/// operating regimes.
#[pyfunction]
#[pyo3(signature = (v0, w0, alpha, beta, epsilon, rho, dt, v_peak, n_steps, current))]
#[allow(clippy::too_many_arguments)]
fn py_terman_wang_simulate<'py>(
    py: Python<'py>,
    v0: f64,
    w0: f64,
    alpha: f64,
    beta: f64,
    epsilon: f64,
    rho: f64,
    dt: f64,
    v_peak: f64,
    n_steps: usize,
    current: f64,
) -> PyResult<PyTermanWangBatch<'py>> {
    let mut neuron = TermanWangOscillator {
        v: v0,
        w: w0,
        alpha,
        beta,
        epsilon,
        rho,
        dt,
        v_peak,
    };
    let Some((trace, spikes)) = neuron.try_simulate(n_steps, current) else {
        return Err(PyFloatingPointError::new_err(
            "Terman-Wang Rust batch rejected an invalid candidate",
        ));
    };
    Ok((trace.into_pyarray(py), spikes, neuron.v, neuron.w))
}
