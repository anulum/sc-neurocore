// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Izhikevich 2007 PyO3 binding

//! Python binding for the NeuroML Izhikevich 2007 model.

use numpy::{IntoPyArray, PyArray1};
use pyo3::exceptions::PyFloatingPointError;
use pyo3::prelude::*;

/// Register the Izhikevich 2007 batch simulator with the extension module.
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(py_izhikevich2007_simulate, module)?)?;
    Ok(())
}

/// Parity contract with
/// `sc_neurocore.neurons.models.izhikevich2007.Izhikevich2007Neuron.simulate`:
/// for the same parameters and constant input the returned `v` trace, spike
/// count, and final `(v, u)` state are bit-identical to the Python RK4 reference
/// (the NeuroML right-hand side `k (v-vr)(v-vt)/C` is exact arithmetic — products,
/// a sum and a division, no transcendental functions).
#[pyfunction]
#[pyo3(signature = (v0, u0, cap, k, vr, vt, vpeak, a, b, c, d, dt, n_steps, current))]
#[allow(clippy::too_many_arguments)]
fn py_izhikevich2007_simulate<'py>(
    py: Python<'py>,
    v0: f64,
    u0: f64,
    cap: f64,
    k: f64,
    vr: f64,
    vt: f64,
    vpeak: f64,
    a: f64,
    b: f64,
    c: f64,
    d: f64,
    dt: f64,
    n_steps: usize,
    current: f64,
) -> PyResult<(Bound<'py, PyArray1<f64>>, i64, f64, f64)> {
    let mut neuron = crate::rk4_neurons::Izhikevich2007Rk4 {
        v: v0,
        u: u0,
        cap,
        k,
        vr,
        vt,
        vpeak,
        a,
        b,
        c,
        d,
        dt,
    };
    let (trace, spikes) = neuron
        .simulate(n_steps, current)
        .map_err(PyFloatingPointError::new_err)?;
    Ok((trace.into_pyarray(py), spikes, neuron.v, neuron.u))
}
