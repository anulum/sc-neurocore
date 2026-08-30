// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — DPI neuron PyO3 binding

use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::neurons;

type DpiCompletePythonPacket = (Vec<f64>, Vec<f64>, Vec<f64>, Vec<u8>, f64, f64, f64);

py_neuron_default!(
    "DPINeuron",
    PyDPINeuron,
    neurons::DPINeuron,
    state i_mem,
    state i_ahp,
    state refractory_time
);

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyDPINeuron>()?;
    module.add_function(wrap_pyfunction!(dpi_neuron_simulate_complete, module)?)?;
    Ok(())
}

/// Execute the complete configurable DPI contract in one checked Rust batch.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn dpi_neuron_simulate_complete(
    i_mem: f64,
    i_ahp: f64,
    refractory_time: f64,
    i_threshold: f64,
    i_reset: f64,
    i_rest: f64,
    i_tau: f64,
    i_g: f64,
    i_tau_ahp: f64,
    i_ga: f64,
    i_spike: f64,
    i_0: f64,
    kappa: f64,
    alpha: f64,
    tau: f64,
    tau_ahp: f64,
    refractory_period: f64,
    dt: f64,
    n_steps: usize,
    current: f64,
) -> PyResult<DpiCompletePythonPacket> {
    let mut model = neurons::DPINeuron {
        i_mem,
        i_ahp,
        refractory_time,
        i_threshold,
        i_reset,
        i_rest,
        i_tau,
        i_g,
        i_tau_ahp,
        i_ga,
        i_spike,
        i_0,
        kappa,
        alpha,
        tau,
        tau_ahp,
        refractory_period,
        dt,
    };
    let (i_mem_trace, i_ahp_trace, refractory_trace, events) = model
        .simulate_complete(n_steps, current)
        .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_owned()))?;
    Ok((
        i_mem_trace,
        i_ahp_trace,
        refractory_trace,
        events,
        model.i_mem,
        model.i_ahp,
        model.refractory_time,
    ))
}
