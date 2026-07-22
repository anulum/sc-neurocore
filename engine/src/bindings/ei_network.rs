// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Excitatory/inhibitory network PyO3 binding

//! Python binding for the seeded excitatory/inhibitory network simulator.

use numpy::IntoPyArray;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::ei_network;

/// Register the excitatory/inhibitory network simulator with the extension module.
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(py_simulate_ei_network, module)?)?;
    Ok(())
}

#[pyfunction]
#[pyo3(signature = (
    n_exc=80, n_inh=20,
    w_ee=0.1, w_ei=0.4, w_ie=0.1, w_ii=0.4,
    p_conn=0.2, ext_rate=5.0,
    duration=200.0, dt=0.1, seed=42
))]
fn py_simulate_ei_network<'py>(
    py: Python<'py>,
    n_exc: usize,
    n_inh: usize,
    w_ee: f64,
    w_ei: f64,
    w_ie: f64,
    w_ii: f64,
    p_conn: f64,
    ext_rate: f64,
    duration: f64,
    dt: f64,
    seed: u64,
) -> PyResult<Py<PyAny>> {
    let r = ei_network::simulate_ei(
        n_exc, n_inh, w_ee, w_ei, w_ie, w_ii, p_conn, ext_rate, duration, dt, seed,
    );
    let n_spikes = r.spike_times.len();
    let d = PyDict::new(py);
    d.set_item("spike_times", r.spike_times.into_pyarray(py))?;
    d.set_item(
        "spike_neurons",
        r.spike_neurons
            .iter()
            .map(|&x| x as i64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    d.set_item("n_exc", r.n_exc)?;
    d.set_item("n_inh", r.n_inh)?;
    d.set_item("n_total", r.n_exc + r.n_inh)?;
    d.set_item("n_spikes", n_spikes)?;
    d.set_item("rate_time", r.rate_time.into_pyarray(py))?;
    d.set_item("exc_rates", r.exc_rates.into_pyarray(py))?;
    d.set_item("inh_rates", r.inh_rates.into_pyarray(py))?;
    d.set_item("duration", duration)?;
    d.set_item("dt", dt)?;
    d.set_item("mean_exc_rate", r.mean_exc_rate)?;
    d.set_item("mean_inh_rate", r.mean_inh_rate)?;
    Ok(d.into_any().unbind())
}
