// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — RK4 neuron integrator PyO3 binding

//! Python binding for the RK4 neuron integrator ports.

use numpy::{IntoPyArray, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use super::{AdExRk4, HodgkinHuxleyRk4, IzhikevichRk4};

/// Register the RK4 neuron simulator with the extension module.
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(py_rk4_neuron_simulate, module)?)?;
    Ok(())
}

/// Simulate an RK4 neuron over a current trace and return its traces as NumPy arrays.
///
/// `model_name` is matched on its ASCII letters and digits, lowercased: `izhikevich`,
/// `scizhikevichneuron` or `izhikevichneuron` (default `dt` 1.0), `hodgkinhuxley` or
/// `hodgkinhuxleyneuron` (default 0.01; each sample then covers about 1 ms), and `adex` or
/// `adexneuron` (default 0.1). One step runs per current sample. The dict holds `v`; `u`,
/// `w` or `m`, `h` and `n`; `spikes`, the indices of the spiking steps; and `n_steps`.
///
/// # Errors
///
/// `ValueError` for an unknown model, a `dt` that is not positive and finite, an empty trace
/// or a non-finite current.
#[pyfunction]
#[pyo3(signature = (model_name, current_trace, dt=None))]
pub fn py_rk4_neuron_simulate<'py>(
    py: Python<'py>,
    model_name: &str,
    current_trace: PyReadonlyArray1<'py, f64>,
    dt: Option<f64>,
) -> PyResult<Py<PyAny>> {
    let currents = current_trace.as_slice()?;
    match normalise_model_name(model_name).as_str() {
        "izhikevich" | "scizhikevichneuron" | "izhikevichneuron" => {
            let dt = validate_trace_dt(currents, dt.unwrap_or(1.0))?;
            simulate_izhikevich(py, currents, dt)
        }
        "hodgkinhuxley" | "hodgkinhuxleyneuron" => {
            let dt = validate_trace_dt(currents, dt.unwrap_or(0.01))?;
            simulate_hodgkin_huxley(py, currents, dt)
        }
        "adex" | "adexneuron" => {
            let dt = validate_trace_dt(currents, dt.unwrap_or(0.1))?;
            simulate_adex(py, currents, dt)
        }
        _ => Err(PyValueError::new_err(format!(
            "unsupported RK4 neuron model {model_name:?}"
        ))),
    }
}

fn validate_trace_dt(currents: &[f64], dt: f64) -> PyResult<f64> {
    if !dt.is_finite() || dt <= 0.0 {
        return Err(PyValueError::new_err("dt must be a positive finite scalar"));
    }
    if currents.is_empty() {
        return Err(PyValueError::new_err("current_trace must be non-empty"));
    }
    if currents.iter().any(|current| !current.is_finite()) {
        return Err(PyValueError::new_err(
            "current_trace must contain only finite values",
        ));
    }
    Ok(dt)
}

fn normalise_model_name(name: &str) -> String {
    name.chars()
        .filter(|ch| ch.is_ascii_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect()
}

fn simulate_izhikevich<'py>(py: Python<'py>, currents: &[f64], dt: f64) -> PyResult<Py<PyAny>> {
    let mut neuron = IzhikevichRk4::new(dt);
    let mut v = Vec::with_capacity(currents.len());
    let mut u = Vec::with_capacity(currents.len());
    let mut spikes = Vec::new();
    for (idx, &current) in currents.iter().enumerate() {
        if neuron.step(current) != 0 {
            spikes.push(idx as u64);
        }
        v.push(neuron.v);
        u.push(neuron.u);
    }
    let d = PyDict::new(py);
    d.set_item("v", v.into_pyarray(py))?;
    d.set_item("u", u.into_pyarray(py))?;
    d.set_item("spikes", spikes.into_pyarray(py))?;
    d.set_item("n_steps", currents.len())?;
    Ok(d.into_any().unbind())
}

fn simulate_adex<'py>(py: Python<'py>, currents: &[f64], dt: f64) -> PyResult<Py<PyAny>> {
    let mut neuron = AdExRk4::new(dt);
    let mut v = Vec::with_capacity(currents.len());
    let mut w = Vec::with_capacity(currents.len());
    let mut spikes = Vec::new();
    for (idx, &current) in currents.iter().enumerate() {
        if neuron.step(current) != 0 {
            spikes.push(idx as u64);
        }
        v.push(neuron.v);
        w.push(neuron.w);
    }
    let d = PyDict::new(py);
    d.set_item("v", v.into_pyarray(py))?;
    d.set_item("w", w.into_pyarray(py))?;
    d.set_item("spikes", spikes.into_pyarray(py))?;
    d.set_item("n_steps", currents.len())?;
    Ok(d.into_any().unbind())
}

fn simulate_hodgkin_huxley<'py>(py: Python<'py>, currents: &[f64], dt: f64) -> PyResult<Py<PyAny>> {
    let mut neuron = HodgkinHuxleyRk4::new(dt);
    let mut v = Vec::with_capacity(currents.len());
    let mut m = Vec::with_capacity(currents.len());
    let mut h = Vec::with_capacity(currents.len());
    let mut n = Vec::with_capacity(currents.len());
    let mut spikes = Vec::new();
    for (idx, &current) in currents.iter().enumerate() {
        if neuron.step(current) != 0 {
            spikes.push(idx as u64);
        }
        v.push(neuron.v);
        m.push(neuron.m);
        h.push(neuron.h);
        n.push(neuron.n);
    }
    let d = PyDict::new(py);
    d.set_item("v", v.into_pyarray(py))?;
    d.set_item("m", m.into_pyarray(py))?;
    d.set_item("h", h.into_pyarray(py))?;
    d.set_item("n", n.into_pyarray(py))?;
    d.set_item("spikes", spikes.into_pyarray(py))?;
    d.set_item("n_steps", currents.len())?;
    Ok(d.into_any().unbind())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_name_normalisation_accepts_common_aliases() {
        assert_eq!(
            normalise_model_name("Hodgkin-HuxleyNeuron"),
            "hodgkinhuxleyneuron"
        );
        assert_eq!(normalise_model_name("AdEx"), "adex");
    }
}
