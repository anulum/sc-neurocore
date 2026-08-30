// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Quadratic integrate-and-fire neuron PyO3 binding

use numpy::{IntoPyArray, PyArray1};
use pyo3::exceptions::PyFloatingPointError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::neurons;

py_neuron_default!("QuadraticIFNeuron", PyQuadraticIFNeuron, neurons::QuadraticIFNeuron, state v);

type QuadraticIFCompletePacket<'py> = (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<u8>>, f64);

#[pyfunction]
#[pyo3(signature = (v, v_reset, v_peak, dt, source_profile, n_steps, current))]
#[allow(clippy::too_many_arguments)]
fn quadratic_if_simulate_complete<'py>(
    py: Python<'py>,
    v: f64,
    v_reset: f64,
    v_peak: f64,
    dt: f64,
    source_profile: bool,
    n_steps: usize,
    current: f64,
) -> PyResult<QuadraticIFCompletePacket<'py>> {
    let model = neurons::QuadraticIFNeuron {
        v,
        v_reset,
        v_peak,
        dt,
        source_profile,
    };
    let (voltage, events, final_v) = model
        .simulate_complete(n_steps, current)
        .map_err(|error| PyFloatingPointError::new_err(error.to_string()))?;
    Ok((voltage.into_pyarray(py), events.into_pyarray(py), final_v))
}

/// Register the quadratic integrate-and-fire neuron class.
pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyQuadraticIFNeuron>()?;
    module.add_function(wrap_pyfunction!(quadratic_if_simulate_complete, module)?)?;
    Ok(())
}
