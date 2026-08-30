// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Theta neuron PyO3 binding

use numpy::{IntoPyArray, PyArray1};
use pyo3::exceptions::PyFloatingPointError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::neurons;

py_neuron_default!("ThetaNeuron", PyThetaNeuron, neurons::ThetaNeuron, state theta);

type ThetaCompletePacket<'py> = (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<u8>>, f64);

#[pyfunction]
#[pyo3(signature = (theta, dt, n_steps, current))]
fn theta_simulate_complete<'py>(
    py: Python<'py>,
    theta: f64,
    dt: f64,
    n_steps: usize,
    current: f64,
) -> PyResult<ThetaCompletePacket<'py>> {
    let model = neurons::ThetaNeuron { theta, dt };
    let (phase, events, final_theta) = model
        .simulate_complete(n_steps, current)
        .map_err(|error| PyFloatingPointError::new_err(error.to_string()))?;
    Ok((phase.into_pyarray(py), events.into_pyarray(py), final_theta))
}

/// Register the theta neuron class.
pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyThetaNeuron>()?;
    module.add_function(wrap_pyfunction!(theta_simulate_complete, module)?)?;
    Ok(())
}
