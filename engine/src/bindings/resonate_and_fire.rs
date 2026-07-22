// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Resonate-and-fire PyO3 exact-flow batch binding

use numpy::{IntoPyArray, PyReadonlyArray1};
use pyo3::exceptions::{PyFloatingPointError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::neurons;
use crate::neurons::simple_spiking::resonate_and_fire::{self, ResonateAndFireError};

py_neuron_default!("ResonateAndFireNeuron", PyResonateAndFireNeuron, neurons::ResonateAndFireNeuron, state x, state y);

fn map_resonate_and_fire_error(error: ResonateAndFireError) -> PyErr {
    match error {
        ResonateAndFireError::NonFiniteCandidate => {
            PyFloatingPointError::new_err(error.to_string())
        }
        _ => PyValueError::new_err(error.to_string()),
    }
}

/// Register the resonate-and-fire class and exact-flow batch function.
pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyResonateAndFireNeuron>()?;
    module.add_function(wrap_pyfunction!(py_resonate_and_fire_simulate, module)?)?;
    Ok(())
}

/// Simulate one complete piecewise-constant current batch.
#[pyfunction]
#[pyo3(signature = (x, y, b, omega, threshold, dt, current))]
fn py_resonate_and_fire_simulate<'py>(
    py: Python<'py>,
    x: f64,
    y: f64,
    b: f64,
    omega: f64,
    threshold: f64,
    dt: f64,
    current: PyReadonlyArray1<'py, f64>,
) -> PyResult<Py<PyAny>> {
    let drive = current.as_slice()?;
    let result = resonate_and_fire::simulate(x, y, b, omega, threshold, dt, drive)
        .map_err(map_resonate_and_fire_error)?;
    let mapping = PyDict::new(py);
    mapping.set_item("x", result.x.into_pyarray(py))?;
    mapping.set_item("y", result.y.into_pyarray(py))?;
    mapping.set_item("spikes", result.spikes.into_pyarray(py))?;
    mapping.set_item("x_final", result.final_state[0])?;
    mapping.set_item("y_final", result.final_state[1])?;
    mapping.set_item("spike_count", result.spike_count)?;
    Ok(mapping.into_any().unbind())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn configured_batch_rejects_nonfinite_drive_before_result_construction() {
        let result =
            resonate_and_fire::simulate(0.0, 0.0, -1.0, 10.0, 1.0, 0.01, &[0.25, f64::NAN, 0.5]);
        assert!(matches!(result, Err(ResonateAndFireError::NonFiniteInput)));
    }
}
