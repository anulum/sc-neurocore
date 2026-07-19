// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Adaptive-threshold PyO3 exact-relaxation batch binding

use numpy::{IntoPyArray, PyReadonlyArray1};
use pyo3::exceptions::{PyFloatingPointError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::neurons::trivial::adaptive_threshold_if::{self, AdaptiveThresholdIFError};

fn map_adaptive_threshold_if_error(error: AdaptiveThresholdIFError) -> PyErr {
    match error {
        AdaptiveThresholdIFError::NonFiniteCandidate => {
            PyFloatingPointError::new_err(error.to_string())
        }
        _ => PyValueError::new_err(error.to_string()),
    }
}

/// Register the configured exact-relaxation batch function.
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(py_adaptive_threshold_if_simulate, module)?)?;
    Ok(())
}

/// Simulate one complete piecewise-constant current batch.
#[pyfunction]
#[pyo3(signature = (v, theta, v_rest, v_reset, theta_rest, delta_theta, tau_m, tau_theta, dt, current))]
#[allow(clippy::too_many_arguments)]
fn py_adaptive_threshold_if_simulate<'py>(
    py: Python<'py>,
    v: f64,
    theta: f64,
    v_rest: f64,
    v_reset: f64,
    theta_rest: f64,
    delta_theta: f64,
    tau_m: f64,
    tau_theta: f64,
    dt: f64,
    current: PyReadonlyArray1<'py, f64>,
) -> PyResult<Py<PyAny>> {
    let drive = current.as_slice()?;
    let result = adaptive_threshold_if::simulate(
        v,
        theta,
        v_rest,
        v_reset,
        theta_rest,
        delta_theta,
        tau_m,
        tau_theta,
        dt,
        drive,
    )
    .map_err(map_adaptive_threshold_if_error)?;
    let mapping = PyDict::new(py);
    mapping.set_item("v", result.v.into_pyarray(py))?;
    mapping.set_item("theta", result.theta.into_pyarray(py))?;
    mapping.set_item("spikes", result.spikes.into_pyarray(py))?;
    mapping.set_item("v_final", result.final_state[0])?;
    mapping.set_item("theta_final", result.final_state[1])?;
    mapping.set_item("spike_count", result.spike_count)?;
    Ok(mapping.into_any().unbind())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn configured_batch_rejects_nonfinite_drive_before_result_construction() {
        let result = adaptive_threshold_if::simulate(
            -65.0,
            -50.0,
            -65.0,
            -65.0,
            -50.0,
            5.0,
            10.0,
            50.0,
            0.1,
            &[0.25, f64::NAN, 0.5],
        );
        assert!(matches!(
            result,
            Err(AdaptiveThresholdIFError::NonFiniteInput)
        ));
    }
}
