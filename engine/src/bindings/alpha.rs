// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Alpha-synapse PyO3 exact-flow batch binding

use numpy::{IntoPyArray, PyReadonlyArray1};
use pyo3::exceptions::{PyFloatingPointError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::neurons::simple_spiking::alpha::{self, AlphaError};

fn map_alpha_error(error: AlphaError) -> PyErr {
    match error {
        AlphaError::NonFiniteCandidate => PyFloatingPointError::new_err(error.to_string()),
        _ => PyValueError::new_err(error.to_string()),
    }
}

/// Register the configured exact-flow batch function.
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(py_alpha_simulate, module)?)?;
    Ok(())
}

/// Simulate one complete piecewise-constant excitatory/inhibitory drive batch.
#[pyfunction]
#[pyo3(signature = (v, a_exc, i_exc, a_inh, i_inh, v_rest, v_threshold, tau_v, tau_exc, tau_inh, dt, exc_current, inh_current))]
#[allow(clippy::too_many_arguments)]
fn py_alpha_simulate<'py>(
    py: Python<'py>,
    v: f64,
    a_exc: f64,
    i_exc: f64,
    a_inh: f64,
    i_inh: f64,
    v_rest: f64,
    v_threshold: f64,
    tau_v: f64,
    tau_exc: f64,
    tau_inh: f64,
    dt: f64,
    exc_current: PyReadonlyArray1<'py, f64>,
    inh_current: PyReadonlyArray1<'py, f64>,
) -> PyResult<Py<PyAny>> {
    let exc_drive = exc_current.as_slice()?;
    let inh_drive = inh_current.as_slice()?;
    let result = alpha::simulate(
        v,
        a_exc,
        i_exc,
        a_inh,
        i_inh,
        v_rest,
        v_threshold,
        tau_v,
        tau_exc,
        tau_inh,
        dt,
        exc_drive,
        inh_drive,
    )
    .map_err(map_alpha_error)?;
    let mapping = PyDict::new(py);
    mapping.set_item("v", result.v.into_pyarray(py))?;
    mapping.set_item("a_exc", result.a_exc.into_pyarray(py))?;
    mapping.set_item("i_exc", result.i_exc.into_pyarray(py))?;
    mapping.set_item("a_inh", result.a_inh.into_pyarray(py))?;
    mapping.set_item("i_inh", result.i_inh.into_pyarray(py))?;
    mapping.set_item("spikes", result.spikes.into_pyarray(py))?;
    mapping.set_item("v_final", result.final_state[0])?;
    mapping.set_item("a_exc_final", result.final_state[1])?;
    mapping.set_item("i_exc_final", result.final_state[2])?;
    mapping.set_item("a_inh_final", result.final_state[3])?;
    mapping.set_item("i_inh_final", result.final_state[4])?;
    mapping.set_item("spike_count", result.spike_count)?;
    Ok(mapping.into_any().unbind())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn configured_batch_rejects_nonfinite_drive_before_result_construction() {
        let result = alpha::simulate(
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            20.0,
            5.0,
            10.0,
            1.0,
            &[0.25, f64::NAN, 0.5],
            &[0.1, 0.1, 0.1],
        );
        assert!(matches!(result, Err(AlphaError::NonFiniteInput)));
    }

    #[test]
    fn configured_batch_rejects_mismatched_drive_lengths() {
        let result = alpha::simulate(
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            20.0,
            5.0,
            10.0,
            1.0,
            &[0.25, 0.5],
            &[0.1],
        );
        assert!(matches!(result, Err(AlphaError::NonFiniteInput)));
    }
}
