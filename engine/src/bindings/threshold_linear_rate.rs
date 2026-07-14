// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Threshold-linear rate PyO3 simulation binding

//! Python binding for the configurable memoryless threshold-linear contract.

use numpy::{IntoPyArray, PyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Register this binding through the neuron registry rather than the crate root.
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(py_threshold_linear_rate_simulate, module)?)?;
    Ok(())
}

fn simulate_threshold_linear_rate(
    r: f64,
    theta: f64,
    gain: f64,
    n_steps: usize,
    current: f64,
) -> Result<(Vec<f64>, f64), String> {
    let mut neuron = crate::neurons::ThresholdLinearRateNeuron::with_parameters(r, theta, gain)?;
    let mut trace = Vec::with_capacity(n_steps);
    for _ in 0..n_steps {
        trace.push(neuron.try_step(current)?);
    }
    Ok((trace, neuron.r))
}

/// Evaluate a constant-input threshold-linear rate trace.
#[pyfunction]
#[pyo3(signature = (r, theta, gain, n_steps, current))]
fn py_threshold_linear_rate_simulate<'py>(
    py: Python<'py>,
    r: f64,
    theta: f64,
    gain: f64,
    n_steps: usize,
    current: f64,
) -> PyResult<(Bound<'py, PyArray1<f64>>, f64)> {
    let (trace, final_rate) = simulate_threshold_linear_rate(r, theta, gain, n_steps, current)
        .map_err(PyValueError::new_err)?;
    Ok((trace.into_pyarray(py), final_rate))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn configured_batch_matches_piecewise_linear_golden() {
        let (trace, final_rate) = simulate_threshold_linear_rate(0.25, 1.5, 2.0, 6, 3.0).unwrap();
        assert_eq!(trace, vec![3.0; 6]);
        assert_eq!(final_rate, 3.0);
    }

    #[test]
    fn empty_batch_preserves_initial_rate() {
        let (trace, final_rate) = simulate_threshold_linear_rate(0.25, 1.5, 2.0, 0, 3.0).unwrap();
        assert!(trace.is_empty());
        assert_eq!(final_rate, 0.25);
    }

    #[test]
    fn batch_rejects_invalid_contracts() {
        assert!(simulate_threshold_linear_rate(-0.1, 1.5, 2.0, 1, 3.0).is_err());
        assert!(simulate_threshold_linear_rate(0.25, f64::NAN, 2.0, 1, 3.0).is_err());
        assert!(simulate_threshold_linear_rate(0.25, 1.5, -2.0, 1, 3.0).is_err());
        assert!(simulate_threshold_linear_rate(0.25, 1.5, 2.0, 1, f64::NAN).is_err());
    }

    #[test]
    fn overflow_rejection_is_atomic() {
        assert!(simulate_threshold_linear_rate(0.25, 0.0, 1.0e308, 1, 1.0e308).is_err());
    }
}
