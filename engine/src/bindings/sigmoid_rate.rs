// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Sigmoid-rate PyO3 simulation binding

//! Python binding for the configurable exact-relaxation sigmoid-rate contract.

use numpy::{IntoPyArray, PyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Register this binding through the neuron registry rather than the crate root.
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(py_sigmoid_rate_simulate, module)?)?;
    Ok(())
}

fn simulate_sigmoid_rate(
    r: f64,
    tau: f64,
    beta: f64,
    theta: f64,
    dt: f64,
    n_steps: usize,
    current: f64,
) -> Result<(Vec<f64>, f64), String> {
    let mut neuron = crate::neurons::SigmoidRateNeuron::with_parameters(r, tau, beta, theta, dt)?;
    let mut trace = Vec::with_capacity(n_steps);
    for _ in 0..n_steps {
        trace.push(neuron.try_step(current)?);
    }
    Ok((trace, neuron.r))
}

/// Simulate a constant-current exact-relaxation rate trajectory.
#[pyfunction]
#[pyo3(signature = (r, tau, beta, theta, dt, n_steps, current))]
fn py_sigmoid_rate_simulate<'py>(
    py: Python<'py>,
    r: f64,
    tau: f64,
    beta: f64,
    theta: f64,
    dt: f64,
    n_steps: usize,
    current: f64,
) -> PyResult<(Bound<'py, PyArray1<f64>>, f64)> {
    let (trace, final_rate) = simulate_sigmoid_rate(r, tau, beta, theta, dt, n_steps, current)
        .map_err(PyValueError::new_err)?;
    Ok((trace.into_pyarray(py), final_rate))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batch_matches_python_exact_relaxation_golden() {
        let (trace, final_rate) = simulate_sigmoid_rate(0.25, 10.0, 2.0, 1.0, 0.5, 6, 3.0).unwrap();
        let expected = [
            0.2857007338135623,
            0.3196603222932904,
            0.3519636820991432,
            0.38269158845670403,
            0.41192087713731845,
            0.43972463658754457,
        ];
        assert_eq!(trace.len(), expected.len());
        for (actual, target) in trace.into_iter().zip(expected) {
            assert!((actual - target).abs() <= 2.0e-15, "{actual} != {target}");
        }
        assert!((final_rate - expected[5]).abs() <= 2.0e-15);
    }

    #[test]
    fn empty_batch_preserves_initial_rate() {
        let (trace, final_rate) = simulate_sigmoid_rate(0.25, 10.0, 2.0, 1.0, 0.5, 0, 3.0).unwrap();
        assert!(trace.is_empty());
        assert_eq!(final_rate, 0.25);
    }

    #[test]
    fn batch_rejects_invalid_contracts() {
        assert!(simulate_sigmoid_rate(-0.1, 10.0, 2.0, 1.0, 0.5, 1, 3.0).is_err());
        assert!(simulate_sigmoid_rate(0.25, 0.0, 2.0, 1.0, 0.5, 1, 3.0).is_err());
        assert!(simulate_sigmoid_rate(0.25, 10.0, 2.0, 1.0, 0.5, 1, f64::NAN).is_err());
    }

    #[test]
    fn large_timestep_batch_remains_in_unit_interval() {
        let (trace, final_rate) =
            simulate_sigmoid_rate(1.0, 0.1, 1.0, 0.0, 5.0, 2, -100.0).unwrap();
        assert!(trace.iter().all(|rate| (0.0..=1.0).contains(rate)));
        assert!((0.0..=1.0).contains(&final_rate));
    }
}
