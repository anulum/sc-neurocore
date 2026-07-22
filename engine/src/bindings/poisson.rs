// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Poisson PyO3 binding

//! Python binding for the homogeneous Poisson binary-bin generator.

use numpy::{IntoPyArray, PyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Register the homogeneous Poisson simulator with the extension module.
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(py_poisson_simulate, module)?)?;
    Ok(())
}

/// Full-contract homogeneous-Poisson batch with the canonical LFSR16 trial.
#[pyfunction]
#[pyo3(signature = (rate_hz, dt_ms, rng_state, n_steps, rate_override=-1.0))]
fn py_poisson_simulate<'py>(
    py: Python<'py>,
    rate_hz: f64,
    dt_ms: f64,
    rng_state: u16,
    n_steps: usize,
    rate_override: f64,
) -> PyResult<(Bound<'py, PyArray1<u8>>, u16)> {
    let mut neuron = crate::neurons::PoissonNeuron {
        rate_hz,
        dt_ms,
        rng_state,
        initial_seed: rng_state,
    };
    if !neuron.valid() || !rate_override.is_finite() {
        return Err(PyValueError::new_err(
            "invalid Poisson simulation state or rate override",
        ));
    }
    let mut events = Vec::with_capacity(n_steps);
    for _ in 0..n_steps {
        let spike = neuron
            .try_step(rate_override)
            .map_err(PyValueError::new_err)?;
        events.push(spike as u8);
    }
    Ok((events.into_pyarray(py), neuron.rng_state))
}
