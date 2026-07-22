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
use pyo3::types::PyDict;

use crate::neurons;

/// Register the homogeneous Poisson simulator with the extension module.
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyPoissonNeuron>()?;
    module.add_function(wrap_pyfunction!(py_poisson_simulate, module)?)?;
    Ok(())
}

#[pyclass(
    name = "PoissonNeuron",
    module = "sc_neurocore_engine.sc_neurocore_engine"
)]
#[derive(Clone)]
pub struct PyPoissonNeuron {
    inner: neurons::PoissonNeuron,
}

#[pymethods]
impl PyPoissonNeuron {
    #[new]
    #[pyo3(signature = (rate_hz=100.0, dt_ms=1.0, seed=0xACE1))]
    fn new(rate_hz: f64, dt_ms: f64, seed: u64) -> PyResult<Self> {
        let inner = neurons::PoissonNeuron::new(rate_hz, dt_ms, seed);
        if !inner.valid() {
            return Err(PyValueError::new_err(
                "invalid Poisson rate, timestep, or seed",
            ));
        }
        Ok(Self { inner })
    }
    #[pyo3(signature = (rate_override=-1.0))]
    fn step(&mut self, rate_override: f64) -> PyResult<i32> {
        self.inner
            .try_step(rate_override)
            .map_err(PyValueError::new_err)
    }
    fn reset(&mut self) {
        self.inner.reset();
    }
    fn get_state(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let d = PyDict::new(py);
        d.set_item("rate_hz", self.inner.rate_hz)?;
        d.set_item("dt_ms", self.inner.dt_ms)?;
        d.set_item("rng_state", self.inner.rng_state)?;
        d.set_item("initial_seed", self.inner.initial_seed)?;
        Ok(d.into_any().unbind())
    }
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
