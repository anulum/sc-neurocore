// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — deprecated compatibility PyO3 identity

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::neurons;

/// Deprecated Python compatibility wrapper for the retained SC project map.
#[pyclass(
    name = "KilincBhattMapNeuron",
    module = "sc_neurocore_engine.sc_neurocore_engine"
)]
#[derive(Clone)]
pub struct PyKilincBhattMapNeuron {
    inner: neurons::SCAdaptiveThresholdMapNeuron,
}

#[pymethods]
impl PyKilincBhattMapNeuron {
    #[new]
    fn new() -> Self {
        Self {
            inner: neurons::SCAdaptiveThresholdMapNeuron::default(),
        }
    }

    fn step(&mut self, current: f64) -> PyResult<i32> {
        self.inner
            .try_step(current)
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    fn reset(&mut self) {
        self.inner.reset();
    }

    fn get_state(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let state = PyDict::new(py);
        state.set_item("x", self.inner.x)?;
        state.set_item("theta", self.inner.theta)?;
        Ok(state.into_any().unbind())
    }
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyKilincBhattMapNeuron>()?;
    Ok(())
}
