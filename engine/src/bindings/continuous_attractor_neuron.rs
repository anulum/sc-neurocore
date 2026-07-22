// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Continuous-attractor neuron PyO3 binding

use numpy::IntoPyArray;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::neurons;

#[pyclass(
    name = "RustContinuousAttractorNeuron",
    module = "sc_neurocore_engine.sc_neurocore_engine"
)]
#[derive(Clone)]
pub struct PyContinuousAttractorNeuron {
    inner: neurons::ContinuousAttractorNeuron,
}

#[pymethods]
impl PyContinuousAttractorNeuron {
    #[new]
    #[pyo3(signature = (n_units=16))]
    fn new(n_units: usize) -> Self {
        Self {
            inner: neurons::ContinuousAttractorNeuron::new(n_units),
        }
    }
    fn step(&mut self, current: f64) -> i32 {
        self.inner.step(current)
    }
    fn bump_position(&self) -> usize {
        self.inner.bump_position()
    }
    fn reset(&mut self) {
        self.inner.reset();
    }
    fn get_state(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let d = PyDict::new(py);
        d.set_item("u", self.inner.u.clone().into_pyarray(py))?;
        Ok(d.into_any().unbind())
    }
}

/// Register the continuous-attractor neuron class.
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyContinuousAttractorNeuron>()?;
    Ok(())
}
