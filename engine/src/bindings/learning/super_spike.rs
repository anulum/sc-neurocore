// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — SuperSpike neuron PyO3 binding

use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::neurons;

#[pyclass(
    name = "SuperSpikeNeuron",
    module = "sc_neurocore_engine.sc_neurocore_engine"
)]
#[derive(Clone)]
pub struct PySuperSpikeNeuron {
    inner: neurons::SuperSpikeNeuron,
}

#[pymethods]
impl PySuperSpikeNeuron {
    #[new]
    #[pyo3(signature = (tau_m=10.0, tau_e=10.0, dt=1.0))]
    fn new(tau_m: f64, tau_e: f64, dt: f64) -> Self {
        Self {
            inner: neurons::SuperSpikeNeuron::new(tau_m, tau_e, dt),
        }
    }

    fn step(&mut self, current: f64) -> i32 {
        self.inner.step(current)
    }

    fn reset(&mut self) {
        self.inner.reset();
    }

    fn get_state(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let d = PyDict::new(py);
        d.set_item("v", self.inner.v)?;
        d.set_item("trace", self.inner.trace)?;
        Ok(d.into_any().unbind())
    }
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PySuperSpikeNeuron>()?;
    Ok(())
}
