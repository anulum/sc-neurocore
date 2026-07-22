// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Multicompartment MCN neuron PyO3 binding

use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::neurons;

#[pyclass(
    name = "RustMulticompartmentMCNNeuron",
    module = "sc_neurocore_engine.sc_neurocore_engine"
)]
#[derive(Clone)]
pub struct PyMulticompartmentMCNNeuron {
    inner: neurons::MulticompartmentMCNNeuron,
}

#[pymethods]
impl PyMulticompartmentMCNNeuron {
    #[new]
    fn new() -> Self {
        Self {
            inner: neurons::MulticompartmentMCNNeuron::new(),
        }
    }

    fn step(&mut self, current: f64) -> i32 {
        self.inner.step(current)
    }

    fn step_compartments(&mut self, x_basal: f64, x_apical: f64, i_soma: f64) -> i32 {
        self.inner.step_compartments(x_basal, x_apical, i_soma)
    }

    fn reset(&mut self) {
        self.inner.reset();
    }

    fn get_state(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let d = PyDict::new(py);
        d.set_item("u", self.inner.u)?;
        d.set_item("v_basal", self.inner.v_basal)?;
        d.set_item("v_apical", self.inner.v_apical)?;
        Ok(d.into_any().unbind())
    }
}

/// Register the multicompartment MCN neuron class.
pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyMulticompartmentMCNNeuron>()?;
    Ok(())
}
