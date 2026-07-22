// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Two-compartment LIF neuron PyO3 binding

use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::neurons;

#[pyclass(
    name = "TwoCompartmentLIFNeuron",
    module = "sc_neurocore_engine.sc_neurocore_engine"
)]
#[derive(Clone)]
pub struct PyTwoCompartmentLIFNeuron {
    inner: neurons::TwoCompartmentLIFNeuron,
}

#[pymethods]
impl PyTwoCompartmentLIFNeuron {
    #[new]
    fn new() -> Self {
        Self {
            inner: neurons::TwoCompartmentLIFNeuron::new(),
        }
    }

    #[pyo3(signature = (i_soma, i_dend=0.0))]
    fn step(&mut self, i_soma: f64, i_dend: f64) -> i32 {
        self.inner.step(i_soma, i_dend)
    }

    fn reset(&mut self) {
        self.inner.reset();
    }

    fn get_state(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let d = PyDict::new(py);
        d.set_item("v_s", self.inner.v_s)?;
        d.set_item("v_d", self.inner.v_d)?;
        Ok(d.into_any().unbind())
    }
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyTwoCompartmentLIFNeuron>()?;
    Ok(())
}
