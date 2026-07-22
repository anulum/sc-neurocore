// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Triplet STDP synapse PyO3 binding

use pyo3::prelude::*;
use pyo3::types::PyDict;

#[pyclass(
    name = "RustTripletStdpSynapse",
    module = "sc_neurocore_engine.sc_neurocore_engine"
)]
#[derive(Clone)]
pub struct PyTripletStdpSynapse {
    inner: crate::synapses::TripletStdpSynapse,
}

#[pymethods]
impl PyTripletStdpSynapse {
    #[new]
    #[pyo3(signature = (weight=0.5, w_min=0.0, w_max=1.0))]
    fn new(weight: f64, w_min: f64, w_max: f64) -> Self {
        Self {
            inner: crate::synapses::TripletStdpSynapse::new(weight, w_min, w_max),
        }
    }

    fn step(&mut self, pre_spike: bool, post_spike: bool) {
        self.inner.step(pre_spike, post_spike);
    }

    #[getter]
    fn weight(&self) -> f64 {
        self.inner.weight
    }

    fn get_state(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let d = PyDict::new(py);
        d.set_item("weight", self.inner.weight)?;
        d.set_item("r1", self.inner.r1)?;
        d.set_item("o1", self.inner.o1)?;
        d.set_item("r2", self.inner.r2)?;
        d.set_item("o2", self.inner.o2)?;
        Ok(d.into_any().unbind())
    }
}

/// Register the triplet STDP synapse class.
pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyTripletStdpSynapse>()?;
    Ok(())
}
