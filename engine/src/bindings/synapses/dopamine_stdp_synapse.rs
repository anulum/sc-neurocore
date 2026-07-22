// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Dopamine-modulated STDP synapse PyO3 binding

use pyo3::prelude::*;
use pyo3::types::PyDict;

#[pyclass(
    name = "RustDopamineStdpSynapse",
    module = "sc_neurocore_engine.sc_neurocore_engine"
)]
#[derive(Clone)]
pub struct PyDopamineStdpSynapse {
    inner: crate::synapses::DopamineStdpSynapse,
}

#[pymethods]
impl PyDopamineStdpSynapse {
    #[new]
    #[pyo3(signature = (weight=0.5, w_min=0.0, w_max=1.0))]
    fn new(weight: f64, w_min: f64, w_max: f64) -> Self {
        Self {
            inner: crate::synapses::DopamineStdpSynapse::new(weight, w_min, w_max),
        }
    }

    fn step(&mut self, pre_spike: bool, post_spike: bool, reward: f64) {
        self.inner.step(pre_spike, post_spike, reward);
    }

    fn reset(&mut self) {
        self.inner.reset();
    }

    #[getter]
    fn weight(&self) -> f64 {
        self.inner.weight
    }

    #[getter]
    fn dopamine(&self) -> f64 {
        self.inner.dopamine
    }

    #[getter]
    fn eligibility(&self) -> f64 {
        self.inner.eligibility
    }

    fn get_state(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let d = PyDict::new(py);
        d.set_item("weight", self.inner.weight)?;
        d.set_item("eligibility", self.inner.eligibility)?;
        d.set_item("dopamine", self.inner.dopamine)?;
        d.set_item("trace_pre", self.inner.trace_pre)?;
        d.set_item("trace_post", self.inner.trace_post)?;
        Ok(d.into_any().unbind())
    }
}

/// Register the dopamine-modulated STDP synapse class.
pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyDopamineStdpSynapse>()?;
    Ok(())
}
