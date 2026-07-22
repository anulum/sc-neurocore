// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Wendling neural-mass PyO3 binding

use pyo3::prelude::*;

use crate::neurons;

#[pyclass(
    name = "WendlingNeuron",
    module = "sc_neurocore_engine.sc_neurocore_engine"
)]
#[derive(Clone)]
pub struct PyWendlingNeuron {
    inner: neurons::WendlingNeuron,
}

#[pymethods]
impl PyWendlingNeuron {
    #[new]
    fn new() -> Self {
        Self {
            inner: neurons::WendlingNeuron::new(),
        }
    }

    #[pyo3(signature = (p_ext=220.0))]
    fn step(&mut self, p_ext: f64) -> f64 {
        self.inner.step(p_ext)
    }

    fn reset(&mut self) {
        self.inner.reset();
    }
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyWendlingNeuron>()?;
    Ok(())
}
