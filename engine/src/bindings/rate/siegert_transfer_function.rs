// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Siegert transfer-function PyO3 binding

use pyo3::prelude::*;

use crate::neurons;

#[pyclass(
    name = "SiegertTransferFunction",
    module = "sc_neurocore_engine.sc_neurocore_engine"
)]
#[derive(Clone)]
pub struct PySiegertTransferFunction {
    inner: neurons::SiegertTransferFunction,
}

#[pymethods]
impl PySiegertTransferFunction {
    #[new]
    fn new() -> Self {
        Self {
            inner: neurons::SiegertTransferFunction::new(),
        }
    }

    fn step(&self, current: f64) -> f64 {
        self.inner.step(current)
    }
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PySiegertTransferFunction>()?;
    Ok(())
}
