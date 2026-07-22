// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rall dendrite PyO3 binding

//! Python binding for the branched Rall cable-model simulator.

use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

use crate::rall_dendrite;

/// Register the Rall dendrite simulator with the extension module.
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyRallDendrite>()?;
    Ok(())
}

#[pyclass(
    name = "RallDendriteRust",
    module = "sc_neurocore_engine.sc_neurocore_engine"
)]
pub struct PyRallDendrite {
    inner: rall_dendrite::RallDendriteRust,
}

#[pymethods]
impl PyRallDendrite {
    #[new]
    fn new(n_branches: usize, branch_length: usize, tau: f64, coupling: f64, dt: f64) -> Self {
        Self {
            inner: rall_dendrite::RallDendriteRust::new(
                n_branches,
                branch_length,
                tau,
                coupling,
                dt,
            ),
        }
    }

    fn step(&mut self, branch_inputs: PyReadonlyArray1<'_, f64>) -> PyResult<f64> {
        let inputs = branch_inputs.as_slice()?;
        Ok(self.inner.step(inputs))
    }

    fn reset(&mut self) {
        self.inner.reset();
    }

    #[getter]
    fn soma_v(&self) -> f64 {
        self.inner.soma_v
    }
}
