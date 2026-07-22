// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Inhomogeneous Poisson neuron PyO3 binding

use pyo3::prelude::*;

use crate::neurons;

#[pyclass(
    name = "InhomogeneousPoissonNeuron",
    module = "sc_neurocore_engine.sc_neurocore_engine"
)]
#[derive(Clone)]
pub struct PyInhomogeneousPoissonNeuron {
    inner: neurons::InhomogeneousPoissonNeuron,
}

#[pymethods]
impl PyInhomogeneousPoissonNeuron {
    #[new]
    #[pyo3(signature = (dt_ms=1.0, seed=42))]
    fn new(dt_ms: f64, seed: u64) -> Self {
        Self {
            inner: neurons::InhomogeneousPoissonNeuron::new(dt_ms, seed),
        }
    }

    fn step(&mut self, rate_hz: f64) -> i32 {
        self.inner.step(rate_hz)
    }

    fn reset(&mut self) {
        self.inner.reset();
    }
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyInhomogeneousPoissonNeuron>()?;
    Ok(())
}
