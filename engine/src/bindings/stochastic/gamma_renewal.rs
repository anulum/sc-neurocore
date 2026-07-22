// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Gamma-renewal neuron PyO3 binding

use pyo3::prelude::*;

use crate::neurons;

#[pyclass(
    name = "GammaRenewalNeuron",
    module = "sc_neurocore_engine.sc_neurocore_engine"
)]
#[derive(Clone)]
pub struct PyGammaRenewalNeuron {
    inner: neurons::GammaRenewalNeuron,
}

#[pymethods]
impl PyGammaRenewalNeuron {
    #[new]
    #[pyo3(signature = (rate_hz=50.0, shape_k=3, seed=42))]
    fn new(rate_hz: f64, shape_k: u32, seed: u64) -> Self {
        Self {
            inner: neurons::GammaRenewalNeuron::new(rate_hz, shape_k, seed),
        }
    }

    #[pyo3(signature = (rate_override=-1.0))]
    fn step(&mut self, rate_override: f64) -> i32 {
        self.inner.step(rate_override)
    }

    fn reset(&mut self) {
        self.inner.reset();
    }
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyGammaRenewalNeuron>()?;
    Ok(())
}
