// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Parallel-spiking neuron PyO3 binding

use pyo3::prelude::*;

use crate::neurons;

#[pyclass(
    name = "ParallelSpikingNeuron",
    module = "sc_neurocore_engine.sc_neurocore_engine"
)]
#[derive(Clone)]
pub struct PyParallelSpikingNeuron {
    inner: neurons::ParallelSpikingNeuron,
}

#[pymethods]
impl PyParallelSpikingNeuron {
    #[new]
    #[pyo3(signature = (kernel_size=8, v_threshold=1.0))]
    fn new(kernel_size: usize, v_threshold: f64) -> Self {
        Self {
            inner: neurons::ParallelSpikingNeuron::new(kernel_size, v_threshold),
        }
    }

    fn step(&mut self, current: f64) -> i32 {
        self.inner.step(current)
    }

    fn reset(&mut self) {
        self.inner.reset();
    }
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyParallelSpikingNeuron>()?;
    Ok(())
}
