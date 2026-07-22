// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — STDP synapse PyO3 binding

//! Python binding for the fixed-point spike-timing-dependent plasticity synapse.

use pyo3::prelude::*;

use crate::synapses;

/// Register the fixed-point STDP synapse with the extension module.
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<StdpSynapse>()?;
    Ok(())
}

#[pyclass(
    name = "StdpSynapse",
    module = "sc_neurocore_engine.sc_neurocore_engine"
)]
pub struct StdpSynapse {
    inner: synapses::StdpSynapse,
}

#[pymethods]
impl StdpSynapse {
    #[new]
    #[pyo3(signature = (initial_weight, data_width=16, fraction=8))]
    fn new(initial_weight: i16, data_width: u32, fraction: u32) -> Self {
        Self {
            inner: synapses::StdpSynapse::new(initial_weight, data_width, fraction),
        }
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (pre_spike, post_spike, a_plus=16, a_minus=-16, decay=250, w_min=0, w_max=32767))]
    fn step(
        &mut self,
        pre_spike: bool,
        post_spike: bool,
        a_plus: i16,
        a_minus: i16,
        decay: i16,
        w_min: i16,
        w_max: i16,
    ) {
        let params = synapses::StdpParams {
            a_plus,
            a_minus,
            decay,
            w_min,
            w_max,
        };
        self.inner.step(pre_spike, post_spike, &params);
    }

    #[getter]
    fn weight(&self) -> i16 {
        self.inner.weight
    }

    #[setter]
    fn set_weight(&mut self, value: i16) {
        self.inner.weight = value;
    }
}
