// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — SCPN metrics PyO3 binding

//! Python binding for aggregate SCPN coherence metrics.

use pyo3::prelude::*;

use crate::scpn;

/// Register aggregate SCPN metrics with the extension module.
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PySCPNMetrics>()?;
    Ok(())
}

#[pyclass(
    name = "SCPNMetrics",
    module = "sc_neurocore_engine.sc_neurocore_engine"
)]
pub struct PySCPNMetrics;

#[pymethods]
impl PySCPNMetrics {
    #[new]
    fn new() -> Self {
        Self
    }

    #[staticmethod]
    fn global_coherence(weights: [f64; 7], metrics: [f64; 7]) -> f64 {
        scpn::SCPNMetrics::global_coherence(&weights, &metrics)
    }

    #[staticmethod]
    fn consciousness_index(phases_l4: Vec<f64>, glyph_l7: [f64; 6]) -> f64 {
        scpn::SCPNMetrics::consciousness_index(&phases_l4, &glyph_l7)
    }
}
