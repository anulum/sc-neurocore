// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Synapse PyO3 binding composition

use pyo3::prelude::*;

mod dopamine_stdp_synapse;
mod short_term_plasticity_synapse;
mod triplet_stdp_synapse;

/// Register the three model-owned synapse classes in stable ABI order.
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    triplet_stdp_synapse::register(module)?;
    short_term_plasticity_synapse::register(module)?;
    dopamine_stdp_synapse::register(module)?;
    Ok(())
}
