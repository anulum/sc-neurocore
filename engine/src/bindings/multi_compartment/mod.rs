// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Multi-compartment PyO3 binding composition

use pyo3::prelude::*;

mod astrocyte_lif;
mod dendritic_nmda;
mod multicompartment_mcn;

pub use astrocyte_lif::PyAstrocyteLIFNeuron;
pub use dendritic_nmda::PyDendriticNMDANeuron;
pub use multicompartment_mcn::PyMulticompartmentMCNNeuron;

/// Register the three specialised multi-compartment neuron classes.
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    dendritic_nmda::register(module)?;
    multicompartment_mcn::register(module)?;
    astrocyte_lif::register(module)?;
    Ok(())
}
