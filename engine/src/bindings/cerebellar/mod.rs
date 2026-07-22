// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Cerebellar neuron PyO3 binding composition

use pyo3::prelude::*;

mod dcn_neuron;
mod golgi_cell;
mod granule_cell;
mod lugaro_cell;
mod stellate_cell;
mod unipolar_brush_cell;

/// Register the six model-owned cerebellar neuron classes in stable ABI order.
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    granule_cell::register(module)?;
    golgi_cell::register(module)?;
    stellate_cell::register(module)?;
    lugaro_cell::register(module)?;
    unipolar_brush_cell::register(module)?;
    dcn_neuron::register(module)?;
    Ok(())
}
