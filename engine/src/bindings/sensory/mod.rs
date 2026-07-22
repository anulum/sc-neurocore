// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Sensory-neuron PyO3 binding composition

use pyo3::prelude::*;

#[macro_use]
mod graded_binding;

mod cone_photoreceptor;
mod inner_hair_cell;
mod merkel_cell;
mod nociceptor;
mod olfactory_receptor_neuron;
mod outer_hair_cell;
mod pacinian_corpuscle;
mod retinal_ganglion_cell;
mod rod_photoreceptor;
mod taste_receptor_cell;

/// Register the ten model-owned sensory-neuron classes in stable ABI order.
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    inner_hair_cell::register(module)?;
    outer_hair_cell::register(module)?;
    rod_photoreceptor::register(module)?;
    cone_photoreceptor::register(module)?;
    retinal_ganglion_cell::register(module)?;
    merkel_cell::register(module)?;
    pacinian_corpuscle::register(module)?;
    nociceptor::register(module)?;
    olfactory_receptor_neuron::register(module)?;
    taste_receptor_cell::register(module)?;
    Ok(())
}
