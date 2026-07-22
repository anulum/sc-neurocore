// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Learning-rule PyO3 binding composition

use pyo3::prelude::*;

mod eprop_alif;
mod super_spike;

pub use eprop_alif::PyEPropALIFNeuron;
pub use super_spike::PySuperSpikeNeuron;

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    eprop_alif::register(module)?;
    super_spike::register(module)?;
    Ok(())
}
