// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Population-model PyO3 binding composition

use pyo3::prelude::*;

mod brunel_network;
mod el_boustani_network;
mod montbrio_mean_field;
mod tum_network;

pub use brunel_network::PyBrunelNetwork;
pub use el_boustani_network::PyElBoustaniNetwork;
pub use montbrio_mean_field::PyMontbrioMeanField;
pub use tum_network::PyTUMNetwork;

/// Register the four model-owned population classes in stable ABI order.
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    montbrio_mean_field::register(module)?;
    brunel_network::register(module)?;
    tum_network::register(module)?;
    el_boustani_network::register(module)?;
    Ok(())
}
