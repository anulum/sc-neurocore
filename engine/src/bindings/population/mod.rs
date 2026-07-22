// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Population-model PyO3 binding composition

use pyo3::prelude::*;

mod brunel_network;
mod brunel_wang;
mod el_boustani_network;
mod larter_breakspear;
mod montbrio_mean_field;
mod tum_network;
mod wendling;
mod wilson_cowan_unit;
mod wong_wang_unit;

pub use brunel_network::PyBrunelNetwork;
pub use brunel_wang::PyBrunelWangNeuron;
pub use el_boustani_network::PyElBoustaniNetwork;
pub use larter_breakspear::PyLarterBreakspearNeuron;
pub use montbrio_mean_field::PyMontbrioMeanField;
pub use tum_network::PyTUMNetwork;
pub use wendling::PyWendlingNeuron;
pub use wilson_cowan_unit::PyWilsonCowanUnit;
pub use wong_wang_unit::PyWongWangUnit;

pub(crate) fn register_brunel_wang(module: &Bound<'_, PyModule>) -> PyResult<()> {
    brunel_wang::register(module)?;
    Ok(())
}

pub(crate) fn register_wilson_cowan_unit(module: &Bound<'_, PyModule>) -> PyResult<()> {
    wilson_cowan_unit::register(module)?;
    Ok(())
}

pub(crate) fn register_wong_wang_unit(module: &Bound<'_, PyModule>) -> PyResult<()> {
    wong_wang_unit::register(module)?;
    Ok(())
}

pub(crate) fn register_neural_mass_tail(module: &Bound<'_, PyModule>) -> PyResult<()> {
    wendling::register(module)?;
    larter_breakspear::register(module)?;
    Ok(())
}

/// Register the four model-owned population classes in stable ABI order.
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    montbrio_mean_field::register(module)?;
    brunel_network::register(module)?;
    tum_network::register(module)?;
    el_boustani_network::register(module)?;
    Ok(())
}
