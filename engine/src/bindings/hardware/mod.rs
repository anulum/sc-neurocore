// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Hardware-target PyO3 binding composition

use pyo3::prelude::*;

mod akida;
mod brainscales_adex;
mod dpi;
mod loihi2;
mod loihi_cuba;
mod neurogrid;
mod spinnaker2;
mod spinnaker_lif;
mod truenorth;

pub use akida::PyAkidaNeuron;
pub use brainscales_adex::PyBrainScaleSAdExNeuron;
pub use dpi::PyDPINeuron;
pub use loihi2::PyLoihi2Neuron;
pub use loihi_cuba::PyLoihiCUBANeuron;
pub use neurogrid::PyNeuroGridNeuron;
pub use spinnaker2::PySpiNNaker2Neuron;
pub use spinnaker_lif::PySpiNNakerLIFNeuron;
pub use truenorth::PyTrueNorthNeuron;

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    loihi_cuba::register(module)?;
    loihi2::register(module)?;
    truenorth::register(module)?;
    brainscales_adex::register(module)?;
    spinnaker_lif::register(module)?;
    spinnaker2::register(module)?;
    dpi::register(module)?;
    akida::register(module)?;
    neurogrid::register(module)?;
    Ok(())
}
