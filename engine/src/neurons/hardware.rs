// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Hardware Neuron Emulator Facade

//! Stable exports for hardware neuromorphic chip emulators.

mod akida;
mod brainscales_adex;
mod dpi_neuron;
mod loihi2;
mod loihi_cuba;
mod neurogrid;
mod spinnaker2;
mod spinnaker_lif;
mod truenorth;

pub use akida::AkidaNeuron;
pub use brainscales_adex::BrainScaleSAdExNeuron;
pub use dpi_neuron::DPINeuron;
pub use loihi2::Loihi2Neuron;
pub use loihi_cuba::LoihiCUBANeuron;
pub use neurogrid::NeuroGridNeuron;
pub use spinnaker2::SpiNNaker2Neuron;
pub use spinnaker_lif::SpiNNakerLIFNeuron;
pub use truenorth::TrueNorthNeuron;
