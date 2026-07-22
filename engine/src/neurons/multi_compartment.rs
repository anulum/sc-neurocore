// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Multi-compartment neuron models

//! Multi-compartment neuron models grouped by model responsibility.
//!
//! Each private child owns one model's equations, state, and tests.
//! Public re-exports preserve the established module surface.

mod astrocyte_lif;
mod booth_rinzel;
mod dendrify;
mod dendritic_nmda;
mod hay_l5;
mod marder_stg;
mod multicompartment_mcn;
mod pinsky_rinzel;
mod rall_cable;
mod two_compartment_lif;

pub use astrocyte_lif::AstrocyteLIFNeuron;
pub use booth_rinzel::BoothRinzelNeuron;
pub use dendrify::DendrifyNeuron;
pub use dendritic_nmda::DendriticNMDANeuron;
pub use hay_l5::HayL5PyramidalNeuron;
pub use marder_stg::MarderSTGNeuron;
pub use multicompartment_mcn::MulticompartmentMCNNeuron;
pub use pinsky_rinzel::PinskyRinzelNeuron;
pub use rall_cable::RallCableNeuron;
pub use two_compartment_lif::TwoCompartmentLIFNeuron;
