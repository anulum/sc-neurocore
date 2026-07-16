// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Ion Channel Variant Neuron Models

//! Compatibility facade for the ion-channel variant neuron family.
//!
//! Seven independent model implementations and their owned tests live in bounded child modules.
//! Historical public re-exports remain unchanged.

mod a_type_k;
mod bk;
mod ih;
mod nmda;
mod persistent_na;
mod sk;
mod t_type_ca;

pub use a_type_k::ATypeKNeuron;
pub use bk::BKNeuron;
pub use ih::IhNeuron;
pub use nmda::NMDANeuron;
pub use persistent_na::PersistentNaNeuron;
pub use sk::SKNeuron;
pub use t_type_ca::TTypeCaNeuron;
