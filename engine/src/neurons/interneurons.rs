// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Interneuron compatibility facade

//! Compatibility facade for bounded cortical and cerebellar interneuron modules.

mod cerebellar_basket_neuron;
mod chandelier_neuron;
mod martinotti_neuron;
mod pv_fast_spiking;
mod sst_neuron;
mod vip_neuron;

pub use cerebellar_basket_neuron::CerebellarBasketNeuron;
pub use chandelier_neuron::ChandelierNeuron;
pub use martinotti_neuron::MartinottiNeuron;
pub use pv_fast_spiking::PVFastSpikingNeuron;
pub use sst_neuron::SSTNeuron;
pub use vip_neuron::VIPNeuron;
