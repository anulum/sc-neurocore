// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Two-dimensional and higher spiking neuron models

//! Compatibility facade for two-dimensional and higher spiking neuron models.
//!
//! Each model owns its implementation and tests in a bounded child module
//! while the historical public re-exports remain unchanged.

mod alpha;
mod balanced_resonate_and_fire;
mod benda_herz;
mod brunel_wang;
mod butera_respiratory;
mod chay;
mod chay_keizer;
mod coba_lif;
mod e_prop_alif;
mod fitzhugh_nagumo;
mod fitzhugh_rinzel;
mod gutkin_ermentrout;
mod hindmarsh_rose;
mod lnm;
mod mckean;
mod morris_lecar;
mod pernarowski;
mod resonate_and_fire;
mod sherman_rinzel_keizer;
mod superspike_neuron;
mod terman_wang;
mod wilson_hr;

pub use alpha::AlphaNeuron;
pub use balanced_resonate_and_fire::{
    brf_sustain_oscillation_boundary, BalancedResonateAndFireNeuron,
};
pub use benda_herz::BendaHerzNeuron;
pub use brunel_wang::BrunelWangNeuron;
pub use butera_respiratory::ButeraRespiratoryNeuron;
pub use chay::ChayNeuron;
pub use chay_keizer::ChayKeizerNeuron;
pub use coba_lif::COBALIFNeuron;
pub use e_prop_alif::EPropALIFNeuron;
pub use fitzhugh_nagumo::FitzHughNagumoNeuron;
pub use fitzhugh_rinzel::FitzHughRinzelNeuron;
pub use gutkin_ermentrout::GutkinErmentroutNeuron;
pub use hindmarsh_rose::HindmarshRoseNeuron;
pub use lnm::LearnableNeuronModel;
pub use mckean::McKeanNeuron;
pub use morris_lecar::MorrisLecarNeuron;
pub use pernarowski::PernarowskiNeuron;
pub use resonate_and_fire::ResonateAndFireNeuron;
pub use sherman_rinzel_keizer::ShermanRinzelKeizerNeuron;
pub use superspike_neuron::SuperSpikeNeuron;
pub use terman_wang::TermanWangOscillator;
pub use wilson_hr::WilsonHRNeuron;
