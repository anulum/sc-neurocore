// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Neurons Module

pub mod ai_optimized;
pub mod aihara_map;
pub mod biophysical;
pub mod cazelles_map;
pub mod cerebellar;
pub mod channels;
pub mod chialvo_map;
pub mod courage_nekorkin_map;
pub mod ermentrout_kopell_map;
pub mod ermentrout_kopell_pop;
pub mod hardware;
pub mod ibarz_tanaka_map;
pub mod interneurons;
pub mod jansen_rit;
pub mod kilinc_bhatt_map;
pub mod medvedev_map;
pub mod misc;
pub mod motor;
pub mod multi_compartment;
pub mod nagumo_sato_map;
pub mod population;
pub mod rate;
pub mod rulkov_map;
pub mod sc_adaptive_threshold_map;
pub mod sc_chaotic_map;
pub mod sensory;
pub mod simple_spiking;
pub mod special;
pub mod trivial;

/// Compatibility namespace for callers that used the former aggregate module.
pub mod maps {
    pub use super::cazelles_map::CazellesMapNeuron;
    pub use super::chialvo_map::ChialvoMapNeuron;
    pub use super::courage_nekorkin_map::CourageNekorkinMapNeuron;
    pub use super::ermentrout_kopell_map::ErmentroutKopellMapNeuron;
    pub use super::ibarz_tanaka_map::IbarzTanakaMapNeuron;
    pub use super::kilinc_bhatt_map::KilincBhattMapNeuron;
    pub use super::medvedev_map::MedvedevMapNeuron;
    pub use super::nagumo_sato_map::NagumoSatoMapNeuron;
    pub use super::rulkov_map::RulkovMapNeuron;
    pub use super::sc_adaptive_threshold_map::SCAdaptiveThresholdMapNeuron;
    pub use super::sc_chaotic_map::SCChaoticMapNeuron;
}

pub use ai_optimized::*;
pub use aihara_map::*;
pub use biophysical::*;
pub use cazelles_map::*;
pub use cerebellar::*;
pub use channels::*;
pub use chialvo_map::*;
pub use courage_nekorkin_map::*;
pub use ermentrout_kopell_map::*;
pub use hardware::*;
pub use ibarz_tanaka_map::*;
pub use interneurons::*;
pub use jansen_rit::*;
pub use kilinc_bhatt_map::*;
pub use medvedev_map::*;
pub use misc::*;
pub use motor::*;
pub use multi_compartment::*;
pub use nagumo_sato_map::*;
pub use population::*;
pub use rate::*;
pub use rulkov_map::*;
pub use sc_adaptive_threshold_map::*;
pub use sc_chaotic_map::*;
pub use sensory::*;
pub use simple_spiking::*;
pub use special::*;
pub use trivial::*;
