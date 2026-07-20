// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — compatibility facade for former special models

//! Compatibility facade for the former mixed special-model module.
//!
//! Implementations now live in modules named for their mathematical or
//! biological responsibility. Historical re-exports remain stable.

mod larter_breakspear_neural_mass;
mod spike_response_models;
mod stochastic_point_processes;
mod stochastic_spiking_models;
mod wendling_neural_mass;
mod wilson_cowan_population;

#[cfg(test)]
mod canonical_population_reexport_tests;

pub use larter_breakspear_neural_mass::LarterBreakspearNeuron;
pub use spike_response_models::{GLMNeuron, SpikeResponseNeuron};
pub use stochastic_point_processes::{
    GammaRenewalNeuron, InhomogeneousPoissonNeuron, PoissonNeuron,
};
pub use stochastic_spiking_models::{GalvesLocherbachNeuron, StochasticIFNeuron};
pub use wendling_neural_mass::WendlingNeuron;
pub use wilson_cowan_population::WilsonCowanUnit;

/// Canonical Wong-Wang implementation lives in the dedicated engine module.
pub use crate::wong_wang::WongWangUnit;

/// Canonical Montbrió population implementation lives in a dedicated module.
pub use crate::neurons::ermentrout_kopell_pop::ErmentroutKopellPopulation;
