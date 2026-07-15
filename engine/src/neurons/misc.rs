// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Miscellaneous Neuron and Cell Models

//! Compatibility facade for the former miscellaneous model collection.
//!
//! Implementations are grouped by biological responsibility while the
//! historical public re-exports remain unchanged.

mod cardiac_purkinje;
mod endocrine_beta_cell;
mod gap_junction;
mod graded_synapse;
mod myelinated_axon;
mod smooth_muscle;

pub use cardiac_purkinje::CardiacPurkinjeFibre;
pub use endocrine_beta_cell::EndocrineBetaCell;
pub use gap_junction::GapJunctionNeuron;
pub use graded_synapse::GradedSynapseNeuron;
pub use myelinated_axon::{FrankenhaeUserHuxleyAxon, MyelinatedAxon, NodeOfRanvier};
pub use smooth_muscle::SmoothMuscleCell;
