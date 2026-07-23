// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Neuron Models

//! Stable composition surface for the v3 engine's foundational neuron models.

mod adex;
mod bitstream_averager;
mod dendritic_neuron;
mod exp_if;
mod fixed_point_lif;
mod homeostatic_lif;
mod izhikevich;
mod lapicque;

pub use adex::AdExNeuron;
pub use bitstream_averager::BitstreamAverager;
pub use dendritic_neuron::DendriticNeuron;
pub use exp_if::ExpIfNeuron;
pub use fixed_point_lif::{mask, FixedPointLif};
pub use homeostatic_lif::HomeostaticLif;
pub use izhikevich::Izhikevich;
pub use lapicque::LapicqueNeuron;
