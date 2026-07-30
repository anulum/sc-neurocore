// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Simple integrate-and-fire variants

//! Compatibility facade for simple integrate-and-fire variants.
//!
//! Nineteen independent model implementations and their inherited tests
//! live in bounded child modules while historical public re-exports remain
//! unchanged.

pub mod adaptive_threshold_if;
mod closed_form_continuous;
mod complementary_lif;
mod energy_lif;
mod escape_rate;
mod gated_lif;
mod inhibitory_lif;
mod integer_qif;
mod klif;
mod mat;
mod non_resetting_lif;
mod nonlinear_lif;
mod parametric_lif;
mod perfect_integrator;
mod quadratic_if;
mod sc_resetting_mat;
mod sfa;
mod sigma_delta;
mod stochastic_lif;
mod theta;

pub use adaptive_threshold_if::AdaptiveThresholdIFNeuron;
pub use closed_form_continuous::ClosedFormContinuousNeuron;
pub use complementary_lif::ComplementaryLIFNeuron;
pub use energy_lif::EnergyLIFNeuron;
pub use escape_rate::EscapeRateNeuron;
pub use gated_lif::GatedLIFNeuron;
pub use inhibitory_lif::InhibitoryLIFNeuron;
pub use integer_qif::IntegerQIFNeuron;
pub use klif::KLIFNeuron;
pub use mat::MATNeuron;
pub use non_resetting_lif::NonResettingLIFNeuron;
pub use nonlinear_lif::NonlinearLIFNeuron;
pub use parametric_lif::ParametricLIFNeuron;
pub use perfect_integrator::PerfectIntegratorNeuron;
pub use quadratic_if::QuadraticIFNeuron;
pub use sc_resetting_mat::SCResettingMATNeuron;
pub use sfa::SFANeuron;
pub use sigma_delta::SigmaDeltaNeuron;
pub use stochastic_lif::StochasticLIFNeuron;
pub use theta::ThetaNeuron;
