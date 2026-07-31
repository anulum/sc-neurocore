// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Simple integrate-and-fire variants

//! Compatibility facade for simple integrate-and-fire variants.

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
mod sc_non_resetting_adaptive_lif;
mod sc_normalized_energy_lif;
mod sc_resetting_mat;
mod sc_sigma_delta_accumulator;
mod sfa;
mod sigma_delta;
mod stochastic_lif;
mod theta;
mod reexports;
pub use reexports::*;
