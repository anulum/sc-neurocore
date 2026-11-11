// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Biophysical Neuron Models

//! Compatibility facade for biophysical and conductance-based neuron models.
//!
//! Each model owns its implementation and tests in a bounded child module
//! while the historical public re-exports and `safe_rate` path remain stable.

mod av_ron_cardiac;
mod bertram_phantom;
mod connor_stevens;
mod de_schutter_purkinje;
mod destexhe_thalamic;
mod durstewitz_dopamine;
mod gif_population;
mod glif;
mod golomb_fs;
mod hill_tononi;
mod hodgkin_huxley;
mod huber_braun;
mod mainen_sejnowski;
mod mihalas_niebur;
mod plant_r15;
mod pospischil;
mod prescott;
mod rate_functions;
mod reexports;
mod sc_four_state_glif;
mod sc_scaled_reset_adaptive_if;
mod sc_six_state_thalamocortical;
mod traub_miles;
mod wang_buzsaki;
mod yamada;
pub use rate_functions::safe_rate;
pub use reexports::*;
