// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Two-dimensional and higher spiking neuron models

//! Compatibility facade for two-dimensional and higher spiking models.

pub mod alpha;
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
mod reexports;
pub mod resonate_and_fire;
mod sc_stochastic_rate_adaptation;
mod sc_triangular_mckean;
mod sherman_rinzel_keizer;
mod superspike_neuron;
mod terman_wang;
mod wilson_hr;
pub use reexports::*;
