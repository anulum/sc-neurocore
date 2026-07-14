// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Spike train analysis (Rust engine)

pub mod basic;
pub(crate) mod bindings;
pub mod causality;
pub mod correlation;
pub mod decoding;
pub mod dimensionality;
pub mod distance;
pub mod gpfa;
pub mod information;
pub mod lfp;
pub mod network;
pub mod neural_decoders;
pub mod patterns;
pub mod point_process;
pub mod rate;
pub mod sorting_quality;
pub mod spade;
pub mod spectral;
pub mod statistics;
pub mod stimulus;
pub mod surrogates;
pub mod temporal;
pub mod variability;
pub mod waveform;
