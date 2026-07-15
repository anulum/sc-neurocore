// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Cerebellar Circuit Neuron Models

//! Compatibility facade for cerebellar circuit neuron models.
//!
//! Each model owns its implementation and tests in a bounded child module
//! while the historical public re-exports remain unchanged.

mod dcn;
mod golgi;
mod granule;
mod lugaro;
mod stellate;
mod unipolar_brush;

pub use dcn::DCNNeuron;
pub use golgi::GolgiCell;
pub use granule::GranuleCell;
pub use lugaro::LugaroCell;
pub use stellate::StellateCell;
pub use unipolar_brush::UnipolarBrushCell;
