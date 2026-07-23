// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Network runner: high-performance Rust simulation backend

//! High-performance network simulation backend.
//!
//! Replaces the Python per-neuron loop with Rayon-parallel Rust execution
//! over CSR-stored projections and heterogeneous neuron populations.

mod input_adapters;
pub use input_adapters::*;

mod neuron_variant;
pub use neuron_variant::NeuronVariant;

mod population_runner;
pub use population_runner::PopulationRunner;

mod projection_runner;
pub use projection_runner::ProjectionRunner;

mod simulation_results;
pub use simulation_results::SimResults;

mod network_execution;
pub use network_execution::NetworkRunner;

mod model_factory;
pub use model_factory::{create_neuron, create_population, supported_models};
