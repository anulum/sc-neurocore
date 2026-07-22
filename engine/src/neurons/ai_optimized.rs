// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — AI-optimised neuron compatibility facade

//! Compatibility facade for bounded AI-optimised neuron modules.

mod adaptive_threshold_moe_neuron;
mod arcane_neuron;
mod attention_gated_neuron;
mod compositional_binding_neuron;
mod continuous_attractor_neuron;
mod differentiable_surrogate_neuron;
mod hybrid_linear_attention_neuron;
mod meta_plastic_neuron;
mod multi_timescale_neuron;
mod predictive_coding_neuron;
mod quantum_inspired_lif_neuron;
mod self_referential_neuron;

pub use adaptive_threshold_moe_neuron::AdaptiveThresholdMoENeuron;
pub use arcane_neuron::ArcaneNeuron;
pub use attention_gated_neuron::AttentionGatedNeuron;
pub use compositional_binding_neuron::CompositionalBindingNeuron;
pub use continuous_attractor_neuron::ContinuousAttractorNeuron;
pub use differentiable_surrogate_neuron::DifferentiableSurrogateNeuron;
pub use hybrid_linear_attention_neuron::HybridLinearAttentionNeuron;
pub use meta_plastic_neuron::MetaPlasticNeuron;
pub use multi_timescale_neuron::MultiTimescaleNeuron;
pub use predictive_coding_neuron::PredictiveCodingNeuron;
pub use quantum_inspired_lif_neuron::QuantumInspiredLIFNeuron;
pub use self_referential_neuron::SelfReferentialNeuron;
