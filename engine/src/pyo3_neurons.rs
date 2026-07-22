// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — PyO3 wrappers for all neuron models

//! PyO3 wrappers and compatibility exports for all neuron models.

use pyo3::prelude::*;

#[macro_use]
#[path = "bindings/default_neuron.rs"]
mod default_neuron_binding;

#[path = "bindings/adex_neuron.rs"]
mod adex_neuron_binding;
#[path = "bindings/alpha.rs"]
mod alpha_binding;
#[path = "bindings/ermentrout_kopell_pop.rs"]
mod ermentrout_kopell_pop_binding;
#[path = "bindings/jansen_rit.rs"]
mod jansen_rit_binding;
#[path = "bindings/lapicque_neuron.rs"]
mod lapicque_neuron_binding;
#[path = "bindings/mcculloch_pitts.rs"]
mod mcculloch_pitts_binding;
#[path = "bindings/sigmoid_rate.rs"]
mod sigmoid_rate_binding;
#[path = "bindings/threshold_linear_rate.rs"]
mod threshold_linear_rate_binding;
#[path = "bindings/wong_wang.rs"]
mod wong_wang_binding;

#[path = "bindings/biophysical/mod.rs"]
mod biophysical_bindings;
#[path = "bindings/hardware/mod.rs"]
mod hardware_bindings;
#[path = "bindings/learning/mod.rs"]
mod learning_bindings;
#[path = "bindings/maps/mod.rs"]
mod map_bindings;
#[path = "bindings/multi_compartment/mod.rs"]
mod multi_compartment_bindings;
#[path = "bindings/simple_spiking/mod.rs"]
mod simple_spiking_bindings;
#[path = "bindings/stochastic/mod.rs"]
mod stochastic_bindings;
#[path = "bindings/trivial/mod.rs"]
mod trivial_bindings;

#[path = "bindings/adaptive_threshold_moe_neuron.rs"]
mod adaptive_threshold_moe_neuron_binding;
#[path = "bindings/arcane_neuron.rs"]
mod arcane_neuron_binding;
#[path = "bindings/attention_gated_neuron.rs"]
mod attention_gated_neuron_binding;
#[path = "bindings/cerebellar/mod.rs"]
mod cerebellar_bindings;
#[path = "bindings/channels/mod.rs"]
mod channel_bindings;
#[path = "bindings/compositional_binding_neuron.rs"]
mod compositional_binding_neuron_binding;
#[path = "bindings/continuous_attractor_neuron.rs"]
mod continuous_attractor_neuron_binding;
#[path = "bindings/differentiable_surrogate_neuron.rs"]
mod differentiable_surrogate_neuron_binding;
#[path = "bindings/hybrid_linear_attention_neuron.rs"]
mod hybrid_linear_attention_neuron_binding;
#[path = "bindings/interneurons/mod.rs"]
mod interneuron_bindings;
#[path = "bindings/meta_plastic_neuron.rs"]
mod meta_plastic_neuron_binding;
#[path = "bindings/misc/mod.rs"]
mod misc_bindings;
#[path = "bindings/motor/mod.rs"]
mod motor_bindings;
#[path = "bindings/multi_timescale_neuron.rs"]
mod multi_timescale_neuron_binding;
#[path = "bindings/population/mod.rs"]
mod population_bindings;
#[path = "bindings/predictive_coding_neuron.rs"]
mod predictive_coding_neuron_binding;
#[path = "bindings/quantum_inspired_lif_neuron.rs"]
mod quantum_inspired_lif_neuron_binding;
#[path = "bindings/rate/mod.rs"]
mod rate_bindings;
#[path = "bindings/self_referential_neuron.rs"]
mod self_referential_neuron_binding;
#[path = "bindings/sensory/mod.rs"]
mod sensory_bindings;
#[path = "bindings/synapses/mod.rs"]
mod synapse_bindings;

pub use biophysical_bindings::*;
pub use cerebellar_bindings::*;
pub use channel_bindings::*;
pub use hardware_bindings::*;
pub use interneuron_bindings::*;
pub use learning_bindings::*;
pub use map_bindings::*;
pub use misc_bindings::*;
pub use motor_bindings::*;
pub use multi_compartment_bindings::*;
pub use population_bindings::*;
pub use rate_bindings::*;
pub use sensory_bindings::*;
pub use simple_spiking_bindings::*;
pub use stochastic_bindings::*;
pub use synapse_bindings::*;
pub use trivial_bindings::*;

pub use adaptive_threshold_moe_neuron_binding::PyAdaptiveThresholdMoENeuron;
pub use adex_neuron_binding::PyAdExNeuron;
pub use alpha_binding::PyAlphaNeuron;
pub use arcane_neuron_binding::PyArcaneNeuron;
pub use attention_gated_neuron_binding::PyAttentionGatedNeuron;
pub use compositional_binding_neuron_binding::PyCompositionalBindingNeuron;
pub use continuous_attractor_neuron_binding::PyContinuousAttractorNeuron;
pub use differentiable_surrogate_neuron_binding::PyDifferentiableSurrogateNeuron;
pub use ermentrout_kopell_pop_binding::PyErmentroutKopellPopulation;
pub use hybrid_linear_attention_neuron_binding::PyHybridLinearAttentionNeuron;
pub use jansen_rit_binding::PyJansenRitUnit;
pub use lapicque_neuron_binding::PyLapicqueNeuron;
pub use mcculloch_pitts_binding::PyMcCullochPittsNeuron;
pub use meta_plastic_neuron_binding::PyMetaPlasticNeuron;
pub use multi_timescale_neuron_binding::PyMultiTimescaleNeuron;
pub use predictive_coding_neuron_binding::PyPredictiveCodingNeuron;
pub use quantum_inspired_lif_neuron_binding::PyQuantumInspiredLIFNeuron;
pub use self_referential_neuron_binding::PySelfReferentialNeuron;
pub use sigmoid_rate_binding::PySigmoidRateNeuron;
pub use threshold_linear_rate_binding::PyThresholdLinearRateNeuron;

// ═══════════════════════════════════════════════════════════════════
// Registration function — call from lib.rs pymodule init
// ═══════════════════════════════════════════════════════════════════

pub fn register_neuron_classes(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // ai_optimized
    multi_timescale_neuron_binding::register(m)?;
    attention_gated_neuron_binding::register(m)?;
    predictive_coding_neuron_binding::register(m)?;
    self_referential_neuron_binding::register(m)?;
    compositional_binding_neuron_binding::register(m)?;
    differentiable_surrogate_neuron_binding::register(m)?;
    continuous_attractor_neuron_binding::register(m)?;
    meta_plastic_neuron_binding::register(m)?;
    // gap models (ai_optimized)
    adaptive_threshold_moe_neuron_binding::register(m)?;
    hybrid_linear_attention_neuron_binding::register(m)?;
    quantum_inspired_lif_neuron_binding::register(m)?;
    trivial_bindings::register(m)?;
    simple_spiking_bindings::register_primary(m)?;
    stochastic_bindings::register_adaptation(m)?;
    population_bindings::register_brunel_wang(m)?;
    alpha_binding::register(m)?;
    simple_spiking_bindings::register_conductance_models(m)?;
    learning_bindings::register(m)?;
    simple_spiking_bindings::register_tail(m)?;
    map_bindings::register(m)?;
    biophysical_bindings::register(m)?;
    multi_compartment_bindings::register(m)?;
    stochastic_bindings::register(m)?;
    population_bindings::register_wilson_cowan_unit(m)?;
    jansen_rit_binding::register(m)?;
    population_bindings::register_wong_wang_unit(m)?;
    wong_wang_binding::register(m)?;
    ermentrout_kopell_pop_binding::register(m)?;
    population_bindings::register_neural_mass_tail(m)?;
    hardware_bindings::register(m)?;
    // rate
    mcculloch_pitts_binding::register(m)?;
    sigmoid_rate_binding::register(m)?;
    threshold_linear_rate_binding::register(m)?;
    rate_bindings::register(m)?;
    arcane_neuron_binding::register(m)?;
    // neuron.rs (legacy)
    adex_neuron_binding::register(m)?;
    crate::exp_if_binding::register_legacy_alias(m)?;
    lapicque_neuron_binding::register(m)?;
    interneuron_bindings::register(m)?;
    motor_bindings::register(m)?;
    sensory_bindings::register(m)?;
    cerebellar_bindings::register(m)?;
    channel_bindings::register(m)?;
    population_bindings::register(m)?;
    misc_bindings::register(m)?;
    synapse_bindings::register(m)?;
    Ok(())
}
