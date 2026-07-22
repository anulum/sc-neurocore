// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Engine Crate Root

#![allow(
    clippy::useless_conversion,
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    deprecated
)]

use pyo3::prelude::*;

pub mod adc_to_spike;
#[path = "bindings/adc_to_spike.rs"]
mod adc_to_spike_binding;
pub mod analysis;
pub mod attention;
pub mod bitstream;
#[path = "bindings/bitstream.rs"]
mod bitstream_binding;
#[path = "bindings/brunel.rs"]
mod brunel_binding;
#[path = "bindings/escape_rate.rs"]
mod escape_rate_binding;
#[path = "bindings/evolution.rs"]
mod evolution_binding;
#[path = "bindings/exp_if.rs"]
mod exp_if_binding;
#[path = "bindings/hdc.rs"]
mod hdc_binding;
pub use hdc_binding::PyBitStreamTensor;
pub mod brunel;
#[path = "bindings/coba_lif.rs"]
mod coba_lif_binding;
pub mod connectome;
pub mod conv;
pub mod cordiv;
#[path = "bindings/cordiv.rs"]
mod cordiv_binding;
pub mod cortical_column;
#[path = "bindings/cortical_column.rs"]
mod cortical_column_binding;
pub mod cortical_inject;
#[path = "bindings/cortical_inject.rs"]
mod cortical_inject_binding;
#[path = "bindings/dcls.rs"]
mod dcls_binding;
#[path = "bindings/dense_layer.rs"]
mod dense_layer_binding;
pub mod dna;
pub mod ei_network;
#[path = "bindings/ei_network.rs"]
mod ei_network_binding;
pub mod encoder;
pub mod evo;
pub mod fault;
#[path = "bindings/fault.rs"]
mod fault_binding;
#[path = "bindings/fixed_point_lif.rs"]
mod fixed_point_lif_binding;
pub mod fusion;
#[cfg(feature = "gpu")]
pub mod gpu;
pub mod grad;
pub mod graph;
#[path = "bindings/iqif.rs"]
mod iqif_binding;
pub mod ir;
#[path = "bindings/izhikevich2007.rs"]
mod izhikevich2007_binding;
#[path = "bindings/izhikevich.rs"]
mod izhikevich_binding;
#[path = "bindings/kuramoto.rs"]
mod kuramoto_binding;
pub mod layer;
pub(crate) mod learning_bindings;
pub mod lgssm;
#[path = "bindings/lgssm.rs"]
mod lgssm_binding;
#[path = "bindings/matrix_inputs.rs"]
mod matrix_inputs_binding;
#[path = "bindings/mixed_dense.rs"]
mod mixed_dense_binding;
pub mod network_runner;
#[path = "bindings/network_runner.rs"]
mod network_runner_binding;
pub mod neuron;
pub mod neurons;
#[path = "bindings/ollivier_ricci.rs"]
mod ollivier_ricci_binding;
pub mod optimizer;
#[path = "bindings/optimizer.rs"]
mod optimizer_binding;
pub mod partition;
#[path = "bindings/partition.rs"]
mod partition_binding;
pub mod phi;
#[path = "bindings/phi.rs"]
mod phi_binding;
pub mod photonic;
pub mod ping;
#[path = "bindings/ping.rs"]
mod ping_binding;
#[path = "bindings/poisson.rs"]
mod poisson_binding;
pub mod predictive_coding;
#[path = "bindings/predictive_coding.rs"]
mod predictive_coding_binding;
pub mod pyo3_neurons;
pub mod quantum;
pub mod rall_dendrite;
#[path = "bindings/rall_dendrite.rs"]
mod rall_dendrite_binding;
pub mod recorder;
pub mod recurrent;
pub mod rk4_neurons;
#[path = "bindings/runtime_control.rs"]
mod runtime_control_binding;
pub mod sc_inference;
#[path = "bindings/sc_inference.rs"]
mod sc_inference_binding;
pub mod scpn;
#[path = "bindings/scpn_metrics.rs"]
mod scpn_metrics_binding;
pub mod simd;
pub mod sobol;
#[path = "bindings/stdp_synapse.rs"]
mod stdp_synapse_binding;
#[cfg(feature = "z3")]
pub mod supervisor;
pub mod synapses;
pub mod topology;
pub mod wilson_cowan;
#[path = "bindings/wilson_cowan.rs"]
mod wilson_cowan_binding;
pub mod wong_wang;

/// SC-NeuroCore ─ High-Performance Rust Engine

#[pymodule]
fn sc_neurocore_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    runtime_control_binding::register(m)?;
    bitstream_binding::register(m)?;
    fixed_point_lif_binding::register(m)?;
    dcls_binding::register(m)?;
    mixed_dense_binding::register(m)?;
    adc_to_spike_binding::register(m)?;
    sc_inference_binding::register(m)?;
    wilson_cowan_binding::register(m)?;
    dense_layer_binding::register(m)?;
    #[cfg(feature = "gpu")]
    m.add_class::<gpu::PyGpuDenseLayer>()?;
    #[cfg(feature = "gpu")]
    m.add_class::<gpu::PyGpuLifBatch>()?;
    #[cfg(feature = "gpu")]
    m.add_class::<gpu::PyGpuKuramoto>()?;
    #[cfg(feature = "gpu")]
    m.add_class::<gpu::PyGpuIzhikevichBatch>()?;
    stdp_synapse_binding::register(m)?;
    learning_bindings::register(m)?;
    kuramoto_binding::register(m)?;
    scpn_metrics_binding::register(m)?;
    hdc_binding::register(m)?;
    brunel_binding::register(m)?;
    izhikevich_binding::register(m)?;
    ir::bindings::register(m)?;
    exp_if_binding::register(m)?;
    pyo3_neurons::register_neuron_classes(m)?;
    network_runner_binding::register(m)?;
    #[cfg(feature = "z3")]
    m.add_class::<supervisor::PySpikingControllerPool>()?;
    ei_network_binding::register(m)?;
    m.add_function(wrap_pyfunction!(rk4_neurons::py_rk4_neuron_simulate, m)?)?;
    cordiv_binding::register(m)?;
    predictive_coding_binding::register(m)?;
    phi_binding::register(m)?;
    cortical_column_binding::register(m)?;
    rall_dendrite_binding::register(m)?;
    analysis::bindings::register(m)?;
    dna::bindings::register(m)?;
    quantum::bindings::register(m)?;
    // Photonic NoC acceleration
    photonic::bindings::register(m)?;
    optimizer_binding::register(m)?;
    evolution_binding::register(m)?;
    lgssm_binding::register(m)?;
    ollivier_ricci_binding::register(m)?;
    coba_lif_binding::register(m)?;
    escape_rate_binding::register(m)?;
    poisson_binding::register(m)?;
    iqif_binding::register(m)?;
    izhikevich2007_binding::register(m)?;
    fault_binding::register(m)?;
    partition_binding::register(m)?;
    ping_binding::register(m)?;
    cortical_inject_binding::register(m)?;
    Ok(())
}
