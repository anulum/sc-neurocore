// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Network runner input-adapter public contract

use sc_neurocore_engine::network_runner::{
    WrAlpha, WrCOBALIF, WrLoihiCUBA, WrMcCullochPitts, WrSigmoidRate,
};
use sc_neurocore_engine::neurons::{
    AlphaNeuron, COBALIFNeuron, LoihiCUBANeuron, SigmoidRateNeuron,
};

#[test]
fn additional_input_adapters_preserve_public_default_semantics() {
    let mut alpha_adapter = WrAlpha::new();
    let mut alpha_reference = AlphaNeuron::new();
    assert_eq!(alpha_adapter.step(2.0), alpha_reference.step(2.0, 0.0));
    assert_eq!(alpha_adapter.v(), alpha_reference.v);

    let mut coba_adapter = WrCOBALIF::new();
    let mut coba_reference = COBALIFNeuron::new();
    assert_eq!(coba_adapter.step(2.0), coba_reference.step(2.0, 0.0, 0.0));
    assert_eq!(coba_adapter.v(), coba_reference.v);
}

#[test]
fn integer_input_adapter_preserves_public_float_conversion() {
    let mut adapter = WrLoihiCUBA::new();
    let mut reference = LoihiCUBANeuron::new();
    assert_eq!(adapter.step(3.9), reference.step(3));
    assert_eq!(adapter.v(), f64::from(reference.v));
}

#[test]
fn graded_adapter_preserves_public_threshold_contract() {
    let mut adapter = WrSigmoidRate::new();
    let mut reference = SigmoidRateNeuron::new();
    let output = reference.step(0.25);
    assert_eq!(adapter.step(0.25), i32::from(output > 0.5));
    assert_eq!(adapter.v(), reference.r);
}

#[test]
fn logical_adapter_preserves_public_signed_transport() {
    let mut adapter = WrMcCullochPitts::new();
    assert_eq!(adapter.step(0.0), 0);
    assert_eq!(adapter.step(1.0), 1);
    assert_eq!(adapter.step(-1.0), 0);
    assert_eq!(adapter.step(1.5), 0);
    assert_eq!(adapter.step(f64::NAN), 0);
    adapter.reset();
    assert_eq!(adapter.v(), 0.0);
}
