// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Homeostatic LIF engine unit tests

use super::HomeostaticLif;

#[test]
fn strong_input_produces_spikes() {
    let mut neuron = HomeostaticLif::with_defaults();
    let spikes: i32 = (0..200).map(|_| neuron.step(25.0)).sum();
    assert!(spikes > 0, "must fire with strong input");
}

#[test]
fn repeated_spiking_adapts_threshold() {
    let mut neuron = HomeostaticLif::with_defaults();
    let initial = neuron.v_threshold;
    for _ in 0..500 {
        neuron.step(25.0);
    }
    assert!(
        (neuron.v_threshold - initial).abs() > 1e-6,
        "threshold must adapt"
    );
}

#[test]
fn zero_input_remains_silent() {
    let mut neuron = HomeostaticLif::with_defaults();
    let spikes: i32 = (0..100).map(|_| neuron.step(0.0)).sum();
    assert_eq!(spikes, 0);
}

#[test]
fn adaptive_threshold_remains_bounded() {
    let mut neuron = HomeostaticLif::with_defaults();
    for _ in 0..10_000 {
        neuron.step(50.0);
    }
    assert!((0.1..=10.0).contains(&neuron.v_threshold));
}
