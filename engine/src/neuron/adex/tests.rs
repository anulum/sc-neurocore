// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — AdEx scalar regression tests

use super::AdExNeuron;

#[test]
fn strong_input_produces_spikes() {
    let mut neuron = AdExNeuron::new();
    let spikes: i32 = (0..2_000).map(|_| neuron.step(500.0)).sum();
    assert!(spikes > 0, "AdEx must fire with strong input");
}

#[test]
fn adaptation_does_not_increase_late_rate() {
    let mut neuron = AdExNeuron::new();
    let first: i32 = (0..1_000).map(|_| neuron.step(400.0)).sum();
    let second: i32 = (0..1_000).map(|_| neuron.step(400.0)).sum();
    assert!(second <= first + 5, "first={first}, second={second}");
}

#[test]
fn matches_python_golden_spike_counts() {
    for (current, expected) in [(0.0, 0), (200.0, 4), (500.0, 12)] {
        let mut neuron = AdExNeuron::new();
        let spikes: i32 = (0..1_000).map(|_| neuron.step(current)).sum();
        assert_eq!(spikes, expected, "current={current}");
    }
}

#[test]
fn invalid_input_is_mutation_free() {
    let mut neuron = AdExNeuron::new();
    let before = (neuron.v, neuron.w);
    assert!(neuron.try_step(f64::INFINITY).is_err());
    assert_eq!((neuron.v, neuron.w), before);
}

#[test]
fn nonfinite_candidate_is_mutation_free() {
    let mut neuron = AdExNeuron::new();
    neuron.dt = 1.0e308;
    let before = (neuron.v, neuron.w);
    assert!(neuron.try_step(1.0e308).is_err());
    assert_eq!((neuron.v, neuron.w), before);
}

#[test]
fn no_input_remains_silent() {
    let mut neuron = AdExNeuron::new();
    let spikes: i32 = (0..1_000).map(|_| neuron.step(0.0)).sum();
    assert_eq!(spikes, 0);
}

#[test]
fn negative_current_remains_silent() {
    let mut neuron = AdExNeuron::new();
    let spikes: i32 = (0..500).map(|_| neuron.step(-100.0)).sum();
    assert_eq!(spikes, 0);
}

#[test]
fn reset_matches_fresh_neuron() {
    let mut neuron = AdExNeuron::new();
    for _ in 0..200 {
        neuron.step(500.0);
    }
    assert!(neuron.w > 0.0);
    neuron.reset();
    assert_eq!(neuron.v, neuron.v_rest);
    assert_eq!(neuron.w, 0.0);

    let mut fresh = AdExNeuron::new();
    let reset_spikes: i32 = (0..100).map(|_| neuron.step(500.0)).sum();
    let fresh_spikes: i32 = (0..100).map(|_| fresh.step(500.0)).sum();
    assert_eq!(reset_spikes, fresh_spikes);
}

#[test]
fn sustained_high_input_keeps_state_finite() {
    let mut neuron = AdExNeuron::new();
    for _ in 0..5_000 {
        neuron.step(1_000.0);
    }
    assert!(neuron.v.is_finite());
    assert!(neuron.w.is_finite());
}

#[test]
fn sustained_input_produces_many_spikes() {
    let mut neuron = AdExNeuron::new();
    let spikes: i32 = (0..10_000).map(|_| neuron.step(500.0)).sum();
    assert!(spikes > 100, "got {spikes}");
    assert!(neuron.v.is_finite());
}

#[test]
fn ten_thousand_steps_complete_within_smoke_limit() {
    let mut neuron = AdExNeuron::new();
    let start = std::time::Instant::now();
    for _ in 0..10_000 {
        neuron.step(500.0);
    }
    assert!(start.elapsed().as_millis() < 50);
}
