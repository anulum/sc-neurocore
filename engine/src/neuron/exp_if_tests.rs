// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — ExpIF engine unit tests

use super::ExpIfNeuron;

fn rk4_reference(neuron: &ExpIfNeuron, current: f64) -> f64 {
    let rhs = |v: f64| {
        let bounded_v = v.min(neuron.v_threshold);
        let exp_arg = (bounded_v - neuron.v_rh) / neuron.delta_t;
        (-(bounded_v - neuron.v_rest) + neuron.delta_t * exp_arg.exp() + current) / neuron.tau
    };
    let k1 = rhs(neuron.v);
    let k2 = rhs(neuron.v + 0.5 * neuron.dt * k1);
    let k3 = rhs(neuron.v + 0.5 * neuron.dt * k2);
    let k4 = rhs(neuron.v + neuron.dt * k3);
    neuron.v + (neuron.dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
}

#[test]
fn optimised_step_matches_rk4_reference() {
    let mut neuron = ExpIfNeuron::new();
    neuron.v = -60.0;
    let expected = rk4_reference(&neuron, 10.0);
    assert_eq!(neuron.step(10.0), 0);
    assert!((neuron.v - expected).abs() < 1e-12);
}

#[test]
fn strong_input_produces_spikes() {
    let mut neuron = ExpIfNeuron::new();
    let spikes: i32 = (0..2_000).map(|_| neuron.step(500.0)).sum();
    assert!(spikes > 0);
}

#[test]
fn zero_input_remains_silent() {
    let mut neuron = ExpIfNeuron::new();
    let spikes: i32 = (0..500).map(|_| neuron.step(0.0)).sum();
    assert_eq!(spikes, 0);
}

#[test]
fn negative_input_remains_silent() {
    let mut neuron = ExpIfNeuron::new();
    let spikes: i32 = (0..500).map(|_| neuron.step(-100.0)).sum();
    assert_eq!(spikes, 0);
}

#[test]
fn configurable_step_matches_rk4_reference() {
    let mut neuron = ExpIfNeuron::new();
    neuron.v = -60.0;
    neuron.dt = 0.25;
    neuron.tau = 20.0;
    let expected = rk4_reference(&neuron, 12.0);
    assert_eq!(neuron.step(12.0), 0);
    assert!((neuron.v - expected).abs() < 1e-12);
}

#[test]
fn reset_matches_fresh_neuron() {
    let mut neuron = ExpIfNeuron::new();
    for _ in 0..200 {
        neuron.step(500.0);
    }
    neuron.reset();
    let mut fresh = ExpIfNeuron::new();
    let reset_spikes: i32 = (0..100).map(|_| neuron.step(500.0)).sum();
    let fresh_spikes: i32 = (0..100).map(|_| fresh.step(500.0)).sum();
    assert_eq!(reset_spikes, fresh_spikes);
}

#[test]
fn high_input_keeps_voltage_finite() {
    let mut neuron = ExpIfNeuron::new();
    for _ in 0..5_000 {
        neuron.step(1_000.0);
    }
    assert!(neuron.v.is_finite());
}

#[test]
fn enrolled_event_counts_are_stable() {
    for (current, expected) in [(0.0, 0), (5.0, 0), (20.0, 2)] {
        let mut neuron = ExpIfNeuron::new();
        let spikes: i32 = (0..1_000).map(|_| neuron.step(current)).sum();
        assert_eq!(spikes, expected, "current={current}");
    }
}

#[test]
fn refractory_hold_and_invalid_state_fail_closed() {
    let mut neuron = ExpIfNeuron::new();
    neuron.refractory_period = 1.7;
    while neuron.step(50.0) == 0 {}
    assert_eq!(neuron.refractory_remaining, 1.7);
    for _ in 0..10 {
        assert_eq!(neuron.step(50.0), 0);
        assert_eq!(neuron.v, neuron.v_reset);
    }
    assert!((neuron.refractory_remaining - 1.5).abs() < 1.0e-12);

    let voltage = neuron.v;
    neuron.refractory_remaining = 2.0;
    assert_eq!(neuron.step(0.0), 0);
    assert_eq!((neuron.v, neuron.refractory_remaining), (voltage, 2.0));
}

#[test]
fn ten_thousand_steps_complete_within_smoke_limit() {
    let mut neuron = ExpIfNeuron::new();
    let start = std::time::Instant::now();
    for _ in 0..10_000 {
        neuron.step(500.0);
    }
    assert!(start.elapsed().as_millis() < 50);
}

#[test]
fn source_factory_freezes_the_fitted_protocol_boundary() {
    let mut neuron = ExpIfNeuron::fourcaud_trocme_2003();
    assert_eq!(neuron.v_threshold, -30.0);
    assert_eq!(neuron.dt, 0.01);
    assert_eq!(neuron.refractory_period, 1.7);
    assert!(neuron.source_profile);
    neuron.dt = 0.02;
    assert_eq!(neuron.try_step(20.0), Err(super::ExpIfError::InvalidState));
}

#[test]
fn checked_batch_is_aligned_and_failure_atomic() {
    let mut neuron = ExpIfNeuron::new();
    let (voltage, refractory, events) = neuron.simulate_complete(1_000, 20.0).unwrap();
    assert_eq!(voltage.len(), 1_000);
    assert_eq!(refractory.len(), 1_000);
    assert_eq!(events.len(), 1_000);
    assert_eq!(events.iter().map(|event| i32::from(*event)).sum::<i32>(), 2);
    assert_eq!(neuron.v, voltage[999]);

    let before = (neuron.v, neuron.refractory_remaining);
    assert_eq!(
        neuron.simulate_complete(2, f64::INFINITY),
        Err(super::ExpIfError::InvalidInput)
    );
    assert_eq!((neuron.v, neuron.refractory_remaining), before);
}
