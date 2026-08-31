// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Lapicque engine unit tests

use super::LapicqueNeuron;

fn neuron() -> LapicqueNeuron {
    LapicqueNeuron::new(20.0, 1.0, 1.0, 1.0)
}

#[test]
fn sustained_sc_input_produces_spikes() {
    let mut neuron = neuron();
    let spikes: i32 = (0..200).map(|_| neuron.step(5.0)).sum();
    assert!(spikes > 0);
}

#[test]
fn source_profile_matches_lapicque_closed_form_and_latches_once() {
    let mut neuron = LapicqueNeuron::lapicque_1907();
    let source_voltage = 22.0;
    let beta = neuron.capacitance * neuron.series_resistance * neuron.polarization_resistance
        / (neuron.series_resistance + neuron.polarization_resistance);
    let v_inf = source_voltage * neuron.polarization_resistance
        / (neuron.series_resistance + neuron.polarization_resistance);
    let expected = v_inf * (1.0 - (-neuron.dt / beta).exp());
    assert_eq!(neuron.try_step(source_voltage), Ok(0));
    assert!((neuron.v - expected).abs() < 1e-15);
    let events: i32 = (0..200).map(|_| neuron.step(source_voltage)).sum();
    assert_eq!(events, 1);
    assert!(neuron.excited);
    assert!(neuron.v > neuron.v_threshold);
}

#[test]
fn source_profile_has_no_automatic_reset() {
    let neuron = LapicqueNeuron::lapicque_1907();
    let (trace, events, final_v, excited) = neuron
        .simulate_complete(200, 22.0)
        .expect("source batch must succeed");
    assert_eq!(events.iter().map(|event| i32::from(*event)).sum::<i32>(), 1);
    assert!(excited);
    assert_eq!(trace.last().copied(), Some(final_v));
    assert!(final_v > neuron.v_threshold);
}

#[test]
fn reset_restores_profile_state() {
    let mut source = LapicqueNeuron::lapicque_1907();
    for _ in 0..200 {
        source.step(22.0);
    }
    source.reset();
    assert_eq!((source.v, source.excited), (0.0, false));

    let mut sc = neuron();
    for _ in 0..50 {
        sc.step(5.0);
    }
    sc.reset();
    assert!(sc.v.abs() < 1e-12);
}

#[test]
fn sc_exact_flow_matches_closed_form() {
    let mut neuron = LapicqueNeuron::new(20.0, 1.0, 1.0, 5.0);
    neuron.v = 0.25;
    let drive = 0.5;
    let v_inf = neuron.v_rest + neuron.resistance * drive;
    let expected = v_inf + (neuron.v - v_inf) * (-neuron.dt / neuron.tau).exp();
    assert_eq!(neuron.try_step(drive), Ok(0));
    assert!((neuron.v - expected).abs() < 1e-15);
}

#[test]
fn invalid_state_and_drive_do_not_mutate() {
    let mut neuron = neuron();
    neuron.v = 0.25;
    neuron.tau = 0.0;
    assert!(neuron.try_step(1.0).is_err());
    assert_eq!(neuron.v, 0.25);

    let mut valid = self::neuron();
    assert!(valid.try_step(f64::NAN).is_err());
    assert_eq!(valid.v, 0.0);
}

#[test]
fn batch_failure_is_atomic() {
    let mut neuron = neuron();
    neuron.resistance = f64::MAX;
    neuron.v_threshold = f64::MAX;
    assert!(neuron.simulate_complete(3, f64::MAX).is_err());
    assert_eq!(neuron.v, 0.0);
}

#[test]
fn sustained_sc_pipeline_is_finite() {
    let mut neuron = neuron();
    let spikes: i32 = (0..10_000).map(|_| neuron.step(5.0)).sum();
    assert!(spikes > 100);
    assert!(neuron.v.is_finite());
}
