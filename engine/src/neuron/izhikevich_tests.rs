// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Izhikevich neuron engine unit tests

use super::Izhikevich;

#[test]
fn regular_spiking_preset_fires_with_current() {
    let mut neuron = Izhikevich::regular_spiking();
    let spikes: i32 = (0..100).map(|_| neuron.step(10.0)).sum();
    assert!(spikes > 0, "RS neuron must fire with I=10");
}

#[test]
fn regular_spiking_preset_is_silent_without_input() {
    let mut neuron = Izhikevich::regular_spiking();
    let spikes: i32 = (0..100).map(|_| neuron.step(0.0)).sum();
    assert_eq!(spikes, 0, "no spikes without input");
}

#[test]
fn reset_restores_initial_state() {
    let mut neuron = Izhikevich::regular_spiking();
    for _ in 0..50 {
        neuron.step(10.0);
    }
    neuron.reset();
    assert_eq!(neuron.v, neuron.c);
    assert!((neuron.u - neuron.b * neuron.c).abs() < 1e-12);
}

#[test]
fn chattering_preset_fires_more_than_regular_spiking() {
    let mut chattering = Izhikevich::new(0.02, 0.2, -50.0, 2.0, 1.0);
    let mut regular_spiking = Izhikevich::regular_spiking();
    let mut chattering_spikes = 0;
    let mut regular_spikes = 0;
    for _ in 0..200 {
        chattering_spikes += chattering.step(10.0);
        regular_spikes += regular_spiking.step(10.0);
    }
    assert!(
        chattering_spikes > regular_spikes,
        "chattering ({chattering_spikes}) should fire more than RS ({regular_spikes})"
    );
}
