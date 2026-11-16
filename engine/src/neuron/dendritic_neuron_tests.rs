// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Dendritic neuron engine unit tests

use super::DendriticNeuron;

#[test]
fn xor_nonlinearity_matches_truth_table() {
    let mut neuron = DendriticNeuron::new(0.5);
    assert_eq!(neuron.step(0.0, 0.0), 0);
    assert_eq!(neuron.step(1.0, 0.0), 1);
    assert_eq!(neuron.step(0.0, 1.0), 1);
    assert_eq!(neuron.step(1.0, 1.0), 0);
}

#[test]
fn subthreshold_current_does_not_fire() {
    let mut neuron = DendriticNeuron::new(0.5);
    assert_eq!(neuron.step(0.2, 0.1), 0);
}

#[test]
fn reset_clears_last_current() {
    let mut neuron = DendriticNeuron::with_defaults();
    neuron.step(1.0, 0.0);
    neuron.reset();
    assert!(neuron.last_current.abs() < 1e-12);
}
