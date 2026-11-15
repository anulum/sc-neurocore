// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — SpiNNaker 2 engine unit tests

use super::*;

#[test]
fn spinnaker2_fires() {
    let mut n = SpiNNaker2Neuron::new();
    let t: i32 = (0..200).map(|_| n.step(100)).sum();
    assert!(t > 0);
}
#[test]
fn spinnaker2_silent() {
    let mut n = SpiNNaker2Neuron::new();
    let t: i32 = (0..200).map(|_| n.step(0)).sum();
    assert_eq!(t, 0);
}
#[test]
fn spinnaker2_reset() {
    let mut n = SpiNNaker2Neuron::new();
    for _ in 0..50 {
        n.step(100);
    }
    n.reset();
}
#[test]
fn spinnaker2_bounded() {
    let mut n = SpiNNaker2Neuron::new();
    for _ in 0..1000 {
        n.step(10000);
    }
}
