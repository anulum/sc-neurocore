// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Loihi 2 engine unit tests

use super::*;

#[test]
fn loihi2_fires() {
    let mut n = Loihi2Neuron {
        tau3: 8,
        ..Loihi2Neuron::new()
    };
    let t: i32 = (0..500).map(|_| n.step(200)).sum();
    assert!(t > 0);
}
#[test]
fn loihi2_silent() {
    let mut n = Loihi2Neuron::new();
    let t: i32 = (0..200).map(|_| n.step(0)).sum();
    assert_eq!(t, 0);
}
#[test]
fn loihi2_reset() {
    let mut n = Loihi2Neuron::new();
    for _ in 0..50 {
        n.step(200);
    }
    n.reset();
    assert_eq!(n.s1, 0);
}
#[test]
fn loihi2_bounded() {
    let mut n = Loihi2Neuron {
        tau3: 8,
        ..Loihi2Neuron::new()
    };
    for _ in 0..1000 {
        n.step(10000);
    }
}
