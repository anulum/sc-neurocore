// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Loihi CUBA engine unit tests

use super::*;

#[test]
fn loihi_cuba_fires() {
    let mut n = LoihiCUBANeuron::new();
    let t: i32 = (0..200).map(|_| n.step(100)).sum();
    assert!(t > 0);
}
#[test]
fn loihi_cuba_silent() {
    let mut n = LoihiCUBANeuron::new();
    let t: i32 = (0..200).map(|_| n.step(0)).sum();
    assert_eq!(t, 0);
}
#[test]
fn loihi_cuba_reset() {
    let mut n = LoihiCUBANeuron::new();
    for _ in 0..50 {
        n.step(100);
    }
    n.reset();
    assert_eq!(n.v, 0);
    assert_eq!(n.u, 0);
}
#[test]
fn loihi_cuba_bounded() {
    let mut n = LoihiCUBANeuron::new();
    for _ in 0..1000 {
        n.step(10000);
    }
}
#[test]
fn loihi_cuba_decay_truncates_towards_zero() {
    let mut n = LoihiCUBANeuron {
        u: -7,
        ..LoihiCUBANeuron::new()
    };
    n.step(0);
    assert_eq!((n.v, n.u), (-6, -6));
}
#[test]
fn loihi_cuba_negated_drive_gives_negated_state() {
    let drive = [37, -12, 5, 0, -40, 23, 8, -3, 1, -1];
    let (mut positive, mut negative) = (LoihiCUBANeuron::new(), LoihiCUBANeuron::new());
    for _ in 0..4 {
        for &value in &drive {
            assert_eq!(positive.step(value) + negative.step(-value), 0);
            assert_eq!((negative.v, negative.u), (-positive.v, -positive.u));
        }
    }
}
