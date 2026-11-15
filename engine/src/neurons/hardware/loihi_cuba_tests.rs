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
