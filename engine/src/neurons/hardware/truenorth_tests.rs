// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — TrueNorth engine unit tests

use super::*;

#[test]
fn truenorth_fires() {
    let mut n = TrueNorthNeuron::default();
    let t: i32 = (0..10).map(|_| n.step(50)).sum();
    assert!(t > 0);
}
#[test]
fn truenorth_silent() {
    let mut n = TrueNorthNeuron::default();
    let t: i32 = (0..100).map(|_| n.step(0)).sum();
    assert_eq!(t, 0);
}
#[test]
fn truenorth_reset() {
    let mut n = TrueNorthNeuron::default();
    for _ in 0..10 {
        n.step(50);
    }
    n.reset();
    assert_eq!(n.v, 0);
}
