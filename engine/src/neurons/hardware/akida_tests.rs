// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Akida engine unit tests

use super::*;

#[test]
fn akida_fires() {
    let mut n = AkidaNeuron::default();
    let t: i32 = (0..10).map(|_| n.step(50.0)).sum();
    assert!(t > 0);
}
#[test]
fn akida_silent() {
    let mut n = AkidaNeuron::default();
    let t: i32 = (0..100).map(|_| n.step(0.0)).sum();
    assert_eq!(t, 0);
}
#[test]
fn akida_reset() {
    let mut n = AkidaNeuron::default();
    for _ in 0..10 {
        n.step(50.0);
    }
    n.reset();
}
#[test]
fn akida_nan_no_panic() {
    AkidaNeuron::default().step(f64::NAN);
}
