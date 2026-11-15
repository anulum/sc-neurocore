// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — BrainScaleS AdEx engine unit tests

use super::*;

#[test]
fn brainscales_fires() {
    let mut n = BrainScaleSAdExNeuron::new();
    let t: i32 = (0..2000).map(|_| n.step(500.0)).sum();
    assert!(t > 0);
}
#[test]
fn brainscales_silent() {
    let mut n = BrainScaleSAdExNeuron::new();
    let t: i32 = (0..200).map(|_| n.step(0.0)).sum();
    assert_eq!(t, 0);
}
#[test]
fn brainscales_reset() {
    let mut n = BrainScaleSAdExNeuron::new();
    for _ in 0..100 {
        n.step(500.0);
    }
    n.reset();
    assert!((n.v - n.v_rest).abs() < 1e-10);
}
#[test]
fn brainscales_bounded() {
    let mut n = BrainScaleSAdExNeuron::new();
    for _ in 0..2000 {
        n.step(1e4);
    }
    assert!(n.v.is_finite());
}
#[test]
fn brainscales_nan_no_panic() {
    BrainScaleSAdExNeuron::new().step(f64::NAN);
}
