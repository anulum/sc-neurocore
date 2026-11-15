// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — SpiNNaker LIF engine unit tests

use super::*;

#[test]
fn spinnaker_fires() {
    let mut n = SpiNNakerLIFNeuron::new();
    let t: i32 = (0..200).map(|_| n.step(30.0)).sum();
    assert!(t > 0);
}
#[test]
fn spinnaker_silent() {
    let mut n = SpiNNakerLIFNeuron::new();
    let t: i32 = (0..200).map(|_| n.step(0.0)).sum();
    assert_eq!(t, 0);
}
#[test]
fn spinnaker_reset() {
    let mut n = SpiNNakerLIFNeuron::new();
    for _ in 0..50 {
        n.step(30.0);
    }
    n.reset();
    assert!((n.v - n.v_rest).abs() < 1e-10);
}
#[test]
fn spinnaker_bounded() {
    let mut n = SpiNNakerLIFNeuron::new();
    for _ in 0..1000 {
        n.step(1e4);
    }
    assert!(n.v.is_finite());
}
#[test]
fn spinnaker_nan_no_panic() {
    SpiNNakerLIFNeuron::new().step(f64::NAN);
}
