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
#[test]
fn akida_zero_weight_is_not_an_event() {
    let mut n = AkidaNeuron::default();
    for weight in [3.0, 0.0, 3.0, 3.0] {
        n.step(weight);
    }
    assert_eq!((n.v, n.rank), (3 + 2 + 1, 3));
}
#[test]
fn akida_keeps_integrating_after_its_spike() {
    let mut n = AkidaNeuron::new(5);
    assert_eq!(n.step(10.0), 1);
    assert_eq!(n.step(10.0), 0);
    assert_eq!(n.step(10.0), 0);
    assert_eq!((n.v, n.rank), (10 + 7 + 5, 3));
}
#[test]
fn akida_weights_each_event_by_the_running_product() {
    // The Python model multiplies the factor by `modulation` once per event. For 0.95 that product after four events
    // lies one unit in the last place below `powi`'s square-and-multiply, and 160000 times it truncates to 130320
    // where `powi` gives 130321.
    let mut n = AkidaNeuron {
        modulation: 0.95,
        ..AkidaNeuron::default()
    };
    for weight in [1.0, 1.0, 1.0, 1.0, 160_000.0] {
        n.step(weight);
    }
    assert_eq!((n.v, n.rank), (1 + 130_320, 5));
}
#[test]
fn akida_reset_restarts_the_factor() {
    let mut n = AkidaNeuron::new(1_000_000);
    for _ in 0..5 {
        n.step(40.0);
    }
    n.reset();
    n.step(40.0);
    assert_eq!((n.v, n.rank, n.spiked), (40, 1, false));
}
