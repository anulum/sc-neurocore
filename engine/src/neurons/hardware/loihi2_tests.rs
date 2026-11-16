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
#[test]
fn loihi2_decay_truncates_towards_zero() {
    let mut n = Loihi2Neuron {
        s1: -7,
        s2: -7,
        s3: -23,
        ..Loihi2Neuron::new()
    };
    n.step(0);
    assert_eq!((n.s1, n.s2, n.s3), (-7 - 6, -6, -23));
}
#[test]
fn loihi2_negated_drive_gives_negated_state() {
    let drive = [41, -17, 6, 0, -33, 25, 9, -4, 2, -2];
    let mut positive = Loihi2Neuron {
        w13: 1,
        s3: -120,
        ..Loihi2Neuron::new()
    };
    let mut negative = Loihi2Neuron {
        w13: 1,
        s3: 120,
        ..Loihi2Neuron::new()
    };
    for _ in 0..4 {
        for &value in &drive {
            assert_eq!(positive.step(value) + negative.step(-value), 0);
            let mirrored = (-positive.s1, -positive.s2, -positive.s3);
            assert_eq!((negative.s1, negative.s2, negative.s3), mirrored);
        }
    }
}
