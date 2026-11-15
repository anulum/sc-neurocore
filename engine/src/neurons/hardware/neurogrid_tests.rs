// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Neurogrid engine unit tests

use super::*;

#[test]
fn neurogrid_fires() {
    let mut n = NeuroGridNeuron::new();
    let t: i32 = (0..2000).map(|_| n.step(500.0)).sum();
    assert!(t > 0);
}
#[test]
fn neurogrid_silent() {
    let mut n = NeuroGridNeuron::new();
    let t: i32 = (0..200).map(|_| n.step(0.0)).sum();
    assert_eq!(t, 0);
}
#[test]
fn neurogrid_reset() {
    let mut n = NeuroGridNeuron::new();
    for _ in 0..100 {
        n.step(500.0);
    }
    n.reset();
    assert!((n.v_s - (-65.0)).abs() < 1e-10);
}
#[test]
fn neurogrid_bounded() {
    let mut n = NeuroGridNeuron::new();
    for _ in 0..2000 {
        n.step(1e4);
    }
    assert!(n.v_s.is_finite());
}
#[test]
fn neurogrid_nan_no_panic() {
    NeuroGridNeuron::new().step(f64::NAN);
}
#[test]
fn neurogrid_rk4_anchor() {
    let mut n = NeuroGridNeuron::new();
    let spikes: i32 = (0..20_000).map(|_| n.step(100.0)).sum();
    assert_eq!(spikes, 94);
    assert!(n.v_s.is_finite());
    assert!(n.v_d.is_finite());
}
#[test]
fn neurogrid_invalid_input_preserves_state() {
    let mut n = NeuroGridNeuron::new();
    for _ in 0..10 {
        n.step(100.0);
    }
    let old = (n.v_s, n.v_d);
    assert_eq!(n.step(f64::INFINITY), 0);
    assert_eq!((n.v_s, n.v_d), old);
}
