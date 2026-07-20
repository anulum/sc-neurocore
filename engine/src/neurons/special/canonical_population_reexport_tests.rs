// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — canonical population re-export tests

//! Behavioural contracts for population models historically re-exported here.

use super::{ErmentroutKopellPopulation, WongWangUnit};

#[test]
fn wong_wang_populations_diverge() {
    let mut n = WongWangUnit::new(42);
    for _ in 0..5000 {
        n.step(0.02, 0.0).unwrap();
    }
    assert!((n.s1 - n.s2).abs() > 0.001);
}

#[test]
fn wong_wang_reset_restores_initial_state() {
    let mut n = WongWangUnit::new(42);
    for _ in 0..1000 {
        n.step(0.02, 0.0).unwrap();
    }
    n.reset();
    assert!((n.s1 - 0.1).abs() < 1e-10);
    assert!((n.s2 - 0.1).abs() < 1e-10);
}

#[test]
fn wong_wang_state_remains_finite() {
    let mut n = WongWangUnit::new(42);
    for _ in 0..5000 {
        n.step(1.0, 0.0).unwrap();
    }
    assert!(n.s1.is_finite());
    assert!(n.s2.is_finite());
}

#[test]
fn wong_wang_nan_fails_closed() {
    assert!(WongWangUnit::new(42).step(f64::NAN, 0.0).is_err());
}

#[test]
fn ermentrout_kopell_population_fires() {
    let mut n = ErmentroutKopellPopulation::new();
    for _ in 0..1000 {
        n.step(0.0);
    }
    assert!(n.r > 0.0);
}

#[test]
fn ermentrout_kopell_reset_restores_initial_state() {
    let mut n = ErmentroutKopellPopulation::new();
    for _ in 0..500 {
        n.step(0.0);
    }
    n.reset();
    assert!((n.r - 0.1).abs() < 1e-10);
    assert!((n.v - (-2.0)).abs() < 1e-10);
}

#[test]
fn ermentrout_kopell_state_remains_finite() {
    let mut n = ErmentroutKopellPopulation::new();
    for _ in 0..5000 {
        n.step(1.0);
    }
    assert!(n.r.is_finite());
    assert!(n.v.is_finite());
}

#[test]
fn ermentrout_kopell_nan_preserves_state() {
    let mut n = ErmentroutKopellPopulation::new();
    let before = (n.r, n.v);
    assert_eq!(n.step(f64::NAN), before.0);
    assert_eq!((n.r, n.v), before);
}
