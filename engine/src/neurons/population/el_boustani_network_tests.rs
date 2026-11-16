// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — El Boustani network engine unit tests

use super::*;

#[test]
fn elboustani_fires_with_input() {
    let mut n = ElBoustaniNetwork::new();
    let mut spikes = 0;
    for _ in 0..50_000 {
        spikes += n.step(5.0);
    }
    assert!(
        spikes > 0,
        "ElBoustani must produce bursts with input, got {spikes}"
    );
}

#[test]
fn elboustani_silent_without_input() {
    let mut n = ElBoustaniNetwork::new();
    let mut spikes = 0;
    for _ in 0..50_000 {
        spikes += n.step(0.0);
    }
    assert_eq!(
        spikes, 0,
        "ElBoustani must be quiescent without input, got {spikes}"
    );
}

#[test]
fn elboustani_nmda_builds_with_activity() {
    // NMDA gating variable s should increase with sustained E activity
    let mut n = ElBoustaniNetwork::new();
    let s0 = n.s;
    for _ in 0..10_000 {
        n.step(5.0);
    }
    assert!(
        n.s > s0,
        "NMDA gating should increase with activity: s0={s0}, s_now={}",
        n.s
    );
}

#[test]
fn elboustani_ei_balance() {
    // Inhibition should keep E rate bounded
    let mut n = ElBoustaniNetwork::new();
    for _ in 0..50_000 {
        n.step(3.0);
    }
    assert!(
        n.r_e < 50.0,
        "E/I balance should keep r_e bounded, r_e={}",
        n.r_e
    );
    assert!(n.r_i >= 0.0, "r_i must be non-negative");
}

#[test]
fn elboustani_nmda_enhances_excitation() {
    // With NMDA (j_nmda > 0), E rate should be higher than without
    let mut with_nmda = ElBoustaniNetwork::new();
    let mut no_nmda = ElBoustaniNetwork::new();
    no_nmda.j_nmda = 0.0;
    for _ in 0..20_000 {
        with_nmda.step(3.0);
        no_nmda.step(3.0);
    }
    assert!(
        with_nmda.r_e >= no_nmda.r_e,
        "NMDA should enhance excitation: with={:.3} vs without={:.3}",
        with_nmda.r_e,
        no_nmda.r_e
    );
}

#[test]
fn elboustani_nmda_bounded() {
    // NMDA gating s must stay in [0, 1]
    let mut n = ElBoustaniNetwork::new();
    for _ in 0..50_000 {
        n.step(10.0);
    }
    assert!(
        n.s >= 0.0 && n.s <= 1.0,
        "NMDA gating must be in [0,1], s={}",
        n.s
    );
}

#[test]
fn elboustani_rates_non_negative() {
    let mut n = ElBoustaniNetwork::new();
    for _ in 0..50_000 {
        n.step(-10.0);
    }
    assert!(n.r_e >= 0.0);
    assert!(n.r_i >= 0.0);
}

#[test]
fn elboustani_nan_input_stays_finite() {
    let mut n = ElBoustaniNetwork::new();
    n.step(f64::NAN);
    assert!(n.r_e.is_finite());
    assert!(n.r_i.is_finite());
    assert!(n.s.is_finite());
}

#[test]
fn elboustani_extreme_input_bounded() {
    let mut n = ElBoustaniNetwork::new();
    for _ in 0..10_000 {
        n.step(1e6);
    }
    assert!(n.r_e.is_finite() && n.r_e <= 200.0);
    assert!(n.s >= 0.0 && n.s <= 1.0);
}

#[test]
fn elboustani_reset_clears_state() {
    let mut n = ElBoustaniNetwork::new();
    for _ in 0..10_000 {
        n.step(5.0);
    }
    n.reset();
    assert_eq!(n.r_e, 0.1);
    assert_eq!(n.r_i, 0.1);
    assert_eq!(n.s, 0.0);
}
