// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Brunel balanced network engine unit tests

use super::*;

#[test]
fn brunel_fires_with_input() {
    let mut n = BrunelNetwork::new();
    let mut spikes = 0;
    for _ in 0..50_000 {
        spikes += n.step(5.0);
    }
    assert!(
        spikes > 0,
        "Brunel must produce bursts with input, got {spikes}"
    );
}

#[test]
fn brunel_silent_without_input() {
    let mut n = BrunelNetwork::new();
    let mut spikes = 0;
    for _ in 0..50_000 {
        spikes += n.step(0.0);
    }
    assert_eq!(
        spikes, 0,
        "Brunel must be quiescent without input, got {spikes}"
    );
}

#[test]
fn brunel_ei_balance() {
    // Strong inhibition keeps E rate bounded
    let mut n = BrunelNetwork::new();
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
fn brunel_inhibition_suppresses() {
    // Increasing j_ei should reduce E rate
    let mut weak_inh = BrunelNetwork::new();
    weak_inh.j_ei = 0.3;
    let mut strong_inh = BrunelNetwork::new();
    strong_inh.j_ei = 2.0;

    for _ in 0..20_000 {
        weak_inh.step(5.0);
        strong_inh.step(5.0);
    }
    assert!(
        weak_inh.r_e >= strong_inh.r_e,
        "Stronger inhibition → lower E rate: weak={:.2} vs strong={:.2}",
        weak_inh.r_e,
        strong_inh.r_e
    );
}

#[test]
fn brunel_rates_non_negative() {
    let mut n = BrunelNetwork::new();
    for _ in 0..50_000 {
        n.step(-10.0);
    }
    assert!(n.r_e >= 0.0);
    assert!(n.r_i >= 0.0);
}

#[test]
fn brunel_negative_input_no_crash() {
    let mut n = BrunelNetwork::new();
    for _ in 0..50_000 {
        n.step(-100.0);
    }
    assert!(n.r_e.is_finite());
    assert!(n.r_i.is_finite());
}

#[test]
fn brunel_nan_input_stays_finite() {
    let mut n = BrunelNetwork::new();
    n.step(f64::NAN);
    assert!(n.r_e.is_finite());
    assert!(n.r_i.is_finite());
}

#[test]
fn brunel_extreme_input_bounded() {
    let mut n = BrunelNetwork::new();
    for _ in 0..10_000 {
        n.step(1e6);
    }
    assert!(n.r_e.is_finite() && n.r_e <= 200.0);
}

#[test]
fn brunel_reset_clears_state() {
    let mut n = BrunelNetwork::new();
    for _ in 0..10_000 {
        n.step(5.0);
    }
    n.reset();
    assert_eq!(n.r_e, 0.1);
    assert_eq!(n.r_i, 0.1);
}

#[test]
fn brunel_performance_100k_steps() {
    let start = std::time::Instant::now();
    let mut n = BrunelNetwork::new();
    for _ in 0..100_000 {
        std::hint::black_box(n.step(3.0));
    }
    let elapsed = start.elapsed();
    assert!(
        elapsed.as_millis() < 50,
        "100k steps must complete in <50ms"
    );
}
