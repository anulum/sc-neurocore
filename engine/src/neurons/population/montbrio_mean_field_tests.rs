// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Montbrió-Pazó-Roxin mean field engine unit tests

use super::*;

#[test]
fn mpr_fires_with_input() {
    let mut n = MontbrioMeanField::new();
    let mut spikes = 0;
    for _ in 0..50_000 {
        spikes += n.step(10.0);
    }
    assert!(
        spikes > 0,
        "MPR must produce bursts with strong input, got {spikes}"
    );
}

#[test]
fn mpr_silent_without_input() {
    // eta = -5 (below threshold), no input → quiescent
    let mut n = MontbrioMeanField::new();
    let mut spikes = 0;
    for _ in 0..50_000 {
        spikes += n.step(0.0);
    }
    assert_eq!(
        spikes, 0,
        "MPR must be quiescent without input (eta<0), got {spikes}"
    );
}

#[test]
fn mpr_rate_increases_with_input() {
    let mut low = MontbrioMeanField::new();
    let mut high = MontbrioMeanField::new();
    for _ in 0..10_000 {
        low.step(3.0);
        high.step(15.0);
    }
    assert!(
        high.r > low.r,
        "Higher input → higher rate: high={:.3} vs low={:.3}",
        high.r,
        low.r
    );
}

#[test]
fn mpr_two_ode_dynamics() {
    // Verify both r and v evolve from initial conditions
    let mut n = MontbrioMeanField::new();
    let r0 = n.r;
    let v0 = n.v;
    for _ in 0..1000 {
        n.step(5.0);
    }
    assert!(
        n.r != r0 || n.v != v0,
        "State must evolve from initial conditions"
    );
}

#[test]
fn mpr_rate_non_negative() {
    let mut n = MontbrioMeanField::new();
    for _ in 0..50_000 {
        n.step(-10.0);
    }
    assert!(n.r >= 0.0, "Rate must be non-negative, r={}", n.r);
}

#[test]
fn mpr_negative_input_no_crash() {
    let mut n = MontbrioMeanField::new();
    for _ in 0..50_000 {
        n.step(-100.0);
    }
    assert!(n.r.is_finite());
    assert!(n.v.is_finite());
}

#[test]
fn mpr_nan_input_stays_finite() {
    let mut n = MontbrioMeanField::new();
    n.step(f64::NAN);
    assert!(n.r.is_finite());
    assert!(n.v.is_finite());
}

#[test]
fn mpr_extreme_input_bounded() {
    let mut n = MontbrioMeanField::new();
    for _ in 0..10_000 {
        n.step(1e6);
    }
    assert!(n.r.is_finite() && n.r <= 100.0);
    assert!(n.v.is_finite() && n.v <= 50.0);
}

#[test]
fn mpr_reset_clears_state() {
    let mut n = MontbrioMeanField::new();
    for _ in 0..10_000 {
        n.step(10.0);
    }
    n.reset();
    assert_eq!(n.r, 0.01);
    assert_eq!(n.v, -2.0);
}

#[test]
fn mpr_performance_100k_steps() {
    let start = std::time::Instant::now();
    let mut n = MontbrioMeanField::new();
    for _ in 0..100_000 {
        std::hint::black_box(n.step(5.0));
    }
    let elapsed = start.elapsed();
    assert!(
        elapsed.as_millis() < 50,
        "100k steps must complete in <50ms"
    );
}
