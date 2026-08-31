// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — DPI engine unit tests

use super::*;

#[test]
fn dpi_fires() {
    let mut n = DPINeuron::new();
    let t: i32 = (0..20_000).map(|_| n.step(5.0)).sum();
    assert_eq!(t, 44);
}

#[test]
fn dpi_silent() {
    let mut n = DPINeuron::new();
    let t: i32 = (0..100).map(|_| n.step(0.0)).sum();
    assert_eq!(t, 0);
}

#[test]
fn dpi_reset() {
    let mut n = DPINeuron::new();
    for _ in 0..500 {
        n.step(5.0);
    }
    n.reset();
    assert_eq!(n.i_mem, n.i_reset);
    assert_eq!(n.i_ahp, n.i_0);
    assert_eq!(n.refractory_time, 0.0);
}

#[test]
fn dpi_nan_no_panic() {
    let mut n = DPINeuron::new();
    let old = (n.i_mem, n.i_ahp, n.refractory_time);
    assert_eq!(n.step(f64::NAN), 0);
    assert_eq!((n.i_mem, n.i_ahp, n.refractory_time), old);
}

#[test]
fn dpi_coupled_euler_anchor() {
    let mut n = DPINeuron::new();
    assert_eq!(n.step(5.0), 0);
    assert!((n.i_mem - 0.010201975272610835).abs() < 1.0e-17);
    assert!((n.i_ahp - 0.00999).abs() < 1.0e-17);
}

#[test]
fn dpi_refractory_pulse_drives_adaptation() {
    let mut n = DPINeuron {
        refractory_time: 2.0,
        ..DPINeuron::new()
    };
    assert_eq!(n.step(0.0), 0);
    assert_eq!(n.i_mem, n.i_reset);
    assert!(n.i_ahp > 0.01);
    assert_eq!(n.refractory_time, 1.9);
}

#[test]
fn complete_packet_carries_all_states_and_events() {
    let mut n = DPINeuron {
        i_mem: 0.37,
        i_ahp: 0.08,
        i_threshold: 1.3,
        i_reset: 0.2,
        i_rest: 0.15,
        i_tau: 0.9,
        i_g: 1.4,
        i_tau_ahp: 0.12,
        i_ga: 0.8,
        i_spike: 4.2,
        i_0: 0.02,
        kappa: 0.65,
        alpha: 8.0,
        tau: 7.0,
        tau_ahp: 45.0,
        refractory_period: 0.6,
        dt: 0.05,
        ..DPINeuron::new()
    };
    let (i_mem, i_ahp, refractory, events) = n.simulate_complete(400, 5.0).unwrap();
    assert_eq!(i_mem.len(), 400);
    assert_eq!(i_ahp.len(), 400);
    assert_eq!(refractory.len(), 400);
    assert_eq!(
        events
            .iter()
            .map(|event| usize::from(*event))
            .sum::<usize>(),
        4
    );
    assert_eq!(n.i_mem, i_mem[399]);
    assert_eq!(n.i_ahp, i_ahp[399]);
    assert_eq!(n.refractory_time, refractory[399]);
}

#[test]
fn complete_packet_rejection_is_atomic() {
    let mut n = DPINeuron {
        tau: f64::MIN_POSITIVE,
        ..DPINeuron::new()
    };
    let before = n.clone();
    assert!(n.simulate_complete(2, f64::MAX).is_err());
    assert_eq!(n.i_mem, before.i_mem);
    assert_eq!(n.i_ahp, before.i_ahp);
    assert_eq!(n.refractory_time, before.refractory_time);
}
