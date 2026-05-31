// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// (C) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
// (C) Code 2020-2026 Miroslav Sotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore motor unit integration tests

use sc_neurocore_engine::neurons::MotorUnit;

fn relax(previous: f64, steady: f64, tau: f64, dt: f64) -> f64 {
    steady + (previous - steady) * (-dt / tau).exp()
}

fn reference_step(mut unit: MotorUnit, drive: f64) -> MotorUnit {
    let mut force = unit.force * (-unit.dt / unit.tau_twitch).exp();
    let input_drive = unit.gain * drive.max(0.0) - unit.adapt;
    let v_target = unit.v_rest + input_drive;
    let mut v_candidate = relax(unit.v, v_target, unit.tau_m, unit.dt);
    let adapt_target = unit.a_adapt * (v_candidate - unit.v_rest);
    let adapt = relax(unit.adapt, adapt_target, unit.tau_adapt, unit.dt);
    if v_candidate >= unit.v_threshold {
        v_candidate = unit.v_reset;
        force = (force + unit.twitch_amp).min(1.0);
    }
    unit.v = v_candidate;
    unit.adapt = adapt;
    unit.force = force;
    unit
}

fn snapshot(unit: &MotorUnit) -> (f64, f64, f64) {
    (unit.v, unit.adapt, unit.force)
}

#[test]
fn motor_unit_uses_exact_lif_adaptation_and_force_decay_step() {
    let mut unit = MotorUnit::new();
    let expected = reference_step(MotorUnit::new(), 20.0);

    assert_eq!(unit.step(20.0), 0);

    assert!((unit.v - expected.v).abs() <= 1e-12);
    assert!((unit.adapt - expected.adapt).abs() <= 1e-12);
    assert!((unit.force - expected.force).abs() <= 1e-12);
}

#[test]
fn motor_unit_invalid_drive_preserves_state() {
    let mut unit = MotorUnit::new();
    for _ in 0..20 {
        unit.step(20.0);
    }
    let before = snapshot(&unit);

    assert_eq!(unit.step(f64::NAN), 0);
    assert_eq!(snapshot(&unit), before);
    assert_eq!(unit.step(f64::INFINITY), 0);
    assert_eq!(snapshot(&unit), before);
}

#[test]
fn motor_unit_excess_drive_preserves_state() {
    let mut unit = MotorUnit::new();
    let before = snapshot(&unit);

    assert_eq!(unit.step(1.0e8), 0);

    assert_eq!(snapshot(&unit), before);
}

#[test]
fn motor_unit_spike_adds_twitch_and_force_stays_bounded() {
    let mut unit = MotorUnit::fast();
    let spikes: i32 = (0..1000).map(|_| unit.step(50.0)).sum();

    assert!(spikes > 0);
    assert!((0.0..=1.0).contains(&unit.force));
    let force_after_drive = unit.force;
    for _ in 0..200 {
        unit.step(0.0);
    }
    assert!((0.0..=force_after_drive).contains(&unit.force));
}
