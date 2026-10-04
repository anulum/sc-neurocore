// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Retained adaptive LIF native state contracts

use sc_neurocore_engine::neurons::SCNonResettingAdaptiveLIFNeuron;

#[path = "../../src/sc_neurocore/accel/rust/safety/sc_non_resetting_adaptive_lif.rs"]
mod safety;

/// Both maintained Rust lanes refuse every nonfinite field without mutation.
#[test]
fn complete_configuration_refusal_is_atomic() {
    for index in 0..9 {
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let mut engine = SCNonResettingAdaptiveLIFNeuron::new();
            let mut checked = safety::SCNonResettingAdaptiveLIFNeuron::new();
            let (a, b) = match index {
                0 => (&mut engine.v, &mut checked.v),
                1 => (&mut engine.theta, &mut checked.theta),
                2 => (&mut engine.v_rest, &mut checked.v_rest),
                3 => (&mut engine.theta_rest, &mut checked.theta_rest),
                4 => (&mut engine.delta_theta, &mut checked.delta_theta),
                5 => (&mut engine.tau_m, &mut checked.tau_m),
                6 => (&mut engine.tau_theta, &mut checked.tau_theta),
                7 => (&mut engine.r_m, &mut checked.r_m),
                _ => (&mut engine.dt, &mut checked.dt),
            };
            *a = value;
            *b = value;
            let before = (engine.v.to_bits(), engine.theta.to_bits());
            assert!(!engine.validate() && !checked.valid());
            assert!(engine.try_step(20.0).is_err() && checked.step(20.0).is_err());
            assert_eq!((engine.v.to_bits(), engine.theta.to_bits()), before);
            assert_eq!((checked.v.to_bits(), checked.theta.to_bits()), before);
        }
    }
}

/// Invalid rests refuse atomically; valid rests recover invalid dynamic values.
#[test]
fn reset_refusal_and_recovery_match_both_rust_lanes() {
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let mut engine = SCNonResettingAdaptiveLIFNeuron::new();
        let mut checked = safety::SCNonResettingAdaptiveLIFNeuron::new();
        engine.v_rest = value;
        checked.v_rest = value;
        let before = (engine.v, engine.theta);
        assert!(engine.try_reset().is_err() && checked.try_reset().is_err());
        engine.reset();
        checked.reset();
        assert_eq!((engine.v, engine.theta), before);
        assert_eq!((checked.v, checked.theta), before);
        engine.v_rest = -65.0;
        checked.v_rest = -65.0;
        engine.v = f64::NAN;
        checked.v = f64::NAN;
        engine.theta = f64::INFINITY;
        checked.theta = f64::INFINITY;
        assert_eq!(engine.try_reset(), Ok(()));
        assert_eq!(checked.try_reset(), Ok(()));
        assert_eq!((engine.v, engine.theta), (-65.0, -50.0));
        assert_eq!((checked.v, checked.theta), (-65.0, -50.0));
    }
}

/// Configured exact traces include accepted timesteps larger than time constants.
#[test]
fn configured_full_trace_matches_both_rust_lanes() {
    for dt in [0.1, 10.0, 40.0, 1000.0, 1e308] {
        let mut engine = SCNonResettingAdaptiveLIFNeuron::new();
        let mut checked = safety::SCNonResettingAdaptiveLIFNeuron::new();
        engine.dt = dt;
        checked.dt = dt;
        for current in [20.0, 0.0, 60.0, 10.0].into_iter().cycle().take(256) {
            assert_eq!(engine.try_step(current), checked.step(current));
            assert_eq!((engine.v, engine.theta), (checked.v, checked.theta));
        }
        engine.reset();
        checked.reset();
        assert_eq!(engine.step(0.0), checked.step(0.0).unwrap());
    }
}

/// Finite current-gain overflow cannot commit either dynamic value.
#[test]
fn finite_overflow_preserves_state_and_recovers() {
    let mut engine = SCNonResettingAdaptiveLIFNeuron::new();
    let mut checked = safety::SCNonResettingAdaptiveLIFNeuron::new();
    engine.r_m = 1e308;
    checked.r_m = 1e308;
    let before = (engine.v, engine.theta);
    assert!(engine.try_step(20.0).is_err() && checked.step(20.0).is_err());
    assert_eq!((engine.v, engine.theta), before);
    assert_eq!((checked.v, checked.theta), before);
    assert_eq!(engine.try_step(0.0), Ok(0));
    assert_eq!(checked.step(0.0), Ok(0));
}
