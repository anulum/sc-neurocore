// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Native SC resetting-MAT reset and complete trace contracts

use sc_neurocore_engine::neurons::SCResettingMATNeuron;

#[path = "../../src/sc_neurocore/accel/rust/safety/sc_resetting_mat.rs"]
mod safety;

fn values(state: &SCResettingMATNeuron) -> [f64; 13] {
    [
        state.v,
        state.theta1,
        state.theta2,
        state.v_rest,
        state.v_reset,
        state.v_threshold_base,
        state.tau_m,
        state.tau_1,
        state.tau_2,
        state.h1,
        state.h2,
        state.resistance,
        state.dt,
    ]
}

fn checked_values(state: &safety::SCResettingMATNeuron) -> [f64; 13] {
    [
        state.v,
        state.theta1,
        state.theta2,
        state.v_rest,
        state.v_reset,
        state.v_threshold_base,
        state.tau_m,
        state.tau_1,
        state.tau_2,
        state.h1,
        state.h2,
        state.resistance,
        state.dt,
    ]
}

fn pair(values: [f64; 13]) -> (SCResettingMATNeuron, safety::SCResettingMATNeuron) {
    let [v, theta1, theta2, v_rest, v_reset, v_threshold_base, tau_m, tau_1, tau_2, h1, h2, resistance, dt] =
        values;
    (
        SCResettingMATNeuron {
            v,
            theta1,
            theta2,
            v_rest,
            v_reset,
            v_threshold_base,
            tau_m,
            tau_1,
            tau_2,
            h1,
            h2,
            resistance,
            dt,
        },
        safety::SCResettingMATNeuron {
            v,
            theta1,
            theta2,
            v_rest,
            v_reset,
            v_threshold_base,
            tau_m,
            tau_1,
            tau_2,
            h1,
            h2,
            resistance,
            dt,
        },
    )
}

/// Both Rust public lanes preserve every field on reset refusal and recover.
#[test]
fn reset_refusal_preserves_complete_state_in_both_lanes() {
    let defaults = values(&SCResettingMATNeuron::new());
    let mut invalid = vec![
        (3, -500.0),
        (3, 500.0),
        (4, -201.0),
        (4, 101.0),
        (6, 0.0),
        (6, -1.0),
        (7, 0.0),
        (7, -1.0),
        (8, 0.0),
        (8, -1.0),
        (9, -1.0),
        (9, 1.0e9 + 1.0),
        (10, -1.0),
        (10, 1.0e9 + 1.0),
        (11, 0.0),
        (11, -1.0),
        (12, 0.0),
        (12, -1.0),
    ];
    for index in 3..13 {
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            invalid.push((index, bad));
        }
    }
    for (index, bad) in invalid {
        let mut input = defaults;
        input[0] = -65.0;
        input[1] = 2.0;
        input[2] = 3.0;
        input[index] = bad;
        let (mut engine, mut checked) = pair(input);
        let before = input.map(f64::to_bits);
        assert!(engine.try_reset().is_err() && checked.try_reset().is_err());
        assert_eq!(values(&engine).map(f64::to_bits), before);
        assert_eq!(checked_values(&checked).map(f64::to_bits), before);
        engine.reset();
        checked.reset();
        assert_eq!(values(&engine).map(f64::to_bits), before);
        assert_eq!(checked_values(&checked).map(f64::to_bits), before);
        input[index] = defaults[index];
        (engine, checked) = pair(input);
        assert_eq!(engine.try_reset(), Ok(()));
        assert_eq!(checked.try_reset(), Ok(()));
        assert_eq!(values(&engine), checked_values(&checked));
        assert_eq!((engine.v, engine.theta1, engine.theta2), (-70.0, 0.0, 0.0));
        assert_eq!(engine.try_step(0.0), Ok(checked.step(0.0)));
    }
}

/// Valid complete rests repair invalid dynamic values and retain configuration.
#[test]
fn valid_rest_recovers_dynamic_corruption_in_both_lanes() {
    for index in 0..3 {
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -1e308, 1e308] {
            let mut input = values(&SCResettingMATNeuron::new());
            input[3] = -65.0;
            input[6] = 12.0;
            input[9] = 4.0;
            input[index] = bad;
            let (mut engine, mut checked) = pair(input);
            assert_eq!(engine.try_reset(), Ok(()));
            assert_eq!(checked.try_reset(), Ok(()));
            assert_eq!((engine.v, engine.theta1, engine.theta2), (-65.0, 0.0, 0.0));
            assert_eq!(values(&engine), checked_values(&checked));
            assert_eq!(values(&engine)[3..], input[3..]);
            assert_eq!(engine.try_step(0.0), Ok(checked.step(0.0)));
        }
    }
}

/// Configured complete 256-step trajectories agree with the independent safety lane.
#[test]
fn configured_complete_traces_match_both_rust_lanes() {
    for dt in [0.25, 1.0, 2.0] {
        for current in [0.0, 20.0, 50.0] {
            let input = [
                -66.0, 2.0, 3.0, -70.0, -71.0, -49.0, 12.0, 15.0, 250.0, 4.0, 2.0, 1.25, dt,
            ];
            let (mut engine, mut checked) = pair(input);
            assert!(engine.validate() && safety::validate_sc_resetting_mat(&checked));
            for drive in [current, 0.0, 50.0, 10.0].into_iter().cycle().take(256) {
                assert_eq!(engine.try_step(drive), Ok(checked.step(drive)));
                assert_eq!(values(&engine), checked_values(&checked));
            }
            engine.reset();
            checked.reset();
            assert_eq!(values(&engine), checked_values(&checked));
            assert_eq!((engine.v, engine.theta1, engine.theta2), (-70.0, 0.0, 0.0));
        }
    }
}

/// Every nonfinite field and finite candidate overflow refuse atomically.
#[test]
fn invalid_step_preserves_complete_state_and_next_valid_transition() {
    for index in 0..13 {
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let mut input = values(&SCResettingMATNeuron::new());
            input[index] = bad;
            let (mut engine, mut checked) = pair(input);
            assert!(!engine.validate() && !safety::validate_sc_resetting_mat(&checked));
            assert!(engine.try_step(50.0).is_err());
            assert_eq!(checked.step(50.0), -1);
            assert_eq!(values(&engine).map(f64::to_bits), input.map(f64::to_bits));
            assert_eq!(
                checked_values(&checked).map(f64::to_bits),
                input.map(f64::to_bits)
            );
        }
    }
    let input = values(&SCResettingMATNeuron::new());
    let (mut engine, mut checked) = pair(input);
    assert!(engine.try_step(1e308).is_err());
    assert_eq!(checked.step(1e308), -1);
    assert_eq!(values(&engine), input);
    assert_eq!(checked_values(&checked), input);
    assert_eq!(engine.try_step(0.0), Ok(checked.step(0.0)));
}
