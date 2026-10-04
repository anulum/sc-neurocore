// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Source MAT complete reset and configured runtime contracts

use sc_neurocore_engine::neurons::MATNeuron;

#[path = "../../src/sc_neurocore/accel/rust/safety/mat.rs"]
mod safety;

fn values(state: &MATNeuron) -> [f64; 13] {
    [
        state.v,
        state.theta1,
        state.theta2,
        state.refractory_remaining,
        state.omega,
        state.tau_m,
        state.tau_1,
        state.tau_2,
        state.alpha_1,
        state.alpha_2,
        state.resistance,
        state.refractory_period,
        state.dt,
    ]
}

fn checked_values(state: &safety::MATNeuron) -> [f64; 13] {
    [
        state.v,
        state.theta1,
        state.theta2,
        state.refractory_remaining,
        state.omega,
        state.tau_m,
        state.tau_1,
        state.tau_2,
        state.alpha_1,
        state.alpha_2,
        state.resistance,
        state.refractory_period,
        state.dt,
    ]
}

fn pair(input: [f64; 13]) -> (MATNeuron, safety::MATNeuron) {
    let [v, theta1, theta2, refractory_remaining, omega, tau_m, tau_1, tau_2, alpha_1, alpha_2, resistance, refractory_period, dt] =
        input;
    (
        MATNeuron {
            v,
            theta1,
            theta2,
            refractory_remaining,
            omega,
            tau_m,
            tau_1,
            tau_2,
            alpha_1,
            alpha_2,
            resistance,
            refractory_period,
            dt,
        },
        safety::MATNeuron {
            v,
            theta1,
            theta2,
            refractory_remaining,
            omega,
            tau_m,
            tau_1,
            tau_2,
            alpha_1,
            alpha_2,
            resistance,
            refractory_period,
            dt,
        },
    )
}

/// Checked and compatibility resets preserve all refused state bits and recover.
#[test]
fn reset_refusal_preserves_complete_state_in_both_lanes() {
    let defaults = values(&MATNeuron::new());
    let mut invalid = vec![
        (4, -1e9 - 1.0),
        (4, 1e9 + 1.0),
        (8, -1.0),
        (8, 1e9 + 1.0),
        (9, -1.0),
        (9, 1e9 + 1.0),
        (11, -1.0),
    ];
    for index in [5, 6, 7, 10, 12] {
        invalid.extend([(index, 0.0), (index, -1.0)]);
    }
    for index in 4..13 {
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            invalid.push((index, bad));
        }
    }
    assert_eq!(invalid.len(), 44);
    for (index, bad) in invalid {
        let mut input = defaults;
        input[..4].copy_from_slice(&[20.0, 2.0, 3.0, 0.5]);
        input[index] = bad;
        let (mut engine, mut checked) = pair(input);
        let before = input.map(f64::to_bits);
        assert_eq!(
            engine.try_reset(),
            Err("invalid MAT reset state or configuration")
        );
        assert_eq!(
            checked.try_reset(),
            Err("invalid MAT reset state or configuration")
        );
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
        assert_eq!(&values(&engine)[..4], &[0.0; 4]);
        assert_eq!(values(&engine), checked_values(&checked));
        assert_eq!(engine.try_step(0.0), Ok(checked.step(0.0)));
    }
}

/// Zero-rest candidates repair bad dynamics and preserve every profile field.
#[test]
fn reset_recovers_dynamic_corruption_in_both_lanes() {
    for index in 0..4 {
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -1e308, 1e308] {
            let mut input = values(&MATNeuron::new());
            input[5] = 8.0;
            input[8] = 7.0;
            input[11] = 0.0;
            input[index] = bad;
            let (mut engine, mut checked) = pair(input);
            assert_eq!(engine.try_reset(), Ok(()));
            assert_eq!(checked.try_reset(), Ok(()));
            assert_eq!(&values(&engine)[..4], &[0.0; 4]);
            assert_eq!(values(&engine), checked_values(&checked));
            assert_eq!(values(&engine)[4..], input[4..]);
            assert_eq!(engine.try_step(0.0), Ok(checked.step(0.0)));
        }
    }
}

/// Configured 256-step traces compare all dynamics and events in both Rust lanes.
#[test]
fn configured_complete_traces_match_both_rust_lanes() {
    for dt in [0.001, 0.05, 0.5] {
        for current in [0.0, 0.5, 0.7] {
            let input = [
                4.0, 2.0, 3.0, 0.5, 15.0, 8.0, 12.0, 250.0, 7.0, 3.0, 40.0, 1.0, dt,
            ];
            let (mut engine, mut checked) = pair(input);
            assert!(engine.validate() && safety::validate_mat(&checked));
            for drive in [current, 0.0, 0.7, 0.3].into_iter().cycle().take(256) {
                assert_eq!(engine.try_step(drive), Ok(checked.step(drive)));
                assert_eq!(values(&engine), checked_values(&checked));
            }
            engine.reset();
            checked.reset();
            assert_eq!(values(&engine), checked_values(&checked));
            assert_eq!(&values(&engine)[..4], &[0.0; 4]);
        }
    }
}

/// All nonfinite fields and input overflow refuse atomically before valid retry.
#[test]
fn invalid_step_preserves_complete_state_and_valid_retry() {
    for index in 0..13 {
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let mut input = values(&MATNeuron::new());
            input[index] = bad;
            let (mut engine, mut checked) = pair(input);
            assert!(!engine.validate() && !safety::validate_mat(&checked));
            assert!(engine.try_step(0.7).is_err());
            assert_eq!(checked.step(0.7), -1);
            assert_eq!(values(&engine).map(f64::to_bits), input.map(f64::to_bits));
            assert_eq!(
                checked_values(&checked).map(f64::to_bits),
                input.map(f64::to_bits)
            );
        }
    }
    let input = values(&MATNeuron::new());
    let (mut engine, mut checked) = pair(input);
    assert!(engine.try_step(1e308).is_err());
    assert_eq!(checked.step(1e308), -1);
    assert_eq!(values(&engine), input);
    assert_eq!(checked_values(&checked), input);
    assert_eq!(engine.try_step(0.0), Ok(checked.step(0.0)));
}

/// Public RS/IB/FS profiles keep non-resetting recurrence and their reset profile.
#[test]
fn paper_profiles_keep_non_resetting_events_and_configuration() {
    for (mut engine, profile) in [
        (MATNeuron::default(), [19.0, 37.0, 2.0]),
        (MATNeuron::intrinsically_bursting(), [26.0, 1.7, 2.0]),
        (MATNeuron::fast_spiking(), [11.0, 10.0, 0.002]),
    ] {
        assert_eq!([engine.omega, engine.alpha_1, engine.alpha_2], profile);
        engine.v = 40.0;
        engine.refractory_period = 0.0;
        assert_eq!(engine.threshold(), profile[0]);
        assert_eq!(engine.step(0.7), 1);
        assert!(engine.v > 39.0);
        assert_eq!(engine.threshold(), profile.iter().sum::<f64>());
        engine.reset();
        assert_eq!(engine.threshold(), profile[0]);
        assert_eq!(engine.refractory_period, 0.0);
    }
}
