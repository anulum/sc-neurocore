// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Retained normalized energy-LIF Rust state contracts

use sc_neurocore_engine::neurons::SCNormalizedEnergyLIFNeuron;

#[path = "../../src/sc_neurocore/accel/rust/safety/sc_normalized_energy_lif.rs"]
mod safety;

/// An invalid resting voltage cannot be accepted or committed by reset.
#[test]
fn resting_configuration_refusal_is_atomic() {
    for rest in [-201.0, 101.0, f64::INFINITY] {
        let mut engine = SCNormalizedEnergyLIFNeuron::new();
        let mut checked = safety::SCNormalizedEnergyLIFNeuron::new();
        engine.v_rest = rest;
        checked.v_rest = rest;
        engine.v_threshold = 102.0;
        checked.v_threshold = 102.0;
        let before = (engine.v, engine.epsilon);
        assert!(!engine.valid() && !checked.valid());
        assert!(engine.try_step(0.0).is_err());
        assert_eq!(checked.step(0.0), -1);
        assert_eq!(
            engine.try_reset(),
            Err("invalid SC normalized EnergyLIF reset state or configuration")
        );
        engine.reset();
        assert_eq!((engine.v, engine.epsilon), before);
        assert_eq!((checked.v, checked.epsilon), before);
        engine.v_rest = -70.0;
        assert_eq!(engine.try_reset(), Ok(()));
        assert_eq!(engine.try_step(30.0), Ok(0));
    }
}

/// Both inclusive resting endpoints recover invalid dynamic state.
#[test]
fn reset_candidate_recovers_without_using_invalid_dynamic_state() {
    for rest in [-200.0, 100.0] {
        for energy in [0.0, 1.0] {
            let mut state = SCNormalizedEnergyLIFNeuron::new();
            state.v_rest = rest;
            state.v_threshold = 101.0;
            state.epsilon_0 = energy;
            state.v = f64::NAN;
            state.epsilon = -1.0;
            assert_eq!(state.try_reset(), Ok(()));
            assert_eq!((state.v, state.epsilon), (rest, energy));
            assert!(state.valid());
            assert_eq!(state.try_step(0.0), Ok(0));
        }
    }
}

/// Every nonfinite field is rejected by the engine and standalone safety lane.
#[test]
fn nonfinite_complete_configuration_is_refused() {
    for index in 0..11 {
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let mut engine = SCNormalizedEnergyLIFNeuron::new();
            let mut checked = safety::SCNormalizedEnergyLIFNeuron::new();
            let (a, b) = match index {
                0 => (&mut engine.v, &mut checked.v),
                1 => (&mut engine.epsilon, &mut checked.epsilon),
                2 => (&mut engine.v_rest, &mut checked.v_rest),
                3 => (&mut engine.v_reset, &mut checked.v_reset),
                4 => (&mut engine.v_threshold, &mut checked.v_threshold),
                5 => (&mut engine.tau_m, &mut checked.tau_m),
                6 => (&mut engine.tau_e, &mut checked.tau_e),
                7 => (&mut engine.alpha, &mut checked.alpha),
                8 => (&mut engine.epsilon_0, &mut checked.epsilon_0),
                9 => (&mut engine.resistance, &mut checked.resistance),
                _ => (&mut engine.dt, &mut checked.dt),
            };
            *a = bad;
            *b = bad;
            let before = (engine.v.to_bits(), engine.epsilon.to_bits());
            assert!(!engine.valid() && !checked.valid());
            assert!(engine.try_step(30.0).is_err());
            assert_eq!(checked.step(30.0), -1);
            assert_eq!((engine.v.to_bits(), engine.epsilon.to_bits()), before);
            assert_eq!((checked.v.to_bits(), checked.epsilon.to_bits()), before);
        }
    }
}

/// Candidate overflow preserves both lanes and permits a normal retry.
#[test]
fn candidate_failure_and_valid_trace_match_both_rust_lanes() {
    let mut engine = SCNormalizedEnergyLIFNeuron::new();
    let mut checked = safety::SCNormalizedEnergyLIFNeuron::new();
    let before = (engine.v, engine.epsilon);
    assert!(engine.try_step(f64::MAX).is_err());
    assert_eq!(checked.step(f64::MAX), -1);
    assert_eq!((engine.v, engine.epsilon), before);
    assert_eq!((checked.v, checked.epsilon), before);
    engine.tau_e = 10.000001;
    checked.tau_e = 10.000001;
    engine.epsilon = 0.25;
    checked.epsilon = 0.25;
    for current in [30.0, 0.0, 50.0, 10.0].into_iter().cycle().take(256) {
        assert_eq!(engine.try_step(current), Ok(checked.step(current)));
        assert_eq!((engine.v, engine.epsilon), (checked.v, checked.epsilon));
    }
}
