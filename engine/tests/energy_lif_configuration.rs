// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — EnergyLIF configuration and reset contracts

use sc_neurocore_engine::neurons::EnergyLIFNeuron;

#[path = "../../src/sc_neurocore/accel/rust/safety/energy_lif.rs"]
mod safety;

/// Both Rust lanes refuse undefined normalizers and unsafe equilibrium states.
#[test]
fn invalid_equilibrium_configuration_refuses_without_mutation() {
    for (e0, alpha, epsilon0) in [
        (-62.5, 1.0, 0.0),
        (-201.0, 1.0, 0.5),
        (101.0, 1.0, 0.5),
        (-62.5, 11.0, 0.5),
        (-62.5, f64::MAX, 2.0),
        (-62.5, 1.0e-300, 1.0e-300),
    ] {
        let mut engine = EnergyLIFNeuron::new();
        engine.e_0 = e0;
        engine.alpha = alpha;
        engine.epsilon_0 = epsilon0;
        let mut checked = safety::EnergyLIFNeuron::new();
        checked.e_0 = e0;
        checked.alpha = alpha;
        checked.epsilon_0 = epsilon0;
        let before = (engine.v, engine.epsilon);
        assert!(!engine.valid());
        assert!(!checked.valid());
        assert!(engine.try_step(80.0).is_err());
        assert_eq!(checked.step(80.0), -1);
        assert!(engine.try_reset().is_err());
        engine.reset();
        assert_eq!((engine.v, engine.epsilon), before);
        assert_eq!((checked.v, checked.epsilon), before);
        engine.e_0 = -62.5;
        engine.alpha = 1.0;
        engine.epsilon_0 = 0.5;
        assert_eq!(engine.try_step(80.0), Ok(0));
    }
}

/// Reset admits inclusive voltage/energy bounds and repairs dynamic state.
#[test]
fn reset_validates_the_candidate_before_committing() {
    for (e0, alpha, epsilon0) in [(-200.0, 10.0, 0.5), (100.0, 0.5, 0.5)] {
        let mut engine = EnergyLIFNeuron::new();
        engine.e_0 = e0;
        engine.alpha = alpha;
        engine.epsilon_0 = epsilon0;
        assert!(engine.valid());
        engine.v = f64::NAN;
        engine.epsilon = -1.0;
        assert_eq!(engine.try_reset(), Ok(()));
        assert_eq!((engine.v, engine.epsilon), (e0, alpha * epsilon0));
        assert!(engine.valid());
    }
}

/// Unsafe candidates leave both Rust states unchanged and allow a valid retry.
#[test]
fn candidate_failure_is_atomic_and_recoverable() {
    let mut engine = EnergyLIFNeuron::new();
    let mut checked = safety::EnergyLIFNeuron::new();
    let before = (engine.v, engine.epsilon);
    assert!(engine.try_step(f64::MAX).is_err());
    assert_eq!(checked.step(f64::MAX), -1);
    assert_eq!((engine.v, engine.epsilon), before);
    assert_eq!((checked.v, checked.epsilon), before);
    assert_eq!(engine.try_step(80.0), Ok(checked.step(80.0)));
    assert!((engine.v - checked.v).abs() <= 2.0e-12);
    assert!((engine.epsilon - checked.epsilon).abs() <= 2.0e-12);
}
