// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Public fixed-point LIF configuration contracts

use sc_neurocore_engine::neuron::FixedPointLif;

/// Reject configurations that cannot execute with the native state width.
#[test]
fn checked_constructor_rejects_invalid_parameters() {
    let cases = [
        (0, 8, 2, "data_width must be in [1, 16]"),
        (17, 8, 2, "data_width must be in [1, 16]"),
        (32, 8, 2, "data_width must be in [1, 16]"),
        (33, 8, 2, "data_width must be in [1, 16]"),
        (u32::MAX, 8, 2, "data_width must be in [1, 16]"),
        (16, 16, 2, "fraction must be less than data_width"),
        (16, 32, 2, "fraction must be less than data_width"),
        (16, u32::MAX, 2, "fraction must be less than data_width"),
        (16, 8, -1, "refractory_period must be nonnegative"),
        (16, 8, i32::MIN, "refractory_period must be nonnegative"),
    ];
    for (width, fraction, refractory, message) in cases {
        let refusal = FixedPointLif::try_new(width, fraction, 0, 0, 256, refractory);
        assert_eq!(refusal.unwrap_err(), message);
    }
}

/// Every accepted width/fraction pair can step, clone independently and reset.
#[test]
fn checked_native_domain_executes_with_independent_state() {
    for width in 1..=16 {
        for fraction in 0..width {
            let original = FixedPointLif::try_new(width, fraction, 0, -1, 1, 3).unwrap();
            let mut clone = original.clone();
            assert_eq!(clone.step(0, 0, 0, 0), (0, 0));
            let expected_spike = i32::from(width > 1);
            assert_eq!(clone.step(0, 0, 0, 1), (expected_spike, -1));
            assert_eq!(original.v, 0);
            assert_eq!(original.refractory_counter, 0);
            assert_eq!(clone.refractory_counter, if width > 1 { 3 } else { 0 });
            clone.reset();
            assert_eq!(clone.v, 0);
            assert_eq!(clone.refractory_counter, 0);
            assert_eq!(clone.step(0, 0, 0, 0), (0, 0));
        }
    }
}

/// Existing infallible Rust construction has an explicit invalid-input panic.
#[test]
#[should_panic(expected = "invalid fixed-point LIF configuration")]
fn compatibility_constructor_panics_on_invalid_configuration() {
    FixedPointLif::new(0, 8, 0, 0, 256, 2);
}
