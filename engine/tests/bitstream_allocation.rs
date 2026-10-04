// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Packed Bernoulli allocation contracts

//! Fallible packed generation preserves sampled bits and RNG state.

use rand::{RngExt, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;
use sc_neurocore_engine::bitstream::{bernoulli_stream, pack, try_bernoulli_packed};

#[test]
fn checked_generation_matches_unpacked_reference_and_rng_position() {
    for length in [0, 1, 63, 64, 65, 129] {
        for probability in [-1.0, 0.0, 0.5, 1.0, 2.0, f64::NAN] {
            let mut checked_rng = Xoshiro256PlusPlus::seed_from_u64(42);
            let mut reference_rng = checked_rng.clone();
            let actual = try_bernoulli_packed(probability, length, &mut checked_rng).unwrap();
            let expected = pack(&bernoulli_stream(probability, length, &mut reference_rng));
            assert_eq!(actual, expected.data);
            assert_eq!(checked_rng.random::<u64>(), reference_rng.random::<u64>());
        }
    }
}

#[test]
fn impossible_reservation_preserves_rng_state() {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(71);
    let mut unchanged = rng.clone();
    assert!(try_bernoulli_packed(0.5, usize::MAX, &mut rng).is_err());
    assert_eq!(rng.random::<u64>(), unchanged.random::<u64>());
}
