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
use sc_neurocore_engine::bitstream::{
    bernoulli_packed_fast, bernoulli_packed_into, bernoulli_packed_simd_into, bernoulli_stream,
    pack, try_bernoulli_packed,
};

#[test]
fn scalar_caller_output_matches_unpacked_sampling_and_rng_position() {
    for length in [0_usize, 1, 63, 64, 65, 129, 1023, 1024, 1025, 1089] {
        for probability in [
            f64::NEG_INFINITY,
            -1.0,
            0.0,
            1.0 / 256.0,
            0.5,
            1.0,
            2.0,
            f64::INFINITY,
            f64::NAN,
        ] {
            let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
            let mut reference = rng.clone();
            let mut output = vec![u64::MAX; length.div_ceil(64)];
            bernoulli_packed_into(probability, length, &mut rng, &mut output).unwrap();
            assert_eq!(
                output,
                pack(&bernoulli_stream(probability, length, &mut reference)).data
            );
            assert_eq!(rng.random::<u64>(), reference.random::<u64>());
        }
    }
}

#[test]
fn simd_caller_output_matches_byte_sampling_and_rng_position() {
    for length in [0_usize, 1, 63, 64, 65, 129, 1023, 1024, 1025, 1089] {
        for probability in [
            f64::NEG_INFINITY,
            -1.0,
            0.0,
            1.0 / 256.0,
            0.5,
            1.0,
            2.0,
            f64::INFINITY,
            f64::NAN,
        ] {
            let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
            let mut reference = rng.clone();
            let mut output = vec![u64::MAX; length.div_ceil(64)];
            bernoulli_packed_simd_into(probability, length, &mut rng, &mut output).unwrap();
            assert_eq!(
                output,
                bernoulli_packed_fast(probability, length, &mut reference)
            );
            assert_eq!(rng.random::<u64>(), reference.random::<u64>());
        }
    }
}

#[test]
fn caller_shape_refusal_preserves_rng_and_every_output_word() {
    for length in [0_usize, 1, 65, 1025] {
        for size in [0, 1, 2, 17, 18] {
            if size == length.div_ceil(64) {
                continue;
            }
            let before = vec![0xACE1_u64; size];
            let mut output = before.clone();
            let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
            let mut reference = rng.clone();
            let error = "packed output length must equal ceil(input length / 64)";
            assert_eq!(
                bernoulli_packed_into(0.5, length, &mut rng, &mut output),
                Err(error)
            );
            assert_eq!(output, before);
            assert_eq!(rng.random::<u64>(), reference.random::<u64>());
            assert_eq!(
                bernoulli_packed_simd_into(0.5, length, &mut rng, &mut output),
                Err(error)
            );
            assert_eq!(output, before);
            assert_eq!(rng.random::<u64>(), reference.random::<u64>());
        }
    }
}

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
