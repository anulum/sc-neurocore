// SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Hardware fault injection for robustness testing

//! Hardware fault injection for robustness testing.

use rand::{RngExt, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

/// Flip random bits in packed u64 words with given probability per bit.
pub fn inject_bitflips(data: &mut [u64], rate: f64, seed: u64) {
    if rate <= 0.0 {
        return;
    }
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    for word in data.iter_mut() {
        let mut flip_mask = 0u64;
        for bit in 0..64 {
            if rng.random::<f64>() < rate {
                flip_mask |= 1u64 << bit;
            }
        }
        *word ^= flip_mask;
    }
}

/// Force bits to a fixed value with given probability per bit.
pub fn inject_stuck_at(data: &mut [u64], rate: f64, value: bool, seed: u64) {
    if rate <= 0.0 {
        return;
    }
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    for word in data.iter_mut() {
        for bit in 0..64 {
            if rng.random::<f64>() < rate {
                if value {
                    *word |= 1u64 << bit;
                } else {
                    *word &= !(1u64 << bit);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_rate_no_change() {
        let mut data = vec![0xDEAD_BEEF_CAFE_BABEu64; 4];
        let original = data.clone();
        inject_bitflips(&mut data, 0.0, 42);
        assert_eq!(data, original);
    }

    #[test]
    fn full_rate_flips_all() {
        let mut data = vec![0u64; 2];
        inject_bitflips(&mut data, 1.0, 42);
        assert_eq!(data, vec![u64::MAX; 2]);
    }

    #[test]
    fn stuck_at_zero() {
        let mut data = vec![u64::MAX; 2];
        inject_stuck_at(&mut data, 1.0, false, 42);
        assert_eq!(data, vec![0u64; 2]);
    }

    #[test]
    fn stuck_at_one() {
        let mut data = vec![0u64; 2];
        inject_stuck_at(&mut data, 1.0, true, 42);
        assert_eq!(data, vec![u64::MAX; 2]);
    }

    #[test]
    fn partial_rate_changes_some() {
        let mut data = vec![0u64; 8];
        inject_bitflips(&mut data, 0.5, 99);
        let total_set: u32 = data.iter().map(|w| w.count_ones()).sum();
        // ~50% of 512 bits = ~256 ± reasonable margin
        assert!(total_set > 100 && total_set < 400);
    }
}
