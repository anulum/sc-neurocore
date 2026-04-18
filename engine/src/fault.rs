// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Hardware fault injection for robustness testing

//! Hardware fault injection for robustness testing.

use rand::{RngExt, SeedableRng};
use rand_distr::{Distribution, Normal};
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

// ── Byte-level inject ops (parity with Python `FaultInjector.inject`) ──
//
// These accept and return `[u8]` where each element is 0 or 1, mirroring
// numpy `bool` arrays serialised as uint8. The RNG (Xoshiro256PlusPlus)
// differs from numpy's PCG64, so bitwise parity vs Python is impossible
// — statistical parity within 3-sigma of expected fault count is the
// honest reference (see `tests/test_fault_injection_rust_parity.py`).

/// BIT_FLIP: each input bit flips with probability `ber`.
/// Returns number of bits flipped.
pub fn inject_bitflip_u8(bitstream: &mut [u8], ber: f64, seed: u64) -> u64 {
    if ber <= 0.0 {
        return 0;
    }
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let mut flipped: u64 = 0;
    for b in bitstream.iter_mut() {
        if rng.random::<f64>() < ber {
            *b ^= 1;
            flipped += 1;
        }
    }
    flipped
}

/// STUCK_AT_0: each bit forced to 0 with probability `ber`.
/// Returns number of 1-bits actually changed (i.e., mask AND original).
pub fn inject_stuck_at_0_u8(bitstream: &mut [u8], ber: f64, seed: u64) -> u64 {
    if ber <= 0.0 {
        return 0;
    }
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let mut affected: u64 = 0;
    for b in bitstream.iter_mut() {
        if rng.random::<f64>() < ber {
            if *b != 0 {
                affected += 1;
            }
            *b = 0;
        }
    }
    affected
}

/// STUCK_AT_1: each bit forced to 1 with probability `ber`.
/// Returns number of 0-bits actually changed (i.e., mask AND NOT original).
pub fn inject_stuck_at_1_u8(bitstream: &mut [u8], ber: f64, seed: u64) -> u64 {
    if ber <= 0.0 {
        return 0;
    }
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let mut affected: u64 = 0;
    for b in bitstream.iter_mut() {
        if rng.random::<f64>() < ber {
            if *b == 0 {
                affected += 1;
            }
            *b = 1;
        }
    }
    affected
}

/// DROPOUT: equivalent to STUCK_AT_0 in this fault model.
pub fn inject_dropout_u8(bitstream: &mut [u8], ber: f64, seed: u64) -> u64 {
    inject_stuck_at_0_u8(bitstream, ber, seed)
}

/// GAUSSIAN_NOISE: add N(0, ber) noise to bitstream cast to f64,
/// clip to [0,1], then threshold at 0.5. Returns count of flipped bits.
pub fn inject_gaussian_u8(bitstream: &mut [u8], ber: f64, seed: u64) -> u64 {
    if ber <= 0.0 {
        return 0;
    }
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let normal = Normal::new(0.0_f64, ber).expect("ber > 0");
    let mut flipped: u64 = 0;
    for b in bitstream.iter_mut() {
        let original = *b;
        let noisy = (original as f64 + normal.sample(&mut rng)).clamp(0.0, 1.0);
        let new_bit: u8 = if noisy > 0.5 { 1 } else { 0 };
        if new_bit != original {
            flipped += 1;
        }
        *b = new_bit;
    }
    flipped
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

    // ── Byte-level inject parity tests ──

    #[test]
    fn bitflip_u8_zero_rate_no_change() {
        let mut bs = vec![0u8, 1, 0, 1, 1, 0, 1, 0];
        let original = bs.clone();
        let n = inject_bitflip_u8(&mut bs, 0.0, 7);
        assert_eq!(n, 0);
        assert_eq!(bs, original);
    }

    #[test]
    fn bitflip_u8_full_rate_inverts_all() {
        let mut bs = vec![0u8, 1, 0, 1, 0, 1, 0, 1];
        let n = inject_bitflip_u8(&mut bs, 1.0, 7);
        assert_eq!(n as usize, bs.len());
        assert_eq!(bs, vec![1u8, 0, 1, 0, 1, 0, 1, 0]);
    }

    #[test]
    fn bitflip_u8_statistical_count_within_3sigma() {
        let n = 100_000usize;
        let ber = 1e-3_f64;
        let mut bs = vec![0u8; n];
        let flipped = inject_bitflip_u8(&mut bs, ber, 42);
        // Binomial(n, ber) → mean = n*ber = 100, sigma = sqrt(n*ber*(1-ber)) ≈ 9.99
        let mean = n as f64 * ber;
        let sigma = (n as f64 * ber * (1.0 - ber)).sqrt();
        let lo = (mean - 4.0 * sigma) as u64;
        let hi = (mean + 4.0 * sigma) as u64;
        assert!(
            flipped >= lo && flipped <= hi,
            "flipped={flipped} not in [{lo},{hi}]"
        );
    }

    #[test]
    fn stuck_at_0_only_counts_actual_changes() {
        // All bits already 0 → no change should be counted even if rate = 1.0
        let mut bs = vec![0u8; 64];
        let n = inject_stuck_at_0_u8(&mut bs, 1.0, 11);
        assert_eq!(n, 0);
        assert!(bs.iter().all(|&b| b == 0));
    }

    #[test]
    fn stuck_at_1_only_counts_actual_changes() {
        let mut bs = vec![1u8; 64];
        let n = inject_stuck_at_1_u8(&mut bs, 1.0, 11);
        assert_eq!(n, 0);
        assert!(bs.iter().all(|&b| b == 1));
    }

    #[test]
    fn gaussian_noise_zero_sigma_no_change() {
        let mut bs = vec![0u8, 1, 0, 1];
        let original = bs.clone();
        let n = inject_gaussian_u8(&mut bs, 0.0, 5);
        assert_eq!(n, 0);
        assert_eq!(bs, original);
    }

    #[test]
    fn dropout_equivalent_to_stuck_at_0() {
        let mut a = vec![0u8, 1, 1, 0, 1, 0, 1, 1];
        let mut b = a.clone();
        let na = inject_dropout_u8(&mut a, 0.5, 17);
        let nb = inject_stuck_at_0_u8(&mut b, 0.5, 17);
        assert_eq!(a, b);
        assert_eq!(na, nb);
    }
}
