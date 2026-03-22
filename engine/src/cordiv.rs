// SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — CORDIV stochastic division (Li et al. 2014)

//! CORDIV: stochastic computing divider.
//!
//! Sequential circuit: P(z=1) ≈ P(x=1) / P(y=1) when P(x) <= P(y).
//!
//! Truth table per bit:
//!   x=1         → z = 1
//!   x=0, y=1    → z = 0
//!   x=0, y=0    → z = z_prev (hold)

/// Compute SC division on unpacked bitstreams.
/// Returns quotient bitstream of same length.
pub fn cordiv(numerator: &[u8], denominator: &[u8]) -> Vec<u8> {
    let len = numerator.len().min(denominator.len());
    let mut out = vec![0u8; len];
    let mut prev = 0u8;
    for t in 0..len {
        if numerator[t] == 1 {
            out[t] = 1;
        } else if denominator[t] == 1 {
            out[t] = 0;
        } else {
            out[t] = prev;
        }
        prev = out[t];
    }
    out
}

/// Compute SC division on packed u64 bitstreams (bit-serial).
/// Each u64 word contains 64 sequential bits.
pub fn cordiv_packed(num_packed: &[u64], den_packed: &[u64], length: usize) -> Vec<u64> {
    let n_words = num_packed.len().min(den_packed.len());
    let mut out = vec![0u64; n_words];
    let mut prev_bit = 0u8;

    let total_bits = length.min(n_words * 64);
    for bit_idx in 0..total_bits {
        let word = bit_idx / 64;
        let bit = bit_idx % 64;
        let x = ((num_packed[word] >> bit) & 1) as u8;
        let y = ((den_packed[word] >> bit) & 1) as u8;

        let z = if x == 1 {
            1u8
        } else if y == 1 {
            0u8
        } else {
            prev_bit
        };

        if z == 1 {
            out[word] |= 1u64 << bit;
        }
        prev_bit = z;
    }
    out
}

/// Compute adaptive bitstream length via Hoeffding bound.
/// Returns minimum length (rounded up to power of 2) for target precision.
pub fn adaptive_length_hoeffding(epsilon: f64, confidence: f64) -> usize {
    let delta = 1.0 - confidence;
    if delta <= 0.0 || epsilon <= 0.0 {
        return 65536;
    }
    let l = -(delta / 2.0).ln() / (2.0 * epsilon * epsilon);
    let l_int = l.ceil() as usize;
    let l_int = l_int.max(64);
    // Round up to power of 2
    let mut p = 1usize;
    while p < l_int {
        p *= 2;
    }
    p.min(65536)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cordiv_half_div_one() {
        // x = alternating 0,1 (p=0.5), y = all 1s (p=1.0)
        // result should be ~0.5
        let x: Vec<u8> = (0..1000).map(|i| if i % 2 == 0 { 1 } else { 0 }).collect();
        let y = vec![1u8; 1000];
        let z = cordiv(&x, &y);
        let p: f64 = z.iter().map(|&b| b as f64).sum::<f64>() / z.len() as f64;
        assert!((p - 0.5).abs() < 0.05, "Expected ~0.5, got {p}");
    }

    #[test]
    fn test_cordiv_zero_numerator() {
        let x = vec![0u8; 100];
        let y = vec![1u8; 100];
        let z = cordiv(&x, &y);
        assert!(z.iter().all(|&b| b == 0));
    }

    #[test]
    fn test_cordiv_identity() {
        // x/x should be ~1.0
        let x: Vec<u8> = (0..1000).map(|i| if i % 3 != 0 { 1 } else { 0 }).collect();
        let z = cordiv(&x, &x);
        let p: f64 = z.iter().map(|&b| b as f64).sum::<f64>() / z.len() as f64;
        assert!(p > 0.9, "Expected ~1.0, got {p}");
    }

    #[test]
    fn test_adaptive_length_hoeffding() {
        let l = adaptive_length_hoeffding(0.01, 0.95);
        assert!(l >= 64);
        assert!(l <= 65536);
        assert!(l & (l - 1) == 0); // power of 2
    }

    #[test]
    fn test_adaptive_tighter_needs_more() {
        let l_coarse = adaptive_length_hoeffding(0.1, 0.95);
        let l_fine = adaptive_length_hoeffding(0.01, 0.95);
        assert!(l_fine > l_coarse);
    }
}
