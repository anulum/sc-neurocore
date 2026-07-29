// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — tinySC Sobol Decorrelator (no_std)

//! Sobol low-discrepancy sequence generator for SC bitstream decorrelation.
//!
//! Provides better uniformity than LFSR-16 at the cost of slightly more
//! compute per step. Uses Gray-code acceleration for O(1) per-sample
//! generation (no matrix multiply needed).

/// Sobol sequence generator (1D, direction numbers for 16-bit).
///
/// Uses 16 direction numbers for full 16-bit resolution.
/// Gray code indexing: only XOR one direction number per step.
pub struct Sobol16 {
    reg: u16,
    index: u32,
    direction: [u16; 16],
}

impl Sobol16 {
    /// Create a new Sobol generator with default direction numbers.
    ///
    /// Direction numbers are from Joe-Kuo tables (dimension 1).
    pub const fn new() -> Self {
        Self {
            reg: 0,
            index: 0,
            direction: [
                0x8000, 0x4000, 0x2000, 0x1000, 0x0800, 0x0400, 0x0200, 0x0100, 0x0080, 0x0040,
                0x0020, 0x0010, 0x0008, 0x0004, 0x0002, 0x0001,
            ],
        }
    }

    /// Create with a scrambling seed (Owen scrambling approximation).
    pub const fn with_seed(seed: u16) -> Self {
        let mut s = Self::new();
        s.reg = seed;
        s
    }

    /// Advance by one step, return the next Sobol value in [0, 65535].
    #[inline]
    pub fn step(&mut self) -> u16 {
        let c = self.index.trailing_zeros() as usize;
        if c < 16 {
            self.reg ^= self.direction[c];
        }
        self.index += 1;
        self.reg
    }

    /// Encode a probability into packed u32 words using Sobol sequence.
    ///
    /// Compare sequence values against threshold for each bit position.
    pub fn encode_into(&mut self, threshold: u16, length: usize, out: &mut [u32]) {
        debug_assert!(out.len() >= length.div_ceil(32));
        for w in out.iter_mut() {
            *w = 0;
        }
        for i in 0..length {
            let val = self.step();
            if val < threshold {
                out[i / 32] |= 1u32 << (i % 32);
            }
        }
    }

    /// Reset to initial state.
    pub fn reset(&mut self) {
        self.reg = 0;
        self.index = 0;
    }

    /// Reset with a new scrambling seed.
    pub fn reset_with_seed(&mut self, seed: u16) {
        self.reg = seed;
        self.index = 0;
    }
}

impl Default for Sobol16 {
    fn default() -> Self {
        Self::new()
    }
}

/// Adaptive decorrelator that can switch between LFSR and Sobol at runtime.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DecorrelatorType {
    Lfsr,
    Sobol,
}

/// Bitstream encoder that supports runtime switching between decorrelators.
pub struct AdaptiveEncoder {
    lfsr: crate::lfsr::Lfsr16,
    sobol: Sobol16,
    active: DecorrelatorType,
}

impl AdaptiveEncoder {
    pub fn new(lfsr_seed: u16, sobol_seed: u16) -> Self {
        Self {
            lfsr: crate::lfsr::Lfsr16::new(lfsr_seed),
            sobol: Sobol16::with_seed(sobol_seed),
            active: DecorrelatorType::Lfsr,
        }
    }

    /// Switch decorrelator type at runtime.
    pub fn set_type(&mut self, dt: DecorrelatorType) {
        self.active = dt;
    }

    pub const fn active_type(&self) -> DecorrelatorType {
        self.active
    }

    /// Encode a bitstream using the active decorrelator.
    pub fn encode_into(&mut self, threshold: u16, length: usize, out: &mut [u32]) {
        match self.active {
            DecorrelatorType::Lfsr => self.lfsr.encode_into(threshold, length, out),
            DecorrelatorType::Sobol => self.sobol.encode_into(threshold, length, out),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitstream;

    #[test]
    fn test_sobol_deterministic() {
        let mut a = Sobol16::new();
        let mut b = Sobol16::new();
        for _ in 0..100 {
            assert_eq!(a.step(), b.step());
        }
    }

    #[test]
    fn test_sobol_unique_values() {
        let mut s = Sobol16::new();
        let mut seen = [false; 65536];
        let mut unique = 0;
        for _ in 0..1000 {
            let v = s.step() as usize;
            if !seen[v] {
                seen[v] = true;
                unique += 1;
            }
        }
        // Sobol should produce many unique values
        assert!(unique > 500, "only {} unique in 1000 samples", unique);
    }

    #[test]
    fn test_sobol_encode_probability() {
        let mut s = Sobol16::new();
        let length: usize = 10000;
        let words = length.div_ceil(32);
        let mut buf = vec![0u32; words];
        s.encode_into(32768, length, &mut buf); // ~50% threshold
        let popcount = bitstream::popcount_slice(&buf);
        let p = popcount as f32 / length as f32;
        assert!((p - 0.5).abs() < 0.05, "got probability {p}");
    }

    #[test]
    fn test_sobol_with_seed() {
        let mut a = Sobol16::with_seed(0x1234);
        let mut b = Sobol16::with_seed(0x5678);
        let va = a.step();
        let vb = b.step();
        assert_ne!(va, vb, "different seeds should give different values");
    }

    #[test]
    fn test_sobol_reset() {
        let mut s = Sobol16::new();
        let first_10: Vec<u16> = (0..10).map(|_| s.step()).collect();
        s.reset();
        let second_10: Vec<u16> = (0..10).map(|_| s.step()).collect();
        assert_eq!(first_10, second_10);
    }

    #[test]
    fn test_adaptive_encoder_lfsr() {
        let mut enc = AdaptiveEncoder::new(0xACE1, 0);
        let mut buf = [0u32; 4];
        enc.encode_into(32768, 64, &mut buf);
        let pop = bitstream::popcount_slice(&buf);
        assert!(pop > 0);
    }

    #[test]
    fn test_adaptive_encoder_sobol() {
        let mut enc = AdaptiveEncoder::new(0xACE1, 0);
        enc.set_type(DecorrelatorType::Sobol);
        assert_eq!(enc.active_type(), DecorrelatorType::Sobol);
        let mut buf = [0u32; 4];
        enc.encode_into(32768, 64, &mut buf);
        let pop = bitstream::popcount_slice(&buf);
        assert!(pop > 0);
    }

    #[test]
    fn test_adaptive_encoder_switch() {
        let mut enc = AdaptiveEncoder::new(0xACE1, 0x1234);
        let mut buf_lfsr = [0u32; 4];
        let mut buf_sobol = [0u32; 4];

        enc.set_type(DecorrelatorType::Lfsr);
        enc.encode_into(32768, 128, &mut buf_lfsr);

        enc.set_type(DecorrelatorType::Sobol);
        enc.encode_into(32768, 128, &mut buf_sobol);

        // Different decorrelators should produce different bitstreams
        assert_ne!(buf_lfsr, buf_sobol);
    }
}
