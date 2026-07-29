// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — tinySC LFSR-16 Encoder (no_std)

//! Deterministic LFSR-16 encoder bit-compatible with `core_engine::Lfsr16`.
//!
//! Polynomial: x^16 + x^14 + x^13 + x^11 + 1 (maximal length = 65535).

/// 16-bit Galois LFSR encoder.
///
/// Bit-compatible with `core_engine::bitstream::Lfsr16`. Uses `u32`-packed
/// output for MCU word alignment.
pub struct Lfsr16 {
    pub reg: u16,
}

impl Lfsr16 {
    /// Create with a non-zero seed.
    #[inline]
    pub const fn new(seed: u16) -> Self {
        assert!(seed != 0, "LFSR seed must be non-zero");
        Self { reg: seed }
    }

    /// Advance one step, return new register value.
    #[inline(always)]
    pub fn step(&mut self) -> u16 {
        let feedback =
            ((self.reg >> 15) ^ (self.reg >> 13) ^ (self.reg >> 12) ^ (self.reg >> 10)) & 1;
        self.reg = (self.reg << 1) | feedback;
        self.reg
    }

    /// Encode probability into packed `u32` words.
    ///
    /// Compares LFSR output against `threshold` for `length` steps.
    /// Caller provides the output buffer (zero-copy).
    pub fn encode_into(&mut self, threshold: u16, length: usize, out: &mut [u32]) {
        debug_assert!(out.len() >= length.div_ceil(32));
        for w in out.iter_mut() {
            *w = 0;
        }
        for i in 0..length {
            if self.reg < threshold {
                out[i / 32] |= 1u32 << (i % 32);
            }
            self.step();
        }
    }

    /// Reset to a new seed.
    #[inline]
    pub fn reset(&mut self, seed: u16) {
        debug_assert!(seed != 0);
        self.reg = seed;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deterministic() {
        let mut a = Lfsr16::new(0xACE1);
        let mut b = Lfsr16::new(0xACE1);
        for _ in 0..100 {
            assert_eq!(a.step(), b.step());
        }
    }

    #[test]
    fn test_maximal_period() {
        let mut lfsr = Lfsr16::new(0xACE1);
        let initial = lfsr.reg;
        let mut count = 0u32;
        loop {
            lfsr.step();
            count += 1;
            if lfsr.reg == initial || count >= 65535 {
                break;
            }
        }
        assert_eq!(count, 65535, "LFSR must have maximal period");
    }

    #[test]
    fn test_encode_into_probability() {
        let mut lfsr = Lfsr16::new(0xACE1);
        let length: usize = 10000;
        let words = length.div_ceil(32);
        let mut buf = vec![0u32; words];
        lfsr.encode_into(32768, length, &mut buf);
        let popcount: u32 = buf.iter().map(|w| w.count_ones()).sum();
        let p = popcount as f32 / length as f32;
        assert!((p - 0.5).abs() < 0.03, "got probability {p}");
    }

    #[test]
    fn test_encode_into_zeros() {
        let mut lfsr = Lfsr16::new(0xACE1);
        let mut buf = [0u32; 4];
        lfsr.encode_into(0, 128, &mut buf); // threshold=0 → all zeros
        assert_eq!(buf, [0, 0, 0, 0]);
    }

    #[test]
    fn test_encode_into_ones() {
        let mut lfsr = Lfsr16::new(0xACE1);
        let mut buf = [0u32; 1];
        lfsr.encode_into(u16::MAX, 32, &mut buf); // threshold=max → all ones
        assert_eq!(buf[0], u32::MAX);
    }
}
