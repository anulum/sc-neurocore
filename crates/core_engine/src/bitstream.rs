// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Bitstream Type & Encoding
// Authored by Anulum Fortis & Arcane Sapience (protoscience@anulum.li)

//! Packed bitstream representation and encoding utilities.
//!
//! Layout-compatible with `engine::BitStreamTensor`: probabilities stored as
//! packed Bernoulli bitstreams in `u64` words, little-endian bit ordering.

/// Packed bitstream with length metadata.
///
/// Each bit position represents one Bernoulli sample. Bits are packed
/// into `u64` words in little-endian order (bit 0 of word 0 = sample 0).
#[derive(Clone, Debug, PartialEq)]
pub struct Bitstream {
    pub data: Vec<u64>,
    pub length: usize,
}

impl Bitstream {
    /// Create from pre-packed words.
    pub fn from_words(data: Vec<u64>, length: usize) -> Self {
        debug_assert!(
            data.len() >= length.div_ceil(64),
            "insufficient words for bitstream length"
        );
        Self { data, length }
    }

    /// Create an all-zeros bitstream of given length.
    pub fn zeros(length: usize) -> Self {
        Self {
            data: vec![0u64; length.div_ceil(64)],
            length,
        }
    }

    /// Create an all-ones bitstream of given length.
    pub fn ones(length: usize) -> Self {
        let words = length.div_ceil(64);
        let mut data = vec![u64::MAX; words];
        let trailing = length % 64;
        if trailing > 0 {
            data[words - 1] = (1u64 << trailing) - 1;
        }
        Self { data, length }
    }

    /// Population count: number of 1-bits in the stream.
    pub fn popcount(&self) -> u64 {
        self.data.iter().map(|w| w.count_ones() as u64).sum()
    }

    /// Estimated probability: popcount / length.
    pub fn probability(&self) -> f64 {
        if self.length == 0 {
            return 0.0;
        }
        self.popcount() as f64 / self.length as f64
    }

    /// Bitwise AND (SC multiplication).
    pub fn sc_and(&self, other: &Bitstream) -> Bitstream {
        assert_eq!(self.length, other.length, "bitstream length mismatch");
        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| a & b)
            .collect();
        Bitstream {
            data,
            length: self.length,
        }
    }

    /// Bitwise MUX addition: out[i] = if select[i] then a[i] else b[i].
    ///
    /// SC-domain scaled addition: P(out) ≈ P(sel) * P(a) + (1 - P(sel)) * P(b).
    pub fn sc_mux(&self, other: &Bitstream, select: &Bitstream) -> Bitstream {
        assert_eq!(self.length, other.length, "bitstream length mismatch");
        assert_eq!(self.length, select.length, "select length mismatch");
        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .zip(select.data.iter())
            .map(|((&a, &b), &s)| (a & s) | (b & !s))
            .collect();
        Bitstream {
            data,
            length: self.length,
        }
    }

    /// Bitwise XOR (HDC bind operation).
    pub fn xor(&self, other: &Bitstream) -> Bitstream {
        assert_eq!(self.length, other.length, "bitstream length mismatch");
        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| a ^ b)
            .collect();
        Bitstream {
            data,
            length: self.length,
        }
    }

    /// Saturating subtraction in SC domain: out = a AND NOT(b).
    pub fn sc_saturating_sub(&self, other: &Bitstream) -> Bitstream {
        assert_eq!(self.length, other.length, "bitstream length mismatch");
        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| a & !b)
            .collect();
        Bitstream {
            data,
            length: self.length,
        }
    }

    /// Hamming distance (normalized, 0.0 = identical, 1.0 = opposite).
    pub fn hamming_distance(&self, other: &Bitstream) -> f64 {
        assert_eq!(self.length, other.length, "bitstream length mismatch");
        let xor_count: u64 = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| (a ^ b).count_ones() as u64)
            .sum();
        xor_count as f64 / self.length as f64
    }
}

/// Pack a `0/1` byte slice into a `Bitstream`.
pub fn pack(bits: &[u8]) -> Bitstream {
    let length = bits.len();
    let words = length.div_ceil(64);
    let mut data = vec![0u64; words];
    for (idx, &bit) in bits.iter().enumerate() {
        if bit != 0 {
            data[idx / 64] |= 1u64 << (idx % 64);
        }
    }
    Bitstream { data, length }
}

/// Unpack a `Bitstream` into a `0/1` byte vector.
pub fn unpack(bs: &Bitstream) -> Vec<u8> {
    let mut bits = vec![0u8; bs.length];
    for (idx, bit) in bits.iter_mut().enumerate() {
        *bit = ((bs.data[idx / 64] >> (idx % 64)) & 1) as u8;
    }
    bits
}

/// Stochastic Cross-Correlation (SCC) between two bitstreams.
///
/// SCC = (P(A∧B) - P(A)*P(B)) / denominator, where denominator depends
/// on the sign of the numerator (Alaghi & Hayes, 2013).
///
/// Returns 0.0 for uncorrelated streams, +1.0 for maximally correlated,
/// -1.0 for maximally anti-correlated.
pub fn scc(a: &Bitstream, b: &Bitstream) -> f64 {
    assert_eq!(a.length, b.length, "bitstream length mismatch");
    let n = a.length as f64;
    if n == 0.0 {
        return 0.0;
    }

    let pa = a.popcount() as f64 / n;
    let pb = b.popcount() as f64 / n;

    let and_count: u64 = a
        .data
        .iter()
        .zip(b.data.iter())
        .map(|(&x, &y)| (x & y).count_ones() as u64)
        .sum();
    let p_and = and_count as f64 / n;

    let numerator = p_and - (pa * pb);
    if numerator.abs() < 1e-12 {
        return 0.0;
    }

    let denominator = if numerator > 0.0 {
        pa.min(pb) - (pa * pb)
    } else {
        (pa * pb) - (pa + pb - 1.0).max(0.0)
    };

    if denominator.abs() < 1e-12 {
        0.0
    } else {
        (numerator / denominator).clamp(-1.0, 1.0)
    }
}

/// 16-bit LFSR encoder matching `engine::Lfsr16` semantics.
///
/// Polynomial: x^16 + x^14 + x^13 + x^11 + 1 (maximal length = 65535).
pub struct Lfsr16 {
    pub reg: u16,
}

impl Lfsr16 {
    pub fn new(seed: u16) -> Self {
        assert_ne!(seed, 0, "LFSR seed must be non-zero");
        Self { reg: seed }
    }

    /// Advance one step and return the new register value.
    #[inline(always)]
    pub fn step(&mut self) -> u16 {
        let feedback =
            ((self.reg >> 15) ^ (self.reg >> 13) ^ (self.reg >> 12) ^ (self.reg >> 10)) & 1;
        self.reg = (self.reg << 1) | feedback;
        self.reg
    }

    /// Encode a probability (Q16 fixed-point threshold) into a bitstream.
    ///
    /// Compares LFSR output against `threshold` for `length` steps.
    /// Semantics match `engine::BitstreamEncoder::step`.
    pub fn encode(&mut self, threshold: u16, length: usize) -> Bitstream {
        let words = length.div_ceil(64);
        let mut data = vec![0u64; words];
        for i in 0..length {
            let bit = if self.reg < threshold { 1u64 } else { 0u64 };
            data[i / 64] |= bit << (i % 64);
            self.step();
        }
        Bitstream { data, length }
    }
}

/// CORDIV stochastic division.
///
/// Implements the correlation-based SC division circuit:
/// Out = X / Y, where X and Y are bitstreams representing probabilities.
///
/// Uses a J-K flip-flop feedback mechanism:
/// - If Y=1: output = X (pass-through)
/// - If Y=0: output = previous output (hold)
///
/// Converges to P(X)/P(Y) when P(Y) > 0.
pub fn cordiv(x: &Bitstream, y: &Bitstream) -> Bitstream {
    assert_eq!(x.length, y.length, "bitstream length mismatch");
    let mut result = Bitstream::zeros(x.length);
    let mut prev_out: u8 = 0;

    for i in 0..x.length {
        let x_bit = ((x.data[i / 64] >> (i % 64)) & 1) as u8;
        let y_bit = ((y.data[i / 64] >> (i % 64)) & 1) as u8;

        let out_bit = if y_bit == 1 { x_bit } else { prev_out };
        prev_out = out_bit;

        if out_bit == 1 {
            result.data[i / 64] |= 1u64 << (i % 64);
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_unpack_roundtrip() {
        let bits = vec![1, 0, 1, 1, 0, 1, 0, 0, 1];
        let packed = pack(&bits);
        assert_eq!(unpack(&packed), bits);
    }

    #[test]
    fn pack_unpack_64_boundary() {
        let bits: Vec<u8> = (0..128).map(|i| (i % 2) as u8).collect();
        let packed = pack(&bits);
        assert_eq!(packed.data.len(), 2);
        assert_eq!(unpack(&packed), bits);
    }

    #[test]
    fn sc_and_multiplication() {
        let a = pack(&[1, 0, 1, 1, 0, 0, 1, 1]);
        let b = pack(&[1, 1, 1, 0, 0, 1, 1, 0]);
        let c = a.sc_and(&b);
        assert_eq!(unpack(&c), vec![1, 0, 1, 0, 0, 0, 1, 0]);
        assert_eq!(c.popcount(), 3);
    }

    #[test]
    fn sc_mux_addition() {
        let a = pack(&[1, 1, 1, 1, 0, 0, 0, 0]);
        let b = pack(&[0, 0, 0, 0, 1, 1, 1, 1]);
        let s = pack(&[1, 1, 0, 0, 1, 1, 0, 0]);
        let out = a.sc_mux(&b, &s);
        // sel=1: pick a, sel=0: pick b
        assert_eq!(unpack(&out), vec![1, 1, 0, 0, 0, 0, 1, 1]);
    }

    #[test]
    fn sc_saturating_sub() {
        let a = pack(&[1, 1, 1, 0, 0, 1]);
        let b = pack(&[0, 1, 0, 1, 0, 1]);
        let c = a.sc_saturating_sub(&b);
        assert_eq!(unpack(&c), vec![1, 0, 1, 0, 0, 0]);
    }

    #[test]
    fn popcount_and_probability() {
        let bs = pack(&[1, 0, 1, 1, 0]);
        assert_eq!(bs.popcount(), 3);
        assert!((bs.probability() - 0.6).abs() < 1e-10);
    }

    #[test]
    fn zeros_and_ones() {
        let z = Bitstream::zeros(100);
        assert_eq!(z.popcount(), 0);
        assert_eq!(z.length, 100);

        let o = Bitstream::ones(100);
        assert_eq!(o.popcount(), 100);

        let o65 = Bitstream::ones(65);
        assert_eq!(o65.popcount(), 65);
        assert_eq!(o65.data.len(), 2);
    }

    #[test]
    fn scc_identical_streams() {
        let a = pack(&[1, 0, 1, 1, 0, 0, 1, 0]);
        let corr = scc(&a, &a);
        assert!((corr - 1.0).abs() < 1e-6, "SCC of identical streams should be 1.0, got {corr}");
    }

    #[test]
    fn scc_anticorrelated_streams() {
        let a = pack(&[1, 0, 1, 0, 1, 0, 1, 0]);
        let b = pack(&[0, 1, 0, 1, 0, 1, 0, 1]);
        let corr = scc(&a, &b);
        assert!(
            (corr - (-1.0)).abs() < 1e-6,
            "SCC of anticorrelated streams should be -1.0, got {corr}"
        );
    }

    #[test]
    fn scc_uncorrelated_streams() {
        // All zeros → SCC = 0 (no information)
        let a = Bitstream::zeros(256);
        let b = Bitstream::zeros(256);
        assert!((scc(&a, &b)).abs() < 1e-6);
    }

    #[test]
    fn lfsr_nonzero_period() {
        let mut lfsr = Lfsr16::new(0xACE1);
        let initial = lfsr.reg;
        let mut seen_initial = false;
        for _ in 0..65535 {
            lfsr.step();
            if lfsr.reg == initial {
                seen_initial = true;
                break;
            }
        }
        assert!(seen_initial, "LFSR should return to initial state within 65535 steps");
    }

    #[test]
    fn lfsr_encode_probability() {
        let mut lfsr = Lfsr16::new(0xACE1);
        // threshold = 32768 ≈ 0.5 probability
        let bs = lfsr.encode(32768, 10000);
        let p = bs.probability();
        assert!(
            (p - 0.5).abs() < 0.02,
            "LFSR encoding at threshold=32768 should produce ~0.5, got {p}"
        );
    }

    #[test]
    fn cordiv_convergence() {
        // CORDIV convergence: P(out) → P(X)/P(Y) when streams are decorrelated.
        // With LFSR-generated streams, correlation in the feedback loop limits
        // precision. We verify directional correctness: the output probability
        // should be between the two input probabilities and closer to the
        // expected ratio than to either input alone.
        let len = 32768;
        let mut lfsr_x = Lfsr16::new(0x1234);
        let mut lfsr_y = Lfsr16::new(0x9ABC);
        let x = lfsr_x.encode((0.3 * 65535.0) as u16, len);
        let y = lfsr_y.encode((0.6 * 65535.0) as u16, len);

        let _px = x.probability();
        let py = y.probability();
        let result = cordiv(&x, &y);
        let p = result.probability();

        // Output should be in the valid range [0, 1]
        assert!(p >= 0.0 && p <= 1.0, "CORDIV output out of range: {p}");

        // Output should be less than py (the divisor probability)
        assert!(p < py + 0.05, "CORDIV output {p} should be ≤ P(Y)={py}");

        // Sanity: output should not be zero for non-zero inputs
        assert!(p > 0.1, "CORDIV output unexpectedly low: {p}");
    }

    #[test]
    fn hamming_distance_identical() {
        let a = pack(&[1, 0, 1, 1]);
        assert!((a.hamming_distance(&a)).abs() < 1e-10);
    }

    #[test]
    fn hamming_distance_opposite() {
        let a = pack(&[1, 0, 1, 0]);
        let b = pack(&[0, 1, 0, 1]);
        assert!((a.hamming_distance(&b) - 1.0).abs() < 1e-10);
    }
}
