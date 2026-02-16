// CopyRight: (c) 1998-2026 Miroslav Sotek. All rights reserved.
// Contact us: www.anulum.li  protoscience@anulum.li
// ORCID: https://orcid.org/0009-0009-3560-0851
// License: GNU AFFERO GENERAL PUBLIC LICENSE v3
// Commercial Licensing: Available

//! # Bitstream Operations
//!
//! Core bitstream packing and logic primitives for stochastic computing.
//! Probabilities are represented as packed Bernoulli bitstreams stored in `u64` words.

use rand::Rng;

use crate::simd;

/// Packed bitstream tensor with original bit length metadata.
#[derive(Clone, Debug)]
pub struct BitStreamTensor {
    /// Packed words containing bitstream data.
    pub data: Vec<u64>,
    /// Original unpacked bit length.
    pub length: usize,
}

impl BitStreamTensor {
    /// Create a tensor from pre-packed words.
    pub fn from_words(data: Vec<u64>, length: usize) -> Self {
        Self { data, length }
    }

    // ── HDC / VSA primitives ─────────────────────────────────────────

    /// HDC BIND: In-place XOR with another tensor.
    pub fn xor_inplace(&mut self, other: &BitStreamTensor) {
        assert_eq!(
            self.length, other.length,
            "Bitstream lengths must match for XOR."
        );
        for (a, b) in self.data.iter_mut().zip(other.data.iter()) {
            *a ^= *b;
        }
    }

    /// HDC BIND: XOR returning a new tensor.
    pub fn xor(&self, other: &BitStreamTensor) -> BitStreamTensor {
        assert_eq!(
            self.length, other.length,
            "Bitstream lengths must match for XOR."
        );
        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| a ^ b)
            .collect();
        BitStreamTensor {
            data,
            length: self.length,
        }
    }

    /// HDC PERMUTE: Cyclic right rotation by `shift` bits.
    ///
    /// Rotates the entire logical bitstream, handling cross-word boundaries.
    pub fn rotate_right(&mut self, shift: usize) {
        if self.length == 0 || shift % self.length == 0 {
            return;
        }
        let mut bits = unpack(self);
        bits.rotate_right(shift % self.length);
        *self = pack(&bits);
    }

    /// HDC SIMILARITY: Normalized Hamming distance (0.0 = identical, 1.0 = opposite).
    pub fn hamming_distance(&self, other: &BitStreamTensor) -> f32 {
        assert_eq!(
            self.length, other.length,
            "Bitstream lengths must match for Hamming distance."
        );
        let xor_count: u64 = crate::simd::fused_xor_popcount_dispatch(&self.data, &other.data);
        xor_count as f32 / self.length as f32
    }

    /// HDC BUNDLE: Majority vote across N tensors.
    ///
    /// Bit is 1 if a strict majority (> N/2) of inputs have it set.
    pub fn bundle(vectors: &[&BitStreamTensor]) -> BitStreamTensor {
        assert!(!vectors.is_empty(), "Cannot bundle zero vectors.");
        let length = vectors[0].length;
        let words = vectors[0].data.len();
        let threshold = vectors.len() / 2; // strict majority = count > N/2

        let mut data = vec![0u64; words];
        for bit_idx in 0..length {
            let word = bit_idx / 64;
            let bit = bit_idx % 64;
            let count: usize = vectors
                .iter()
                .filter(|v| (v.data[word] >> bit) & 1 == 1)
                .count();
            if count > threshold {
                data[word] |= 1u64 << bit;
            }
        }
        BitStreamTensor { data, length }
    }
}

/// Pack a `0/1` byte slice into `u64` words.
pub fn pack(bits: &[u8]) -> BitStreamTensor {
    let length = bits.len();
    let words = length.div_ceil(64);
    let mut data = vec![0_u64; words];

    for (idx, bit) in bits.iter().copied().enumerate() {
        if bit != 0 {
            data[idx / 64] |= 1_u64 << (idx % 64);
        }
    }

    BitStreamTensor { data, length }
}

/// Portable fast pack: processes 8 bytes into one output byte at a time.
pub fn pack_fast(bits: &[u8]) -> BitStreamTensor {
    let length = bits.len();
    let words = length.div_ceil(64);
    let mut data = vec![0_u64; words];

    for (word_idx, word) in data.iter_mut().enumerate() {
        let base = word_idx * 64;
        let chunk = &bits[base..std::cmp::min(base + 64, length)];

        for (byte_idx, byte_chunk) in chunk.chunks(8).enumerate() {
            let mut packed_byte: u8 = 0;
            for (bit_idx, &bit) in byte_chunk.iter().enumerate() {
                packed_byte |= u8::from(bit != 0) << bit_idx;
            }
            *word |= (packed_byte as u64) << (byte_idx * 8);
        }
    }

    BitStreamTensor { data, length }
}

/// Unpack a packed tensor back into a `0/1` byte vector.
pub fn unpack(tensor: &BitStreamTensor) -> Vec<u8> {
    let mut bits = vec![0_u8; tensor.length];

    for (idx, bit) in bits.iter_mut().enumerate().take(tensor.length) {
        let word = tensor.data[idx / 64];
        *bit = ((word >> (idx % 64)) & 1) as u8;
    }

    bits
}

/// Compute bitwise-AND between two packed tensors.
pub fn bitwise_and(a: &BitStreamTensor, b: &BitStreamTensor) -> BitStreamTensor {
    assert_eq!(
        a.length, b.length,
        "Bitstream lengths must match for bitwise AND."
    );
    assert_eq!(
        a.data.len(),
        b.data.len(),
        "Packed bitstream shapes must match for bitwise AND."
    );

    let data = a
        .data
        .iter()
        .zip(b.data.iter())
        .map(|(lhs, rhs)| lhs & rhs)
        .collect();

    BitStreamTensor {
        data,
        length: a.length,
    }
}

/// Portable SWAR popcount for a single `u64` word.
pub fn swar_popcount_word(mut x: u64) -> u64 {
    x = x.wrapping_sub((x >> 1) & 0x5555_5555_5555_5555);
    x = (x & 0x3333_3333_3333_3333) + ((x >> 2) & 0x3333_3333_3333_3333);
    x = (x + (x >> 4)) & 0x0f0f_0f0f_0f0f_0f0f;
    x.wrapping_mul(0x0101_0101_0101_0101) >> 56
}

/// Portable popcount over a packed word slice.
pub fn popcount_words_portable(data: &[u64]) -> u64 {
    data.iter().copied().map(swar_popcount_word).sum()
}

/// Popcount of all bits set in a packed tensor.
pub fn popcount(tensor: &BitStreamTensor) -> u64 {
    popcount_words_portable(&tensor.data)
}

/// Generate an unpacked Bernoulli bitstream from a probability.
///
/// Values are clamped into `[0, 1]` before sampling.
pub fn bernoulli_stream<R: Rng + ?Sized>(prob: f64, length: usize, rng: &mut R) -> Vec<u8> {
    let p = prob.clamp(0.0, 1.0);
    let mut out = vec![0_u8; length];
    for bit in &mut out {
        *bit = if rng.gen::<f64>() < p { 1 } else { 0 };
    }
    out
}

/// Generate a packed Bernoulli bitstream directly into `u64` words.
///
/// This is bit-identical to `pack(&bernoulli_stream(...)).data` for the same
/// RNG state while avoiding the intermediate `Vec<u8>` allocation.
pub fn bernoulli_packed<R: Rng + ?Sized>(prob: f64, length: usize, rng: &mut R) -> Vec<u64> {
    let p = prob.clamp(0.0, 1.0);
    let words = length.div_ceil(64);
    let mut data = vec![0_u64; words];
    for (word_idx, word) in data.iter_mut().enumerate() {
        let bits_in_word = std::cmp::min(64, length.saturating_sub(word_idx * 64));
        for bit in 0..bits_in_word {
            if rng.gen::<f64>() < p {
                *word |= 1_u64 << bit;
            }
        }
    }
    data
}

/// Fast packed Bernoulli generation using byte-threshold comparison.
///
/// Instead of generating one f64 (8 bytes) per bit and comparing,
/// this generates one u8 (1 byte) per bit via `rng.fill()` and
/// compares against a u8 threshold = `(prob * 256.0) as u8`.
///
/// This uses 8x less RNG bandwidth than `bernoulli_packed` at the
/// cost of 8-bit probability resolution (1/256 granularity).
/// For bitstream lengths >= 256, the statistical difference is
/// negligible compared to inherent sampling noise.
///
/// The output is NOT bit-identical to `bernoulli_packed` for the
/// same RNG state.
pub fn bernoulli_packed_fast<R: Rng + ?Sized>(prob: f64, length: usize, rng: &mut R) -> Vec<u64> {
    let threshold = (prob.clamp(0.0, 1.0) * 256.0).min(255.0) as u8;
    let words = length.div_ceil(64);
    let mut data = vec![0_u64; words];
    let mut buf = [0_u8; 64];

    for (word_idx, word) in data.iter_mut().enumerate() {
        let bits_in_word = std::cmp::min(64, length.saturating_sub(word_idx * 64));
        rng.fill(&mut buf[..bits_in_word]);
        for (bit, &rb) in buf[..bits_in_word].iter().enumerate() {
            if rb < threshold {
                *word |= 1_u64 << bit;
            }
        }
    }
    data
}

/// SIMD-accelerated packed Bernoulli generation.
///
/// Semantics match `bernoulli_packed_fast` (byte-threshold sampling) while
/// vectorizing the threshold comparison for full 64-bit words.
pub fn bernoulli_packed_simd<R: Rng + ?Sized>(prob: f64, length: usize, rng: &mut R) -> Vec<u64> {
    let threshold = (prob.clamp(0.0, 1.0) * 256.0).min(255.0) as u8;
    let words = length.div_ceil(64);
    let mut data = vec![0_u64; words];
    let full_words = length / 64;
    let mut buf = [0_u8; 64];

    for word in data.iter_mut().take(full_words) {
        rng.fill(&mut buf);
        *word = simd_bernoulli_compare(&buf, threshold);
    }

    if full_words < words {
        let remaining = length - full_words * 64;
        rng.fill(&mut buf[..remaining]);
        let mut tail = 0_u64;
        for (bit, &rb) in buf[..remaining].iter().enumerate() {
            if rb < threshold {
                tail |= 1_u64 << bit;
            }
        }
        data[full_words] = tail;
    }

    data
}

/// Fused encode+AND+popcount without materializing encoded input words.
///
/// Semantically equivalent to:
/// 1. `encoded = bernoulli_packed_simd(prob, length, rng)`
/// 2. `sum(popcount(encoded[word] & weight_words[word]))`
pub fn encode_and_popcount<R: Rng + ?Sized>(
    weight_words: &[u64],
    prob: f64,
    length: usize,
    rng: &mut R,
) -> u64 {
    let threshold = (prob.clamp(0.0, 1.0) * 256.0).min(255.0) as u8;
    let full_words = length / 64;
    let mut total = 0_u64;
    let mut buf = [0_u8; 64];

    for &w_word in weight_words.iter().take(full_words) {
        rng.fill(&mut buf);
        let encoded = simd_bernoulli_compare(&buf, threshold);
        total += (encoded & w_word).count_ones() as u64;
    }

    let remaining = length.saturating_sub(full_words * 64);
    if remaining > 0 && full_words < weight_words.len() {
        rng.fill(&mut buf[..remaining]);
        let mut encoded = 0_u64;
        for (bit, &rb) in buf[..remaining].iter().enumerate() {
            if rb < threshold {
                encoded |= 1_u64 << bit;
            }
        }
        total += (encoded & weight_words[full_words]).count_ones() as u64;
    }

    total
}

/// Compare 64 bytes against a threshold and return a packed bit mask.
#[inline]
fn simd_bernoulli_compare(buf: &[u8], threshold: u8) -> u64 {
    debug_assert!(buf.len() >= 64, "buffer must contain at least 64 bytes");

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512bw") {
            // SAFETY: Runtime feature-gated.
            return unsafe { simd::avx512::bernoulli_compare_avx512(buf, threshold) };
        }
        if is_x86_feature_detected!("avx2") {
            // SAFETY: Runtime feature-gated.
            let lo = unsafe { simd::avx2::bernoulli_compare_avx2(&buf[0..32], threshold) };
            // SAFETY: Runtime feature-gated.
            let hi = unsafe { simd::avx2::bernoulli_compare_avx2(&buf[32..64], threshold) };
            return (lo as u64) | ((hi as u64) << 32);
        }
    }

    let mut mask = 0_u64;
    for (bit, &rb) in buf.iter().take(64).enumerate() {
        if rb < threshold {
            mask |= 1_u64 << bit;
        }
    }
    mask
}

/// Encode a flat matrix of probabilities into packed Bernoulli bitstreams.
///
/// Each value is clamped into `[0, 1]` before sampling.
pub fn encode_matrix_prob_to_packed<R: Rng + ?Sized>(
    values: &[f64],
    rows: usize,
    cols: usize,
    length: usize,
    words: usize,
    rng: &mut R,
) -> Vec<Vec<u64>> {
    let mut packed = Vec::with_capacity(rows * cols);
    for value in values.iter().take(rows * cols) {
        let mut row = bernoulli_packed(*value, length, rng);
        row.resize(words, 0);
        packed.push(row);
    }
    packed
}

#[cfg(test)]
mod tests {
    use super::{
        bernoulli_packed, bernoulli_packed_fast, bernoulli_packed_simd, bernoulli_stream,
        bitwise_and, encode_and_popcount, pack, pack_fast, popcount, unpack,
    };

    #[test]
    fn pack_unpack_roundtrip() {
        let bits = vec![1, 0, 1, 1, 0, 1, 0, 0, 1];
        let packed = pack(&bits);
        let unpacked = unpack(&packed);
        assert_eq!(bits, unpacked);
    }

    #[test]
    fn pack_fast_matches_pack() {
        let cases = [0_usize, 1, 7, 8, 9, 63, 64, 65, 127, 128, 256, 1025];
        for length in cases {
            let bits: Vec<u8> = (0..length).map(|i| ((i * 7 + 3) % 2) as u8).collect();
            let slow = pack(&bits);
            let fast = pack_fast(&bits);
            assert_eq!(fast.length, slow.length);
            assert_eq!(fast.data, slow.data, "Mismatch at length={length}");
        }
    }

    #[test]
    fn pack_fast_roundtrip() {
        let bits: Vec<u8> = (0..2048).map(|i| ((i * 5 + 1) % 2) as u8).collect();
        let packed = pack_fast(&bits);
        let unpacked = unpack(&packed);
        assert_eq!(bits, unpacked);
    }

    #[test]
    fn and_and_popcount() {
        let a = pack(&[1, 0, 1, 1, 0, 0, 1, 1]);
        let b = pack(&[1, 1, 1, 0, 0, 1, 1, 0]);
        let c = bitwise_and(&a, &b);
        assert_eq!(unpack(&c), vec![1, 0, 1, 0, 0, 0, 1, 0]);
        assert_eq!(popcount(&c), 3);
    }

    #[test]
    fn bernoulli_packed_matches_stream_then_pack() {
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;

        let prob = 0.35;
        let length = 200;

        let mut rng1 = ChaCha8Rng::seed_from_u64(999);
        let stream = bernoulli_stream(prob, length, &mut rng1);
        let packed_via_stream = pack(&stream).data;

        let mut rng2 = ChaCha8Rng::seed_from_u64(999);
        let packed_direct = bernoulli_packed(prob, length, &mut rng2);

        assert_eq!(
            packed_via_stream, packed_direct,
            "bernoulli_packed must produce bit-identical output"
        );
    }

    #[test]
    fn bernoulli_packed_fast_statistics() {
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;

        let prob = 0.35;
        let length = 10_000;
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let packed = bernoulli_packed_fast(prob, length, &mut rng);
        let count: u64 = packed.iter().map(|w| w.count_ones() as u64).sum();
        let measured = count as f64 / length as f64;
        assert!(
            (measured - prob).abs() < 0.03,
            "Expected ~{prob}, got {measured}"
        );
    }

    #[test]
    fn bernoulli_packed_fast_deterministic() {
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;

        let mut rng1 = ChaCha8Rng::seed_from_u64(99);
        let a = bernoulli_packed_fast(0.5, 512, &mut rng1);

        let mut rng2 = ChaCha8Rng::seed_from_u64(99);
        let b = bernoulli_packed_fast(0.5, 512, &mut rng2);

        assert_eq!(a, b, "Same seed must produce identical output");
    }

    #[test]
    fn bernoulli_packed_simd_statistics() {
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;

        let prob = 0.35;
        let length = 10_000;
        let mut rng = ChaCha8Rng::seed_from_u64(1337);
        let packed = bernoulli_packed_simd(prob, length, &mut rng);
        let count: u64 = packed.iter().map(|w| w.count_ones() as u64).sum();
        let measured = count as f64 / length as f64;
        assert!(
            (measured - prob).abs() < 0.03,
            "Expected ~{prob}, got {measured}"
        );
    }

    #[test]
    fn bernoulli_packed_simd_deterministic() {
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;

        let mut rng1 = ChaCha8Rng::seed_from_u64(2026);
        let a = bernoulli_packed_simd(0.5, 1024, &mut rng1);

        let mut rng2 = ChaCha8Rng::seed_from_u64(2026);
        let b = bernoulli_packed_simd(0.5, 1024, &mut rng2);

        assert_eq!(a, b, "Same seed must produce identical output");
    }

    #[test]
    fn encode_and_popcount_matches_materialized() {
        use rand::SeedableRng;
        use rand_xoshiro::Xoshiro256PlusPlus;

        let prob = 0.41;
        let lengths = [63_usize, 64, 65, 1003, 1024];
        for length in lengths {
            let words = length.div_ceil(64);
            let weights: Vec<u64> = (0..words)
                .map(|i| (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xA5A5_A5A5_5A5A_5A5A)
                .collect();

            let mut rng1 = Xoshiro256PlusPlus::seed_from_u64(2026);
            let fused = encode_and_popcount(&weights, prob, length, &mut rng1);

            let mut rng2 = Xoshiro256PlusPlus::seed_from_u64(2026);
            let encoded = bernoulli_packed_simd(prob, length, &mut rng2);
            let expected: u64 = encoded
                .iter()
                .zip(weights.iter())
                .map(|(&e, &w)| (e & w).count_ones() as u64)
                .sum();

            assert_eq!(fused, expected, "Mismatch at length={length}");
        }
    }
}
