//! # Bitstream Operations
//!
//! Core bitstream packing and logic primitives for stochastic computing.
//! Probabilities are represented as packed Bernoulli bitstreams stored in `u64` words.

use rand::Rng;

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
    use super::{bernoulli_packed, bernoulli_stream, bitwise_and, pack, popcount, unpack};

    #[test]
    fn pack_unpack_roundtrip() {
        let bits = vec![1, 0, 1, 1, 0, 1, 0, 0, 1];
        let packed = pack(&bits);
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
}
