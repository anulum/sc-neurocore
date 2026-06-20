// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// Copyright (C) 2020-2026 Miroslav Sotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore - Public SC inference over pre-packed weight bitstreams

//! Stochastic forward pass over caller-owned packed weight bitstreams.
//!
//! The input encoder is the 16-bit LFSR comparator (`encoder::Lfsr16` semantics:
//! taps 16, 14, 13, 11; `bit = reg < x_value` then advance), so the result is
//! bit-identical to the NumPy fallback in
//! `src/sc_neurocore/accel/sc_inference.py` for a fixed seed.

use std::error::Error;
use std::fmt;

/// SC inference contract errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ScForwardError {
    InvalidLength(usize),
    WeightLengthMismatch { expected: usize, actual: usize },
    InputLengthMismatch { expected: usize, actual: usize },
    ProbabilityOutOfRange,
}

impl fmt::Display for ScForwardError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidLength(length) => write!(f, "length must be positive, got {length}"),
            Self::WeightLengthMismatch { expected, actual } => write!(
                f,
                "weights_packed length must be n_out*n_in*n_words ({expected}), got {actual}"
            ),
            Self::InputLengthMismatch { expected, actual } => write!(
                f,
                "input_probs length must be n_in ({expected}), got {actual}"
            ),
            Self::ProbabilityOutOfRange => write!(f, "input_probs must lie in [0, 1]"),
        }
    }
}

impl Error for ScForwardError {}

/// Per-input non-zero 16-bit LFSR seed derived from the base seed.
fn input_seed(base_seed: u64, input_idx: usize) -> u16 {
    let masked = (base_seed.wrapping_add(input_idx as u64) & 0xFFFF) as u16;
    if masked == 0 {
        1
    } else {
        masked
    }
}

/// Encode one probability into a `length`-bit LFSR comparator stream, packed
/// LSB-first into the `out` words (matching `bitstream::pack`).
fn lfsr_encode_packed(p: f64, length: usize, seed: u16, out: &mut [u64]) {
    // `round_ties_even` matches NumPy `np.rint`; the ceiling allows p == 1.0.
    let x_value = (p * 65536.0).round_ties_even().clamp(0.0, 65536.0) as u32;
    let mut reg: u16 = seed;
    for tap in 0..length {
        if u32::from(reg) < x_value {
            out[tap / 64] |= 1_u64 << (tap % 64);
        }
        let feedback = ((reg >> 15) ^ (reg >> 13) ^ (reg >> 12) ^ (reg >> 10)) & 1;
        reg = (reg << 1) | feedback;
    }
}

/// Stochastic forward pass over pre-packed unipolar weight bitstreams.
///
/// `weights_packed` is row-major `n_out * n_in * n_words` with
/// `n_words = ceil(length / 64)`. Returns `n_out` estimates of
/// `weights @ input_probs`, the AND-then-popcount count divided by `length`.
pub fn sc_forward_packed(
    weights_packed: &[u64],
    n_out: usize,
    n_in: usize,
    n_words: usize,
    input_probs: &[f64],
    length: usize,
    seed: u64,
) -> Result<Vec<f64>, ScForwardError> {
    if length == 0 || n_words != length.div_ceil(64) {
        return Err(ScForwardError::InvalidLength(length));
    }
    let expected_weights = n_out * n_in * n_words;
    if weights_packed.len() != expected_weights {
        return Err(ScForwardError::WeightLengthMismatch {
            expected: expected_weights,
            actual: weights_packed.len(),
        });
    }
    if input_probs.len() != n_in {
        return Err(ScForwardError::InputLengthMismatch {
            expected: n_in,
            actual: input_probs.len(),
        });
    }
    if input_probs.iter().any(|&p| !(0.0..=1.0).contains(&p)) {
        return Err(ScForwardError::ProbabilityOutOfRange);
    }

    let mut input_words = vec![0_u64; n_in * n_words];
    for input_idx in 0..n_in {
        let seed_i = input_seed(seed, input_idx);
        lfsr_encode_packed(
            input_probs[input_idx],
            length,
            seed_i,
            &mut input_words[input_idx * n_words..(input_idx + 1) * n_words],
        );
    }

    let length_f = length as f64;
    let mut outputs = vec![0.0_f64; n_out];
    for (output_idx, output) in outputs.iter_mut().enumerate() {
        let mut accumulator: u64 = 0;
        for input_idx in 0..n_in {
            let weight_base = (output_idx * n_in + input_idx) * n_words;
            let input_base = input_idx * n_words;
            for word in 0..n_words {
                accumulator += u64::from(
                    (weights_packed[weight_base + word] & input_words[input_base + word])
                        .count_ones(),
                );
            }
        }
        *output = accumulator as f64 / length_f;
    }
    Ok(outputs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_probability_emits_no_ones() {
        let mut words = vec![0_u64; 2];
        lfsr_encode_packed(0.0, 128, 0xACE1, &mut words);
        assert_eq!(words[0].count_ones() + words[1].count_ones(), 0);
    }

    #[test]
    fn full_probability_emits_all_ones() {
        let mut words = vec![0_u64; 1];
        lfsr_encode_packed(1.0, 64, 0xACE1, &mut words);
        assert_eq!(words[0].count_ones(), 64);
    }

    #[test]
    fn all_ones_weights_recover_input_proportion() {
        // Weight bitstream of all ones -> AND-popcount/length recovers the input
        // proportion to within LFSR discretisation.
        let length = 4096;
        let n_words = length / 64;
        let weights = vec![u64::MAX; n_words];
        let result = sc_forward_packed(&weights, 1, 1, n_words, &[0.25], length, 0xACE1).unwrap();
        assert!((result[0] - 0.25).abs() < 0.02);
    }

    #[test]
    fn rejects_bad_shapes() {
        assert_eq!(
            sc_forward_packed(&[0], 1, 1, 1, &[0.0], 0, 1).unwrap_err(),
            ScForwardError::InvalidLength(0)
        );
        assert!(matches!(
            sc_forward_packed(&[0, 0], 1, 1, 1, &[0.0], 64, 1).unwrap_err(),
            ScForwardError::WeightLengthMismatch { .. }
        ));
        assert!(matches!(
            sc_forward_packed(&[0], 1, 1, 1, &[0.0, 0.0], 64, 1).unwrap_err(),
            ScForwardError::InputLengthMismatch { .. }
        ));
        assert_eq!(
            sc_forward_packed(&[0], 1, 1, 1, &[1.5], 64, 1).unwrap_err(),
            ScForwardError::ProbabilityOutOfRange
        );
    }
}
