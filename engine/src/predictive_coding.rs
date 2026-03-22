// SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Zero-multiplication predictive coding (Conjecture C9)

//! Predictive coding via XOR + popcount in packed bitstream domain.
//!
//! Error = XOR(predicted, actual), magnitude = popcount(error) / L.
//! No multiplications needed — maps to XOR gates + popcount tree on FPGA.

use crate::bitstream;

/// Compute prediction error between two packed bitstreams via XOR + popcount.
/// Returns error magnitude in [0, 1].
pub fn prediction_error_packed(predicted: &[u64], actual: &[u64], length: usize) -> f64 {
    if length == 0 {
        return 0.0;
    }
    let n = predicted.len().min(actual.len());
    let mut xor_result = vec![0u64; n];
    for i in 0..n {
        xor_result[i] = predicted[i] ^ actual[i];
    }
    let hamming = bitstream::popcount_words_portable(&xor_result);
    hamming as f64 / length as f64
}

/// Batch prediction error: n_neurons × n_inputs packed streams.
/// Returns per-neuron surprise values.
pub fn batch_prediction_error(
    predicted: &[Vec<u64>], // [n_neurons * n_inputs] flattened
    actual: &[Vec<u64>],    // [n_inputs]
    n_neurons: usize,
    n_inputs: usize,
    length: usize,
) -> Vec<f64> {
    let mut surprises = vec![0.0f64; n_neurons];
    for j in 0..n_neurons {
        let mut total_error = 0.0;
        for i in 0..n_inputs {
            let pred_idx = j * n_inputs + i;
            if pred_idx < predicted.len() && i < actual.len() {
                total_error += prediction_error_packed(&predicted[pred_idx], &actual[i], length);
            }
        }
        surprises[j] = total_error / n_inputs.max(1) as f64;
    }
    surprises
}

/// STDP-like weight update: push prediction weight toward actual probability.
pub fn update_prediction_weights(
    weights: &mut [f64],  // [n_neurons * n_inputs] flattened
    actual_probs: &[f64], // [n_inputs]
    n_neurons: usize,
    n_inputs: usize,
    lr: f64,
) {
    for j in 0..n_neurons {
        for i in 0..n_inputs {
            let idx = j * n_inputs + i;
            if idx < weights.len() && i < actual_probs.len() {
                weights[idx] += lr * (actual_probs[i] - weights[idx]);
                weights[idx] = weights[idx].clamp(0.0, 1.0);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identical_streams_zero_error() {
        let a = vec![0xFF_FF_FF_FF_FF_FF_FF_FFu64; 16]; // all 1s
        let error = prediction_error_packed(&a, &a, 1024);
        assert!((error - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_opposite_streams_max_error() {
        let a = vec![0xFF_FF_FF_FF_FF_FF_FF_FFu64; 16];
        let b = vec![0u64; 16];
        let error = prediction_error_packed(&a, &b, 1024);
        assert!((error - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_batch_error_shape() {
        let pred = vec![vec![0u64; 4]; 6]; // 2 neurons × 3 inputs
        let actual = vec![vec![0xFF_FF_FF_FF_FF_FF_FF_FFu64; 4]; 3];
        let surprises = batch_prediction_error(&pred, &actual, 2, 3, 256);
        assert_eq!(surprises.len(), 2);
        assert!(surprises[0] > 0.0);
    }

    #[test]
    fn test_weight_update() {
        let mut weights = vec![0.5, 0.5, 0.5, 0.5]; // 2×2
        let actual = vec![0.8, 0.2];
        update_prediction_weights(&mut weights, &actual, 2, 2, 0.5);
        assert!(weights[0] > 0.5); // moved toward 0.8
        assert!(weights[1] < 0.5); // moved toward 0.2
    }
}
