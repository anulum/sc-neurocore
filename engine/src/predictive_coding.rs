// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
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

// --- Spike codec predictor loops (EMA + LFSR) for PredictiveSpikeCodec ---

use crate::encoder::Lfsr16;

/// EMA predict-and-XOR loop for spike codec compression.
/// Returns (error_matrix flattened row-major, correct_prediction_count).
pub fn predict_and_xor_ema(
    spikes: &[i8], // (T * N) row-major
    n_channels: usize,
    alpha: f64,
    threshold: f64,
) -> (Vec<i8>, usize) {
    let t_steps = spikes.len() / n_channels;
    let mut rates = vec![0.0f64; n_channels];
    let mut errors = vec![0i8; spikes.len()];
    let mut correct: usize = 0;
    let one_minus_alpha = 1.0 - alpha;

    for t in 0..t_steps {
        let row_start = t * n_channels;
        for ch in 0..n_channels {
            let actual = spikes[row_start + ch];
            let predicted = if rates[ch] > threshold { 1i8 } else { 0i8 };
            let err = actual ^ predicted;
            errors[row_start + ch] = err;
            if err == 0 {
                correct += 1;
            }
            rates[ch] = one_minus_alpha * rates[ch] + alpha * (actual as f64);
        }
    }
    (errors, correct)
}

/// EMA XOR-and-recover loop for spike codec decompression.
pub fn xor_and_recover_ema(
    errors: &[i8],
    n_channels: usize,
    alpha: f64,
    threshold: f64,
) -> Vec<i8> {
    let t_steps = errors.len() / n_channels;
    let mut rates = vec![0.0f64; n_channels];
    let mut spikes = vec![0i8; errors.len()];
    let one_minus_alpha = 1.0 - alpha;

    for t in 0..t_steps {
        let row_start = t * n_channels;
        for ch in 0..n_channels {
            let predicted = if rates[ch] > threshold { 1i8 } else { 0i8 };
            let actual = errors[row_start + ch] ^ predicted;
            spikes[row_start + ch] = actual;
            rates[ch] = one_minus_alpha * rates[ch] + alpha * (actual as f64);
        }
    }
    spikes
}

/// LFSR predict-and-XOR loop: bit-true with sc_bitstream_encoder.v.
/// Returns (error_matrix flattened row-major, correct_prediction_count).
pub fn predict_and_xor_lfsr(
    spikes: &[i8],
    n_channels: usize,
    alpha_q8: i32,
    seed: u16,
) -> (Vec<i8>, usize) {
    let t_steps = spikes.len() / n_channels;
    let mut rates_q8 = vec![0i32; n_channels];
    let mut errors = vec![0i8; spikes.len()];
    let mut correct: usize = 0;

    // Per-channel LFSR (decorrelated seeds)
    let mut lfsrs: Vec<Lfsr16> = (0..n_channels)
        .map(|ch| {
            let s = ((seed as u32).wrapping_add((ch as u32).wrapping_mul(7919))) & 0xFFFF;
            Lfsr16::new(if s == 0 { 1 } else { s as u16 })
        })
        .collect();

    for t in 0..t_steps {
        let row_start = t * n_channels;
        for ch in 0..n_channels {
            let actual = spikes[row_start + ch];
            let predicted = if (lfsrs[ch].reg as i32) < rates_q8[ch] {
                1i8
            } else {
                0i8
            };
            lfsrs[ch].step();

            let err = actual ^ predicted;
            errors[row_start + ch] = err;
            if err == 0 {
                correct += 1;
            }

            let target: i32 = if actual != 0 { 255 } else { 0 };
            rates_q8[ch] += (alpha_q8 * (target - rates_q8[ch])) >> 8;
            rates_q8[ch] = rates_q8[ch].clamp(0, 255);
        }
    }
    (errors, correct)
}

/// LFSR XOR-and-recover loop for decompression.
pub fn xor_and_recover_lfsr(errors: &[i8], n_channels: usize, alpha_q8: i32, seed: u16) -> Vec<i8> {
    let t_steps = errors.len() / n_channels;
    let mut rates_q8 = vec![0i32; n_channels];
    let mut spikes = vec![0i8; errors.len()];

    let mut lfsrs: Vec<Lfsr16> = (0..n_channels)
        .map(|ch| {
            let s = ((seed as u32).wrapping_add((ch as u32).wrapping_mul(7919))) & 0xFFFF;
            Lfsr16::new(if s == 0 { 1 } else { s as u16 })
        })
        .collect();

    for t in 0..t_steps {
        let row_start = t * n_channels;
        for ch in 0..n_channels {
            let predicted = if (lfsrs[ch].reg as i32) < rates_q8[ch] {
                1i8
            } else {
                0i8
            };
            lfsrs[ch].step();

            let actual = errors[row_start + ch] ^ predicted;
            spikes[row_start + ch] = actual;

            let target: i32 = if actual != 0 { 255 } else { 0 };
            rates_q8[ch] += (alpha_q8 * (target - rates_q8[ch])) >> 8;
            rates_q8[ch] = rates_q8[ch].clamp(0, 255);
        }
    }
    spikes
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

    #[test]
    fn test_ema_roundtrip() {
        // All zeros: predict 0, XOR 0 = 0 error
        let spikes = vec![0i8; 100]; // 10 timesteps x 10 channels
        let (errors, correct) = predict_and_xor_ema(&spikes, 10, 0.005, 0.5);
        assert_eq!(errors.len(), 100);
        assert_eq!(correct, 100); // all correct (predict 0, actual 0)
        let recovered = xor_and_recover_ema(&errors, 10, 0.005, 0.5);
        assert_eq!(recovered, spikes);
    }

    #[test]
    fn test_ema_roundtrip_with_spikes() {
        let mut spikes = vec![0i8; 200]; // 20 x 10
        spikes[5] = 1; // spike at t=0, ch=5
        spikes[15] = 1; // spike at t=1, ch=5
        let (errors, _) = predict_and_xor_ema(&spikes, 10, 0.01, 0.5);
        let recovered = xor_and_recover_ema(&errors, 10, 0.01, 0.5);
        assert_eq!(recovered, spikes);
    }

    #[test]
    fn test_lfsr_roundtrip() {
        let spikes = vec![0i8; 100];
        let (errors, correct) = predict_and_xor_lfsr(&spikes, 10, 1, 0xACE1);
        assert_eq!(correct, 100);
        let recovered = xor_and_recover_lfsr(&errors, 10, 1, 0xACE1);
        assert_eq!(recovered, spikes);
    }

    #[test]
    fn test_lfsr_roundtrip_with_spikes() {
        let mut spikes = vec![0i8; 200];
        spikes[5] = 1;
        spikes[15] = 1;
        spikes[100] = 1;
        let (errors, _) = predict_and_xor_lfsr(&spikes, 10, 2, 0x1234);
        let recovered = xor_and_recover_lfsr(&errors, 10, 2, 0x1234);
        assert_eq!(recovered, spikes);
    }

    #[test]
    fn test_lfsr_deterministic() {
        let spikes = vec![0i8; 50]; // 5 x 10
        let (e1, _) = predict_and_xor_lfsr(&spikes, 10, 1, 0xBEEF);
        let (e2, _) = predict_and_xor_lfsr(&spikes, 10, 1, 0xBEEF);
        assert_eq!(e1, e2);
    }
}
