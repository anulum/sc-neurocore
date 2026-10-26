// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Dense Stochastic Layer

//! # Dense Stochastic Layer
//!
//! Dense layer implemented with Bernoulli bitstream encoding and
//! AND+popcount accumulation.

use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;

use crate::bitstream;

mod allocation;
mod batch;
mod forward;
mod prepacked;

pub use allocation::DenseLayerError;
use allocation::{copy_weights, product, zeroed};

/// Minimum number of inputs before rayon parallelism is used for encoding.
const RAYON_ENCODE_THRESHOLD: usize = 128;
/// Minimum number of neurons before rayon parallelism is used for output compute.
const RAYON_NEURON_THRESHOLD: usize = 8;

/// Vectorized stochastic dense layer.
#[derive(Clone, Debug)]
pub struct DenseLayer {
    /// Number of input features.
    pub n_inputs: usize,
    /// Number of output neurons.
    pub n_neurons: usize,
    /// Bitstream length per encoded scalar.
    pub length: usize,
    /// Scale from bit counts to probability-domain activations.
    pub inv_length: f64,
    /// Words per packed input stream (`ceil(length / 64)`).
    pub words_per_input: usize,
    /// Probability-domain weights in `[0, 1]`.
    pub weights: Vec<Vec<f64>>,
    /// Packed weights in row-major `[neuron][input][word]` order.
    packed_weights_flat: Vec<u64>,
    weight_seed: u64,
}

impl DenseLayer {
    /// Create a layer with random weights sampled from `U(0,1)`.
    ///
    /// # Panics
    ///
    /// Panics for zero dimensions or unavailable storage. Foreign-language
    /// callers use [`Self::try_new`] to receive a typed refusal instead.
    pub fn new(n_inputs: usize, n_neurons: usize, length: usize, seed: u64) -> Self {
        assert!(length > 0, "bitstream length must be > 0");
        assert!(n_inputs > 0, "n_inputs must be > 0");
        assert!(n_neurons > 0, "n_neurons must be > 0");
        Self::try_new(n_inputs, n_neurons, length, seed).expect("dense storage cannot be allocated")
    }

    /// Create independently owned weights with checked, fallible storage.
    ///
    /// Sampling order and per-neuron packed seeds match [`Self::new`].
    ///
    /// # Errors
    ///
    /// Returns `Invalid` for zero dimensions, in length/input/neuron order,
    /// or `Allocation` for overflowing products or unavailable storage.
    pub fn try_new(
        n_inputs: usize,
        n_neurons: usize,
        length: usize,
        seed: u64,
    ) -> Result<Self, DenseLayerError> {
        for (value, message) in [
            (length, "bitstream length must be > 0"),
            (n_inputs, "n_inputs must be > 0"),
            (n_neurons, "n_neurons must be > 0"),
        ] {
            if value == 0 {
                return Err(DenseLayerError::Invalid(message.into()));
            }
        }
        let words_per_input = length.div_ceil(64);
        product(product(n_neurons, n_inputs)?, words_per_input)?;
        let mut weights = Vec::new();
        weights.try_reserve_exact(n_neurons)?;
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        for _ in 0..n_neurons {
            let mut row = zeroed(n_inputs)?;
            for probability in &mut row {
                *probability = rng.random::<f64>();
            }
            weights.push(row);
        }
        let mut layer = Self {
            n_inputs,
            n_neurons,
            length,
            inv_length: 1.0 / length as f64,
            words_per_input,
            weights,
            packed_weights_flat: Vec::new(),
            weight_seed: seed.wrapping_add(1),
        };
        layer.try_refresh_packed_weights()?;
        Ok(layer)
    }

    /// Read the row-major `[neuron][input][word]` packed weight buffer.
    #[inline]
    pub fn packed_weights_flat(&self) -> &[u64] {
        &self.packed_weights_flat
    }

    /// Borrow the words for one neuron/input pair.
    #[inline]
    fn weight_slice(&self, neuron: usize, input: usize) -> &[u64] {
        let start = (neuron * self.n_inputs + input) * self.words_per_input;
        &self.packed_weights_flat[start..start + self.words_per_input]
    }

    /// Return an independently owned copy of the weight matrix.
    ///
    /// # Panics
    ///
    /// Panics when storage is unavailable; [`Self::try_get_weights`] is fallible.
    pub fn get_weights(&self) -> Vec<Vec<f64>> {
        self.try_get_weights()
            .expect("dense storage cannot be allocated")
    }

    /// Copy the weight matrix without infallible vector growth.
    ///
    /// # Errors
    ///
    /// Returns `Allocation` if any matrix row or its outer storage cannot fit.
    pub fn try_get_weights(&self) -> Result<Vec<Vec<f64>>, DenseLayerError> {
        copy_weights(&self.weights)
    }

    /// Set probability weights, retaining both old representations on failure.
    pub fn set_weights(&mut self, weights: Vec<Vec<f64>>) -> Result<(), String> {
        self.try_set_weights(weights)
            .map_err(|error| error.to_string())
    }

    /// Build new packed weights before publishing either representation.
    ///
    /// # Errors
    ///
    /// Returns `Invalid` for an incorrect matrix shape, or `Allocation` for
    /// unavailable packed storage. Every refusal preserves the old layer.
    pub fn try_set_weights(&mut self, weights: Vec<Vec<f64>>) -> Result<(), DenseLayerError> {
        self.validate_weights(&weights)?;
        let packed = self.pack_weights(&weights)?;
        self.weights = weights;
        self.packed_weights_flat = packed;
        Ok(())
    }

    /// Rebuild packed bitstreams using the original weight seeds.
    ///
    /// # Panics
    ///
    /// Panics on invalid shape or unavailable storage. Foreign callers use
    /// [`Self::try_refresh_packed_weights`] for failure-atomic refusal.
    pub fn refresh_packed_weights(&mut self) {
        self.try_refresh_packed_weights()
            .expect("dense packed weights cannot be rebuilt");
    }

    /// Replace packed weights only after the complete rebuild succeeds.
    ///
    /// # Errors
    ///
    /// Returns `Invalid` for an incorrect matrix shape or `Allocation` for
    /// overflowing dimensions or unavailable packed storage.
    pub fn try_refresh_packed_weights(&mut self) -> Result<(), DenseLayerError> {
        self.validate_weights(&self.weights)?;
        let packed = self.pack_weights(&self.weights)?;
        self.packed_weights_flat = packed;
        Ok(())
    }

    /// Preserve row-count and first malformed-row refusal priority.
    fn validate_weights(&self, weights: &[Vec<f64>]) -> Result<(), DenseLayerError> {
        if weights.len() != self.n_neurons {
            return Err(DenseLayerError::Invalid(format!(
                "Expected {} rows, got {}.",
                self.n_neurons,
                weights.len()
            )));
        }
        for (index, row) in weights.iter().enumerate() {
            if row.len() != self.n_inputs {
                return Err(DenseLayerError::Invalid(format!(
                    "Row {} has length {}, expected {}.",
                    index,
                    row.len(),
                    self.n_inputs
                )));
            }
        }
        Ok(())
    }

    /// Encode directly into reserved rows, preserving ChaCha8 draw protocols.
    fn pack_weights(&self, weights: &[Vec<f64>]) -> Result<Vec<u64>, DenseLayerError> {
        let row_words = product(self.n_inputs, self.words_per_input)?;
        let mut packed = zeroed(product(self.n_neurons, row_words)?)?;
        packed
            .par_chunks_mut(row_words)
            .enumerate()
            .try_for_each(|(neuron, row)| {
                let mut rng =
                    ChaCha8Rng::seed_from_u64(self.weight_seed.wrapping_add(neuron as u64));
                for (probability, words) in weights[neuron]
                    .iter()
                    .zip(row.chunks_mut(self.words_per_input))
                {
                    bitstream::bernoulli_packed_simd_into(
                        *probability,
                        self.length,
                        &mut rng,
                        words,
                    )
                    .map_err(|message| DenseLayerError::Invalid(message.into()))?;
                }
                Ok::<(), DenseLayerError>(())
            })?;
        Ok(packed)
    }
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    use super::{DenseLayer, DenseLayerError};
    use crate::bitstream;

    /// Keep caller storage intact on malformed internal output and permit retry.
    #[test]
    fn owned_fast_output_refusal_preserves_surrounding_storage_and_retry() {
        let mut layer = DenseLayer::new(2, 2, 65, 24301);
        let weights = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        layer.set_weights(weights.clone()).unwrap();
        let packed = layer.packed_weights_flat().to_vec();
        for length in [0, 1, 3, 4] {
            let mut storage = [17.0; 6];
            assert_eq!(
                layer
                    .validate_inputs(&[1.0, 0.0])
                    .unwrap()
                    .try_forward_fast_into(42, &mut storage[1..1 + length]),
                Err(DenseLayerError::Invalid(format!(
                    "Expected output length 2, got {length}."
                )))
            );
            assert_eq!(storage, [17.0; 6]);
            layer
                .validate_inputs(&[1.0, 0.0])
                .unwrap()
                .try_forward_fast_into(42, &mut storage[1..3])
                .unwrap();
            assert_eq!(storage, [17.0, 1.0, 0.0, 17.0, 17.0, 17.0]);
            assert_eq!(layer.get_weights(), weights);
            assert_eq!(layer.packed_weights_flat(), packed);
        }
    }

    /// Refuse malformed packed output before writes and retain borrowed packed input.
    #[test]
    fn owned_prepacked_output_refusal_preserves_input_storage_and_retry() {
        let mut layer = DenseLayer::new(2, 2, 65, 24301);
        let weights = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        layer.set_weights(weights.clone()).unwrap();
        let inputs = [u64::MAX, 1, 0, 0];
        let packed = layer.packed_weights_flat().to_vec();
        for length in [0, 1, 3, 4] {
            let mut storage = [23.0; 6];
            assert_eq!(
                layer
                    .validate_prepacked_2d(&inputs, 2, 2)
                    .unwrap()
                    .try_forward_into(&mut storage[1..1 + length]),
                Err(DenseLayerError::Invalid(format!(
                    "Expected output length 2, got {length}."
                )))
            );
            assert_eq!(storage, [23.0; 6]);
            layer
                .validate_prepacked_2d(&inputs, 2, 2)
                .unwrap()
                .try_forward_into(&mut storage[1..3])
                .unwrap();
            assert_eq!(storage, [23.0, 1.0, 0.0, 23.0, 23.0, 23.0]);
            assert_eq!(inputs, [u64::MAX, 1, 0, 0]);
            assert_eq!(layer.get_weights(), weights);
            assert_eq!(layer.packed_weights_flat(), packed);
        }
    }

    #[test]
    fn flat_weight_roundtrip() {
        let layer = DenseLayer::new(3, 2, 130, 42);
        let words = 130_usize.div_ceil(64);
        assert_eq!(layer.words_per_input, words);
        assert_eq!(layer.packed_weights_flat.len(), 3 * 2 * words);

        for neuron in 0..2 {
            let mut rng = ChaCha8Rng::seed_from_u64(43 + neuron as u64);
            for input in 0..3 {
                let expected =
                    bitstream::bernoulli_packed_simd(layer.weights[neuron][input], 130, &mut rng);
                assert_eq!(layer.weight_slice(neuron, input), expected.as_slice());
            }
        }
    }

    #[test]
    fn forward_fused_matches_forward_fast() {
        let layer = DenseLayer::new(16, 8, 1024, 42);
        let inputs: Vec<f64> = (0..16).map(|i| (i as f64) / 16.0).collect();
        let seed = 999_u64;

        let fast = layer
            .forward_fast(&inputs, seed)
            .expect("forward_fast should succeed");
        let fused = layer
            .forward_fused(&inputs, seed)
            .expect("forward_fused should succeed");
        assert_eq!(
            fast, fused,
            "forward_fused must be bit-identical to forward_fast"
        );
    }

    #[test]
    fn forward_batch_matches_sequential_fused() {
        let layer = DenseLayer::new(4, 3, 256, 123);
        let n_samples = 5;
        let inputs_flat: Vec<f64> = (0..(n_samples * 4))
            .map(|i| ((i * 17 + 11) % 100) as f64 / 100.0)
            .collect();
        let seed = 77_u64;

        let batch = layer
            .forward_batch(&inputs_flat, n_samples, seed)
            .expect("forward_batch should succeed");

        for sample_idx in 0..n_samples {
            let row = &inputs_flat[sample_idx * 4..(sample_idx + 1) * 4];
            let sample_seed = seed.wrapping_add((sample_idx as u64).wrapping_mul(1_000_000));
            let expected = layer
                .forward_fused(row, sample_seed)
                .expect("forward_fused should succeed");
            let got = &batch[sample_idx * 3..(sample_idx + 1) * 3];
            assert_eq!(got, expected.as_slice(), "sample_idx={sample_idx}");
        }
    }
}
