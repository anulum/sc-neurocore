//! # Dense Stochastic Layer
//!
//! Dense layer implemented with Bernoulli bitstream encoding and
//! AND+popcount accumulation.

use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;

use crate::{bitstream, simd};

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
    /// Words per packed input stream (`ceil(length / 64)`).
    pub words_per_input: usize,
    /// Probability-domain weights in `[0, 1]`.
    pub weights: Vec<Vec<f64>>,
    /// Packed bitstream weights in row-major contiguous layout:
    /// `[neuron][input][word]`.
    packed_weights_flat: Vec<u64>,
    weight_seed: u64,
}

impl DenseLayer {
    /// Create a layer with random weights sampled from `U(0,1)`.
    pub fn new(n_inputs: usize, n_neurons: usize, length: usize, seed: u64) -> Self {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut weights = vec![vec![0.0; n_inputs]; n_neurons];

        for row in &mut weights {
            for p in row {
                *p = rng.gen::<f64>();
            }
        }

        let mut layer = Self {
            n_inputs,
            n_neurons,
            length,
            words_per_input: length.div_ceil(64),
            weights,
            packed_weights_flat: vec![],
            weight_seed: seed.wrapping_add(1),
        };
        layer.refresh_packed_weights();
        layer
    }

    /// Return a single `[word]` slice for one neuron/input pair.
    #[inline]
    fn weight_slice(&self, neuron: usize, input: usize) -> &[u64] {
        let start = (neuron * self.n_inputs + input) * self.words_per_input;
        &self.packed_weights_flat[start..start + self.words_per_input]
    }

    /// Return a copy of weight matrix.
    pub fn get_weights(&self) -> Vec<Vec<f64>> {
        self.weights.clone()
    }

    /// Set probability weights and refresh packed representation.
    pub fn set_weights(&mut self, weights: Vec<Vec<f64>>) -> Result<(), String> {
        if weights.len() != self.n_neurons {
            return Err(format!(
                "Expected {} rows, got {}.",
                self.n_neurons,
                weights.len()
            ));
        }
        for (row_idx, row) in weights.iter().enumerate() {
            if row.len() != self.n_inputs {
                return Err(format!(
                    "Row {} has length {}, expected {}.",
                    row_idx,
                    row.len(),
                    self.n_inputs
                ));
            }
        }
        self.weights = weights;
        self.refresh_packed_weights();
        Ok(())
    }

    /// Rebuild packed weight bitstreams from current weight matrix.
    pub fn refresh_packed_weights(&mut self) {
        let mut rng = ChaCha8Rng::seed_from_u64(self.weight_seed);
        let mut packed_weights_flat =
            vec![0_u64; self.n_neurons * self.n_inputs * self.words_per_input];

        for (neuron_idx, neuron_weights) in self.weights.iter().enumerate().take(self.n_neurons) {
            for (input_idx, weight_prob) in neuron_weights.iter().enumerate().take(self.n_inputs) {
                let packed = bitstream::bernoulli_packed(*weight_prob, self.length, &mut rng);
                let start = (neuron_idx * self.n_inputs + input_idx) * self.words_per_input;
                packed_weights_flat[start..start + self.words_per_input].copy_from_slice(&packed);
            }
        }

        self.packed_weights_flat = packed_weights_flat;
    }

    /// Forward pass using stochastic bitstreams.
    ///
    /// Returns one activation value per neuron.
    pub fn forward(&self, input_values: &[f64], seed: u64) -> Result<Vec<f64>, String> {
        if input_values.len() != self.n_inputs {
            return Err(format!(
                "Expected input of length {}, got {}.",
                self.n_inputs,
                input_values.len()
            ));
        }

        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        let mut packed_inputs = vec![Vec::<u64>::new(); self.n_inputs];
        for (idx, p) in input_values.iter().copied().enumerate() {
            packed_inputs[idx] = bitstream::bernoulli_packed(p, self.length, &mut rng);
        }

        let out: Vec<f64> = if self.n_neurons >= RAYON_NEURON_THRESHOLD {
            (0..self.n_neurons)
                .into_par_iter()
                .map(|neuron_idx| {
                    let total: u64 = packed_inputs
                        .iter()
                        .enumerate()
                        .map(|(input_idx, input_words)| {
                            simd::fused_and_popcount_dispatch(
                                self.weight_slice(neuron_idx, input_idx),
                                input_words,
                            )
                        })
                        .sum();
                    total as f64 / self.length as f64
                })
                .collect()
        } else {
            (0..self.n_neurons)
                .map(|neuron_idx| {
                    let total: u64 = packed_inputs
                        .iter()
                        .enumerate()
                        .map(|(input_idx, input_words)| {
                            simd::fused_and_popcount_dispatch(
                                self.weight_slice(neuron_idx, input_idx),
                                input_words,
                            )
                        })
                        .sum();
                    total as f64 / self.length as f64
                })
                .collect()
        };

        Ok(out)
    }

    /// Forward pass with parallel input encoding.
    ///
    /// Each input is encoded with an independently-seeded RNG:
    /// `seed + input_index` (wrapping).
    pub fn forward_fast(&self, input_values: &[f64], seed: u64) -> Result<Vec<f64>, String> {
        if input_values.len() != self.n_inputs {
            return Err(format!(
                "Expected input of length {}, got {}.",
                self.n_inputs,
                input_values.len()
            ));
        }

        let packed_inputs: Vec<Vec<u64>> = if self.n_inputs >= RAYON_ENCODE_THRESHOLD {
            input_values
                .par_iter()
                .enumerate()
                .map(|(idx, &p)| {
                    let input_seed = seed.wrapping_add(idx as u64);
                    let mut rng = Xoshiro256PlusPlus::seed_from_u64(input_seed);
                    bitstream::bernoulli_packed_simd(p, self.length, &mut rng)
                })
                .collect()
        } else {
            input_values
                .iter()
                .enumerate()
                .map(|(idx, &p)| {
                    let input_seed = seed.wrapping_add(idx as u64);
                    let mut rng = Xoshiro256PlusPlus::seed_from_u64(input_seed);
                    bitstream::bernoulli_packed_simd(p, self.length, &mut rng)
                })
                .collect()
        };

        let out: Vec<f64> = if self.n_neurons >= RAYON_NEURON_THRESHOLD {
            (0..self.n_neurons)
                .into_par_iter()
                .map(|neuron_idx| {
                    let total: u64 = packed_inputs
                        .iter()
                        .enumerate()
                        .map(|(input_idx, input_words)| {
                            simd::fused_and_popcount_dispatch(
                                self.weight_slice(neuron_idx, input_idx),
                                input_words,
                            )
                        })
                        .sum();
                    total as f64 / self.length as f64
                })
                .collect()
        } else {
            (0..self.n_neurons)
                .map(|neuron_idx| {
                    let total: u64 = packed_inputs
                        .iter()
                        .enumerate()
                        .map(|(input_idx, input_words)| {
                            simd::fused_and_popcount_dispatch(
                                self.weight_slice(neuron_idx, input_idx),
                                input_words,
                            )
                        })
                        .sum();
                    total as f64 / self.length as f64
                })
                .collect()
        };

        Ok(out)
    }

    /// Forward pass with fused encode+AND+popcount.
    ///
    /// This path avoids materializing encoded inputs and consumes each encoded
    /// word immediately against its corresponding weight words.
    pub fn forward_fused(&self, input_values: &[f64], seed: u64) -> Result<Vec<f64>, String> {
        if input_values.len() != self.n_inputs {
            return Err(format!(
                "Expected input of length {}, got {}.",
                self.n_inputs,
                input_values.len()
            ));
        }

        let out: Vec<f64> = if self.n_neurons >= RAYON_NEURON_THRESHOLD {
            (0..self.n_neurons)
                .into_par_iter()
                .map(|neuron_idx| {
                    let total: u64 = input_values
                        .iter()
                        .enumerate()
                        .map(|(input_idx, &p)| {
                            let input_seed = seed.wrapping_add(input_idx as u64);
                            let mut rng = Xoshiro256PlusPlus::seed_from_u64(input_seed);
                            bitstream::encode_and_popcount(
                                self.weight_slice(neuron_idx, input_idx),
                                p,
                                self.length,
                                &mut rng,
                            )
                        })
                        .sum();
                    total as f64 / self.length as f64
                })
                .collect()
        } else {
            (0..self.n_neurons)
                .map(|neuron_idx| {
                    let total: u64 = input_values
                        .iter()
                        .enumerate()
                        .map(|(input_idx, &p)| {
                            let input_seed = seed.wrapping_add(input_idx as u64);
                            let mut rng = Xoshiro256PlusPlus::seed_from_u64(input_seed);
                            bitstream::encode_and_popcount(
                                self.weight_slice(neuron_idx, input_idx),
                                p,
                                self.length,
                                &mut rng,
                            )
                        })
                        .sum();
                    total as f64 / self.length as f64
                })
                .collect()
        };

        Ok(out)
    }

    /// Batched forward pass writing into an existing row-major output buffer.
    ///
    /// - `inputs_flat`: shape `[n_samples, n_inputs]`
    /// - `output`: shape `[n_samples, n_neurons]`
    pub fn forward_batch_into(
        &self,
        inputs_flat: &[f64],
        n_samples: usize,
        seed: u64,
        output: &mut [f64],
    ) -> Result<(), String> {
        let expected_inputs = n_samples.checked_mul(self.n_inputs).ok_or_else(|| {
            "Input size overflow when validating n_samples * n_inputs.".to_string()
        })?;
        if inputs_flat.len() != expected_inputs {
            return Err(format!(
                "Expected {} values ({}×{}), got {}.",
                expected_inputs,
                n_samples,
                self.n_inputs,
                inputs_flat.len()
            ));
        }

        let expected_outputs = n_samples.checked_mul(self.n_neurons).ok_or_else(|| {
            "Output size overflow when validating n_samples * n_neurons.".to_string()
        })?;
        if output.len() != expected_outputs {
            return Err(format!(
                "Expected output length {} ({}×{}), got {}.",
                expected_outputs,
                n_samples,
                self.n_neurons,
                output.len()
            ));
        }

        output
            .par_chunks_mut(self.n_neurons)
            .enumerate()
            .for_each(|(sample_idx, out_row)| {
                let start = sample_idx * self.n_inputs;
                let end = start + self.n_inputs;
                let input_row = &inputs_flat[start..end];
                let sample_seed = seed.wrapping_add((sample_idx as u64).wrapping_mul(1_000_000));

                for (neuron_idx, out_val) in out_row.iter_mut().enumerate() {
                    let total: u64 = input_row
                        .iter()
                        .enumerate()
                        .map(|(input_idx, &p)| {
                            let input_seed = sample_seed.wrapping_add(input_idx as u64);
                            let mut rng = Xoshiro256PlusPlus::seed_from_u64(input_seed);
                            bitstream::encode_and_popcount(
                                self.weight_slice(neuron_idx, input_idx),
                                p,
                                self.length,
                                &mut rng,
                            )
                        })
                        .sum();
                    *out_val = total as f64 / self.length as f64;
                }
            });

        Ok(())
    }

    /// Batched forward pass: process N input vectors in one call.
    ///
    /// `inputs_flat` is row-major: `[n_samples, n_inputs]`.
    /// Returns flat output: `[n_samples, n_neurons]`.
    pub fn forward_batch(
        &self,
        inputs_flat: &[f64],
        n_samples: usize,
        seed: u64,
    ) -> Result<Vec<f64>, String> {
        let output_len = n_samples.checked_mul(self.n_neurons).ok_or_else(|| {
            "Output size overflow when allocating n_samples * n_neurons.".to_string()
        })?;
        let mut output = vec![0.0_f64; output_len];
        self.forward_batch_into(inputs_flat, n_samples, seed, &mut output)?;
        Ok(output)
    }

    /// Forward pass with pre-packed input bitstreams.
    ///
    /// `packed_inputs` must have shape:
    /// - outer length = `n_inputs`
    /// - inner length = `ceil(length / 64)`
    pub fn forward_prepacked(&self, packed_inputs: &[Vec<u64>]) -> Result<Vec<f64>, String> {
        if packed_inputs.len() != self.n_inputs {
            return Err(format!(
                "Expected {} packed inputs, got {}.",
                self.n_inputs,
                packed_inputs.len()
            ));
        }
        let expected_words = self.length.div_ceil(64);
        for (idx, pi) in packed_inputs.iter().enumerate() {
            if pi.len() != expected_words {
                return Err(format!(
                    "Packed input {} has {} words, expected {}.",
                    idx,
                    pi.len(),
                    expected_words
                ));
            }
        }

        let out = (0..self.n_neurons)
            .into_par_iter()
            .map(|neuron_idx| {
                let total: u64 = packed_inputs
                    .iter()
                    .enumerate()
                    .map(|(input_idx, input_words)| {
                        simd::fused_and_popcount_dispatch(
                            self.weight_slice(neuron_idx, input_idx),
                            input_words,
                        )
                    })
                    .sum();
                total as f64 / self.length as f64
            })
            .collect();

        Ok(out)
    }

    /// Forward pass with pre-packed inputs from a 2-D contiguous array.
    ///
    /// `packed_flat` is a flat row-major buffer of shape `[n_inputs, words]`.
    /// Each row is one input's packed bitstream words.
    pub fn forward_prepacked_2d(
        &self,
        packed_flat: &[u64],
        n_inputs: usize,
        words: usize,
    ) -> Result<Vec<f64>, String> {
        if n_inputs != self.n_inputs {
            return Err(format!(
                "Expected {} packed inputs, got {}.",
                self.n_inputs, n_inputs
            ));
        }
        let expected_words = self.length.div_ceil(64);
        if words != expected_words {
            return Err(format!(
                "Expected {} words per input, got {}.",
                expected_words, words
            ));
        }
        if packed_flat.len() != n_inputs * words {
            return Err(format!(
                "Flat buffer length {} != n_inputs({}) * words({}).",
                packed_flat.len(),
                n_inputs,
                words
            ));
        }

        let out = (0..self.n_neurons)
            .into_par_iter()
            .map(|neuron_idx| {
                let total: u64 = (0..self.n_inputs)
                    .map(|input_idx| {
                        let row_start = input_idx * words;
                        let input_words = &packed_flat[row_start..row_start + words];
                        simd::fused_and_popcount_dispatch(
                            self.weight_slice(neuron_idx, input_idx),
                            input_words,
                        )
                    })
                    .sum();
                total as f64 / self.length as f64
            })
            .collect();

        Ok(out)
    }

    /// Single-call dense forward with parallel Bernoulli encoding.
    ///
    /// This mirrors `forward_fast` and exists for numpy-native Python bindings.
    pub fn forward_numpy_inner(&self, input_values: &[f64], seed: u64) -> Result<Vec<f64>, String> {
        self.forward_fused(input_values, seed)
    }
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    use super::DenseLayer;
    use crate::bitstream;

    #[test]
    fn flat_weight_roundtrip() {
        let layer = DenseLayer::new(3, 2, 130, 42);
        let words = 130_usize.div_ceil(64);
        assert_eq!(layer.words_per_input, words);
        assert_eq!(layer.packed_weights_flat.len(), 3 * 2 * words);

        let mut rng = ChaCha8Rng::seed_from_u64(43);
        for neuron in 0..2 {
            for input in 0..3 {
                let expected =
                    bitstream::bernoulli_packed(layer.weights[neuron][input], 130, &mut rng);
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
