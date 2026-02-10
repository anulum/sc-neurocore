//! # Dense Stochastic Layer
//!
//! Dense layer implemented with Bernoulli bitstream encoding and
//! AND+popcount accumulation.

use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;

use crate::bitstream;

/// Fused bitwise-AND + popcount over two aligned packed word slices.
///
/// Equivalent to `popcount(bitwise_and(a, b))` but avoids
/// materializing the intermediate buffer.
#[inline]
fn fused_and_popcount(a: &[u64], b: &[u64]) -> u64 {
    a.iter()
        .zip(b.iter())
        .map(|(&wa, &wb)| (wa & wb).count_ones() as u64)
        .sum()
}

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
    /// Probability-domain weights in `[0, 1]`.
    pub weights: Vec<Vec<f64>>,
    /// Packed bitstream weights per neuron/input.
    pub packed_weights: Vec<Vec<Vec<u64>>>,
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
            weights,
            packed_weights: vec![],
            weight_seed: seed.wrapping_add(1),
        };
        layer.refresh_packed_weights();
        layer
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
        let mut packed_weights = vec![vec![Vec::<u64>::new(); self.n_inputs]; self.n_neurons];

        for (neuron_idx, neuron_weights) in self.weights.iter().enumerate().take(self.n_neurons) {
            for (input_idx, weight_prob) in neuron_weights.iter().enumerate().take(self.n_inputs) {
                packed_weights[neuron_idx][input_idx] =
                    bitstream::bernoulli_packed(*weight_prob, self.length, &mut rng);
            }
        }

        self.packed_weights = packed_weights;
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

        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut packed_inputs = vec![Vec::<u64>::new(); self.n_inputs];
        for (idx, p) in input_values.iter().copied().enumerate() {
            packed_inputs[idx] = bitstream::bernoulli_packed(p, self.length, &mut rng);
        }

        let out: Vec<f64> = if self.n_neurons >= RAYON_NEURON_THRESHOLD {
            (0..self.n_neurons)
                .into_par_iter()
                .map(|neuron_idx| {
                    let total: u64 = self.packed_weights[neuron_idx]
                        .iter()
                        .zip(packed_inputs.iter())
                        .map(|(w, i)| fused_and_popcount(w, i))
                        .sum();
                    total as f64 / self.length as f64
                })
                .collect()
        } else {
            (0..self.n_neurons)
                .map(|neuron_idx| {
                    let total: u64 = self.packed_weights[neuron_idx]
                        .iter()
                        .zip(packed_inputs.iter())
                        .map(|(w, i)| fused_and_popcount(w, i))
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
                    let mut rng = ChaCha8Rng::seed_from_u64(input_seed);
                    bitstream::bernoulli_packed_fast(p, self.length, &mut rng)
                })
                .collect()
        } else {
            input_values
                .iter()
                .enumerate()
                .map(|(idx, &p)| {
                    let input_seed = seed.wrapping_add(idx as u64);
                    let mut rng = ChaCha8Rng::seed_from_u64(input_seed);
                    bitstream::bernoulli_packed_fast(p, self.length, &mut rng)
                })
                .collect()
        };

        let out: Vec<f64> = if self.n_neurons >= RAYON_NEURON_THRESHOLD {
            (0..self.n_neurons)
                .into_par_iter()
                .map(|neuron_idx| {
                    let total: u64 = self.packed_weights[neuron_idx]
                        .iter()
                        .zip(packed_inputs.iter())
                        .map(|(w, i)| fused_and_popcount(w, i))
                        .sum();
                    total as f64 / self.length as f64
                })
                .collect()
        } else {
            (0..self.n_neurons)
                .map(|neuron_idx| {
                    let total: u64 = self.packed_weights[neuron_idx]
                        .iter()
                        .zip(packed_inputs.iter())
                        .map(|(w, i)| fused_and_popcount(w, i))
                        .sum();
                    total as f64 / self.length as f64
                })
                .collect()
        };

        Ok(out)
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
                let total: u64 = self.packed_weights[neuron_idx]
                    .iter()
                    .zip(packed_inputs.iter())
                    .map(|(w, i)| fused_and_popcount(w, i))
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
                let total: u64 = self.packed_weights[neuron_idx]
                    .iter()
                    .enumerate()
                    .map(|(input_idx, w)| {
                        let row_start = input_idx * words;
                        let input_words = &packed_flat[row_start..row_start + words];
                        fused_and_popcount(w, input_words)
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
        self.forward_fast(input_values, seed)
    }
}
