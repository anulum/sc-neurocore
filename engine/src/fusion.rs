// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Multi-modal fusion layer using stochastic multiplexing

//! Multi-modal fusion layer using stochastic multiplexing.

/// Weighted stochastic fusion across modalities.
///
/// Computes P(out) = Σ_i w_i · P(in_i) for normalized weights.
#[derive(Clone, Debug)]
pub struct FusionLayer {
    /// Normalized fusion weights (sum to 1.0).
    pub weights: Vec<f64>,
    pub n_modalities: usize,
    pub n_features: usize,
}

impl FusionLayer {
    /// Create with raw weights (will be normalized internally).
    pub fn new(raw_weights: &[f64], n_features: usize) -> Self {
        let total: f64 = raw_weights.iter().sum();
        let weights: Vec<f64> = if total > 0.0 {
            raw_weights.iter().map(|w| w / total).collect()
        } else {
            vec![1.0 / raw_weights.len() as f64; raw_weights.len()]
        };
        Self {
            n_modalities: weights.len(),
            weights,
            n_features,
        }
    }

    /// Forward pass: inputs is flat [n_modalities * n_features].
    /// Returns [n_features].
    pub fn forward(&self, inputs: &[f64]) -> Vec<f64> {
        assert_eq!(inputs.len(), self.n_modalities * self.n_features);
        let mut out = vec![0.0; self.n_features];
        for (m, &w) in self.weights.iter().enumerate() {
            let offset = m * self.n_features;
            for f in 0..self.n_features {
                out[f] += inputs[offset + f] * w;
            }
        }
        out
    }
}

/// Dense layer with memristive hardware non-idealities.
///
/// Prezioso et al., Nature 521:61-64, 2015.
#[derive(Clone, Debug)]
pub struct MemristiveLayer {
    pub inner: crate::layer::DenseLayer,
    pub stuck_mask: Vec<bool>,
    pub stuck_values: Vec<f64>,
}

impl MemristiveLayer {
    pub fn new(
        n_inputs: usize,
        n_neurons: usize,
        length: usize,
        seed: u64,
        stuck_rate: f64,
        variability: f64,
    ) -> Self {
        use rand::{RngExt, SeedableRng};
        use rand_xoshiro::Xoshiro256PlusPlus;

        let mut layer = crate::layer::DenseLayer::new(n_inputs, n_neurons, length, seed);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed.wrapping_add(0xDEFEC7));

        let total = n_neurons * n_inputs;
        let mut stuck_mask = vec![false; total];
        let mut stuck_values = vec![0.0; total];

        for i in 0..n_neurons {
            for j in 0..n_inputs {
                let idx = i * n_inputs + j;
                // Write noise (variability)
                let noise: f64 = rng.random::<f64>() * 2.0 * variability - variability;
                layer.weights[i][j] = (layer.weights[i][j] + noise).clamp(0.0, 1.0);

                // Stuck-at faults
                if rng.random::<f64>() < stuck_rate {
                    stuck_mask[idx] = true;
                    stuck_values[idx] = if rng.random::<bool>() { 1.0 } else { 0.0 };
                    layer.weights[i][j] = stuck_values[idx];
                }
            }
        }
        layer.refresh_packed_weights();

        Self {
            inner: layer,
            stuck_mask,
            stuck_values,
        }
    }

    pub fn forward(&self, input_values: &[f64], seed: u64) -> Result<Vec<f64>, String> {
        self.inner.forward_fused(input_values, seed)
    }
}

/// Online STDP-integrated learning layer.
#[derive(Clone, Debug)]
pub struct LearningLayer {
    pub n_inputs: usize,
    pub n_neurons: usize,
    pub weights: Vec<Vec<f64>>,
    pub learning_rate: f64,
}

impl LearningLayer {
    pub fn new(n_inputs: usize, n_neurons: usize, learning_rate: f64, seed: u64) -> Self {
        use rand::{RngExt, SeedableRng};
        use rand_chacha::ChaCha8Rng;

        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let weights: Vec<Vec<f64>> = (0..n_neurons)
            .map(|_| (0..n_inputs).map(|_| rng.random::<f64>()).collect())
            .collect();
        Self {
            n_inputs,
            n_neurons,
            weights,
            learning_rate,
        }
    }

    /// Forward pass with STDP weight update.
    /// Returns which neurons spiked.
    #[allow(clippy::needless_range_loop)]
    pub fn step(&mut self, input_spikes: &[bool], threshold: f64) -> Vec<bool> {
        assert_eq!(input_spikes.len(), self.n_inputs);
        let mut output = vec![false; self.n_neurons];

        for i in 0..self.n_neurons {
            let mut current = 0.0;
            for j in 0..self.n_inputs {
                if input_spikes[j] {
                    current += self.weights[i][j];
                }
            }
            output[i] = current > threshold;

            // STDP update
            for j in 0..self.n_inputs {
                if input_spikes[j] && output[i] {
                    // LTP
                    self.weights[i][j] = (self.weights[i][j] + self.learning_rate).min(1.0);
                } else if input_spikes[j] && !output[i] {
                    // LTD
                    self.weights[i][j] = (self.weights[i][j] - self.learning_rate * 0.5).max(0.0);
                }
            }
        }
        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fusion_weighted_sum() {
        let layer = FusionLayer::new(&[0.7, 0.3], 4);
        let inputs = vec![
            1.0, 1.0, 1.0, 1.0, // modality 0
            0.0, 0.0, 0.0, 0.0, // modality 1
        ];
        let out = layer.forward(&inputs);
        assert_eq!(out.len(), 4);
        assert!((out[0] - 0.7).abs() < 1e-10);
    }

    #[test]
    fn fusion_equal_weights() {
        let layer = FusionLayer::new(&[1.0, 1.0], 2);
        let inputs = vec![0.6, 0.4, 0.2, 0.8];
        let out = layer.forward(&inputs);
        assert!((out[0] - 0.4).abs() < 1e-10);
        assert!((out[1] - 0.6).abs() < 1e-10);
    }

    #[test]
    fn memristive_forward() {
        let layer = MemristiveLayer::new(4, 2, 256, 42, 0.05, 0.01);
        let out = layer.forward(&[0.5, 0.5, 0.5, 0.5], 99).unwrap();
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn learning_layer_fires() {
        let mut layer = LearningLayer::new(4, 2, 0.01, 42);
        let spikes = vec![true, true, true, true];
        let out = layer.step(&spikes, 0.5);
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn learning_layer_weights_change() {
        let mut layer = LearningLayer::new(4, 2, 0.1, 42);
        let initial = layer.weights.clone();
        for _ in 0..50 {
            layer.step(&[true, true, false, false], 0.3);
        }
        assert_ne!(layer.weights, initial);
    }
}
