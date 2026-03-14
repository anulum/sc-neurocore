// SPDX-License-Identifier: AGPL-3.0-or-later
//! SC recurrent/reservoir layer (echo state network).
//!
//! Jaeger, GMD Report 148, 2001.

use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;

/// Recurrent reservoir layer with state feedback.
#[derive(Clone, Debug)]
pub struct RecurrentLayer {
    pub n_inputs: usize,
    pub n_neurons: usize,
    /// Input weights, flat [n_neurons, n_inputs].
    pub w_in: Vec<f64>,
    /// Recurrent weights, flat [n_neurons, n_neurons].
    pub w_rec: Vec<f64>,
    /// Current state [n_neurons].
    pub state: Vec<f64>,
}

impl RecurrentLayer {
    pub fn new(n_inputs: usize, n_neurons: usize, seed: u64) -> Self {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let w_in: Vec<f64> = (0..n_neurons * n_inputs)
            .map(|_| rng.random::<f64>() * 0.5)
            .collect();
        let w_rec: Vec<f64> = (0..n_neurons * n_neurons)
            .map(|_| rng.random::<f64>() * 0.2)
            .collect();
        Self {
            n_inputs,
            n_neurons,
            w_in,
            w_rec,
            state: vec![0.0; n_neurons],
        }
    }

    /// Process one time step. Returns reference to updated state.
    pub fn step(&mut self, input: &[f64]) -> &[f64] {
        assert_eq!(input.len(), self.n_inputs);
        let mut new_state = vec![0.0; self.n_neurons];
        for i in 0..self.n_neurons {
            let mut val = 0.0;
            for j in 0..self.n_inputs {
                val += self.w_in[i * self.n_inputs + j] * input[j];
            }
            for j in 0..self.n_neurons {
                val += self.w_rec[i * self.n_neurons + j] * self.state[j];
            }
            new_state[i] = val.clamp(0.0, 1.0);
        }
        self.state = new_state;
        &self.state
    }

    pub fn reset(&mut self) {
        self.state.fill(0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn state_changes_after_step() {
        let mut layer = RecurrentLayer::new(3, 5, 42);
        let input = vec![0.5, 0.3, 0.8];
        let state = layer.step(&input).to_vec();
        assert!(state.iter().any(|&s| s > 0.0));
    }

    #[test]
    fn reset_clears_state() {
        let mut layer = RecurrentLayer::new(3, 5, 42);
        layer.step(&[0.5, 0.3, 0.8]);
        layer.reset();
        assert!(layer.state.iter().all(|&s| s == 0.0));
    }

    #[test]
    fn state_bounded() {
        let mut layer = RecurrentLayer::new(2, 4, 99);
        for _ in 0..100 {
            layer.step(&[1.0, 1.0]);
        }
        assert!(layer.state.iter().all(|&s| (0.0..=1.0).contains(&s)));
    }

    #[test]
    fn output_shape() {
        let mut layer = RecurrentLayer::new(4, 8, 0);
        let out = layer.step(&[0.1, 0.2, 0.3, 0.4]);
        assert_eq!(out.len(), 8);
    }
}
