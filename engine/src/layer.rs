use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;

use crate::bitstream::pack;
use crate::simd::popcount_dispatch;

#[derive(Clone, Debug)]
pub struct DenseLayer {
    pub n_inputs: usize,
    pub n_neurons: usize,
    pub length: usize,
    pub weights: Vec<Vec<f64>>,
    pub packed_weights: Vec<Vec<Vec<u64>>>,
    weight_seed: u64,
}

impl DenseLayer {
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

    pub fn get_weights(&self) -> Vec<Vec<f64>> {
        self.weights.clone()
    }

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

    pub fn refresh_packed_weights(&mut self) {
        let mut rng = ChaCha8Rng::seed_from_u64(self.weight_seed);
        let mut packed_weights = vec![vec![Vec::<u64>::new(); self.n_inputs]; self.n_neurons];

        for (neuron_idx, neuron_weights) in self.weights.iter().enumerate().take(self.n_neurons) {
            for (input_idx, weight_prob) in neuron_weights.iter().enumerate().take(self.n_inputs) {
                let bits = bernoulli_stream(*weight_prob, self.length, &mut rng);
                packed_weights[neuron_idx][input_idx] = pack(&bits).data;
            }
        }

        self.packed_weights = packed_weights;
    }

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
            let bits = bernoulli_stream(p, self.length, &mut rng);
            packed_inputs[idx] = pack(&bits).data;
        }

        let out = (0..self.n_neurons)
            .into_par_iter()
            .map(|neuron_idx| {
                let mut total = 0_u64;
                let mut and_buf = Vec::<u64>::new();
                for (w, i) in self.packed_weights[neuron_idx]
                    .iter()
                    .zip(packed_inputs.iter())
                {
                    and_buf.clear();
                    and_buf.extend(w.iter().zip(i.iter()).map(|(a, b)| *a & *b));
                    total += popcount_dispatch(&and_buf);
                }
                total as f64 / self.length as f64
            })
            .collect();

        Ok(out)
    }
}

fn bernoulli_stream(prob: f64, length: usize, rng: &mut ChaCha8Rng) -> Vec<u8> {
    let p = prob.clamp(0.0, 1.0);
    let mut out = vec![0_u8; length];
    for bit in &mut out {
        *bit = if rng.gen::<f64>() < p { 1 } else { 0 };
    }
    out
}
