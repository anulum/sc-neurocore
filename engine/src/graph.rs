//! # Stochastic Graph Layer
//!
//! Graph message-passing layer with both rate-mode and SC-mode forward paths.

use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;

pub struct StochasticGraphLayer {
    /// Number of nodes.
    pub n_nodes: usize,
    /// Features per node.
    pub n_features: usize,
    /// Adjacency matrix in row-major layout.
    pub adj: Vec<f64>,
    /// Learnable weight matrix in row-major layout.
    pub weights: Vec<f64>,
    /// Precomputed row sums of adjacency.
    pub degrees: Vec<f64>,
}

impl StochasticGraphLayer {
    /// Construct a graph layer with random weights.
    pub fn new(adj_flat: Vec<f64>, n_nodes: usize, n_features: usize, seed: u64) -> Self {
        assert_eq!(
            adj_flat.len(),
            n_nodes * n_nodes,
            "adj_flat must have length n_nodes * n_nodes",
        );

        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut weights = vec![0.0_f64; n_features * n_features];
        for w in &mut weights {
            *w = rng.gen::<f64>();
        }

        let mut degrees = vec![0.0_f64; n_nodes];
        for i in 0..n_nodes {
            let mut sum = 0.0_f64;
            for j in 0..n_nodes {
                sum += adj_flat[i * n_nodes + j];
            }
            degrees[i] = sum;
        }

        Self {
            n_nodes,
            n_features,
            adj: adj_flat,
            weights,
            degrees,
        }
    }

    /// Rate-mode graph forward pass.
    pub fn forward(&self, node_features: &[f64]) -> Result<Vec<f64>, String> {
        if node_features.len() != self.n_nodes * self.n_features {
            return Err(format!(
                "node_features length mismatch: got {}, expected {}.",
                node_features.len(),
                self.n_nodes * self.n_features
            ));
        }

        let agg_rows: Vec<Vec<f64>> = (0..self.n_nodes)
            .into_par_iter()
            .map(|i| {
                let mut row = vec![0.0_f64; self.n_features];
                for f in 0..self.n_features {
                    let mut acc = 0.0_f64;
                    for j in 0..self.n_nodes {
                        acc +=
                            self.adj[i * self.n_nodes + j] * node_features[j * self.n_features + f];
                    }
                    row[f] = acc;
                }

                if self.degrees[i] != 0.0 {
                    for x in &mut row {
                        *x /= self.degrees[i];
                    }
                }

                row
            })
            .collect();

        let out_rows: Vec<Vec<f64>> = (0..self.n_nodes)
            .into_par_iter()
            .map(|i| {
                let agg = &agg_rows[i];
                let mut out = vec![0.0_f64; self.n_features];
                for (f_out, out_value) in out.iter_mut().enumerate().take(self.n_features) {
                    let mut acc = 0.0_f64;
                    for (g, agg_value) in agg.iter().enumerate().take(self.n_features) {
                        acc += *agg_value * self.weights[g * self.n_features + f_out];
                    }
                    *out_value = acc.tanh();
                }
                out
            })
            .collect();

        let mut flat = Vec::with_capacity(self.n_nodes * self.n_features);
        for row in out_rows {
            flat.extend(row);
        }
        Ok(flat)
    }

    /// SC-mode forward pass using AND+popcount message passing.
    pub fn forward_sc(
        &self,
        node_features: &[f64],
        length: usize,
        seed: u64,
    ) -> Result<Vec<f64>, String> {
        if node_features.len() != self.n_nodes * self.n_features {
            return Err(format!(
                "node_features length mismatch: got {}, expected {}.",
                node_features.len(),
                self.n_nodes * self.n_features
            ));
        }
        if length == 0 {
            return Err("length must be > 0 for SC mode.".to_string());
        }

        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let words = length.div_ceil(64);

        let adj_packed = crate::bitstream::encode_matrix_prob_to_packed(
            &self.adj,
            self.n_nodes,
            self.n_nodes,
            length,
            words,
            &mut rng,
        );
        let feat_packed = crate::bitstream::encode_matrix_prob_to_packed(
            node_features,
            self.n_nodes,
            self.n_features,
            length,
            words,
            &mut rng,
        );

        let mut agg = vec![0.0_f64; self.n_nodes * self.n_features];
        for i in 0..self.n_nodes {
            for f in 0..self.n_features {
                let mut pop_total = 0_u64;
                for j in 0..self.n_nodes {
                    let a = &adj_packed[i * self.n_nodes + j];
                    let b = &feat_packed[j * self.n_features + f];
                    for w in 0..words {
                        pop_total += crate::bitstream::swar_popcount_word(a[w] & b[w]);
                    }
                }
                agg[i * self.n_features + f] = pop_total as f64 / length as f64;
            }
            if self.degrees[i] != 0.0 {
                for f in 0..self.n_features {
                    agg[i * self.n_features + f] /= self.degrees[i];
                }
            }
        }

        let agg_packed = crate::bitstream::encode_matrix_prob_to_packed(
            &agg,
            self.n_nodes,
            self.n_features,
            length,
            words,
            &mut rng,
        );
        let w_clamped: Vec<f64> = self.weights.iter().map(|w| w.clamp(0.0, 1.0)).collect();
        let w_packed = crate::bitstream::encode_matrix_prob_to_packed(
            &w_clamped,
            self.n_features,
            self.n_features,
            length,
            words,
            &mut rng,
        );

        let mut out = Vec::with_capacity(self.n_nodes * self.n_features);
        for i in 0..self.n_nodes {
            for f_out in 0..self.n_features {
                let mut pop_total = 0_u64;
                for g in 0..self.n_features {
                    let a = &agg_packed[i * self.n_features + g];
                    let b = &w_packed[g * self.n_features + f_out];
                    for w in 0..words {
                        pop_total += crate::bitstream::swar_popcount_word(a[w] & b[w]);
                    }
                }
                out.push((pop_total as f64 / length as f64).tanh());
            }
        }

        Ok(out)
    }

    /// Return a copy of the weight matrix.
    pub fn get_weights(&self) -> Vec<f64> {
        self.weights.clone()
    }

    /// Replace weight matrix.
    pub fn set_weights(&mut self, weights: Vec<f64>) -> Result<(), String> {
        if weights.len() != self.n_features * self.n_features {
            return Err(format!(
                "weights length mismatch: got {}, expected {}.",
                weights.len(),
                self.n_features * self.n_features
            ));
        }
        self.weights = weights;
        Ok(())
    }
}
