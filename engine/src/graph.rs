use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;

pub struct StochasticGraphLayer {
    pub n_nodes: usize,
    pub n_features: usize,
    pub adj: Vec<f64>,
    pub weights: Vec<f64>,
    pub degrees: Vec<f64>,
}

impl StochasticGraphLayer {
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

    pub fn get_weights(&self) -> Vec<f64> {
        self.weights.clone()
    }

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
