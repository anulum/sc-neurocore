// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Stochastic Graph Layer

//! # Stochastic Graph Layer
//!
//! Graph message-passing layer with both rate-mode and SC-mode forward paths.
//! Supports dense O(n²) and sparse CSR O(nnz) adjacency storage.

use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;

/// Compressed Sparse Row storage for adjacency matrices.
#[derive(Clone, Debug)]
pub struct CsrMatrix {
    /// Length n_rows + 1. row_offsets[i]..row_offsets[i+1] indexes into col_indices/values.
    pub row_offsets: Vec<usize>,
    pub col_indices: Vec<usize>,
    pub values: Vec<f64>,
    pub n_rows: usize,
    pub n_cols: usize,
}

impl CsrMatrix {
    pub fn new(
        row_offsets: Vec<usize>,
        col_indices: Vec<usize>,
        values: Vec<f64>,
        n_rows: usize,
        n_cols: usize,
    ) -> Result<Self, String> {
        if row_offsets.len() != n_rows + 1 {
            return Err(format!(
                "row_offsets length {} != n_rows + 1 = {}",
                row_offsets.len(),
                n_rows + 1
            ));
        }
        if col_indices.len() != values.len() {
            return Err(format!(
                "col_indices length {} != values length {}",
                col_indices.len(),
                values.len()
            ));
        }
        let nnz = *row_offsets.last().ok_or("row_offsets must not be empty")?;
        if col_indices.len() != nnz {
            return Err(format!(
                "col_indices length {} != nnz from row_offsets {}",
                col_indices.len(),
                nnz
            ));
        }
        // Row offsets must start at 0 and be non-decreasing, so every row slice
        // row_offsets[i]..row_offsets[i + 1] is a valid, in-bounds range of the value array.
        if row_offsets[0] != 0 {
            return Err(format!(
                "row_offsets must start at 0, got {}",
                row_offsets[0]
            ));
        }
        if let Some(pair) = row_offsets.windows(2).find(|w| w[0] > w[1]) {
            return Err(format!(
                "row_offsets must be non-decreasing, got {} then {}",
                pair[0], pair[1]
            ));
        }
        // Every column index must be in range so consumers (e.g. to_dense) cannot index
        // out of bounds. Without this a malformed CSR was accepted and panicked on use.
        if let Some(&col) = col_indices.iter().find(|&&c| c >= n_cols) {
            return Err(format!("col_index {col} out of range for n_cols {n_cols}"));
        }
        Ok(Self {
            row_offsets,
            col_indices,
            values,
            n_rows,
            n_cols,
        })
    }

    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// Convert dense row-major matrix to CSR, keeping entries with |value| > threshold.
    pub fn from_dense(dense: &[f64], n_rows: usize, n_cols: usize, threshold: f64) -> Self {
        let mut row_offsets = Vec::with_capacity(n_rows + 1);
        let mut col_indices = Vec::new();
        let mut values = Vec::new();

        row_offsets.push(0);
        for i in 0..n_rows {
            for j in 0..n_cols {
                let v = dense[i * n_cols + j];
                if v.abs() > threshold {
                    col_indices.push(j);
                    values.push(v);
                }
            }
            row_offsets.push(col_indices.len());
        }

        Self {
            row_offsets,
            col_indices,
            values,
            n_rows,
            n_cols,
        }
    }

    /// Expand back to dense row-major layout.
    pub fn to_dense(&self) -> Vec<f64> {
        let mut dense = vec![0.0_f64; self.n_rows * self.n_cols];
        for i in 0..self.n_rows {
            for idx in self.row_offsets[i]..self.row_offsets[i + 1] {
                dense[i * self.n_cols + self.col_indices[idx]] = self.values[idx];
            }
        }
        dense
    }

    /// Row-sum (degree) for row i.
    fn row_sum(&self, i: usize) -> f64 {
        let mut s = 0.0_f64;
        for idx in self.row_offsets[i]..self.row_offsets[i + 1] {
            s += self.values[idx];
        }
        s
    }
}

/// Adjacency storage: dense or sparse CSR.
pub enum AdjStorage {
    Dense { adj: Vec<f64> },
    Sparse { csr: CsrMatrix },
}

pub struct StochasticGraphLayer {
    pub n_nodes: usize,
    pub n_features: usize,
    pub storage: AdjStorage,
    pub weights: Vec<f64>,
    pub degrees: Vec<f64>,
}

fn random_weights(n_features: usize, seed: u64) -> Vec<f64> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut weights = vec![0.0_f64; n_features * n_features];
    for w in &mut weights {
        *w = rng.random::<f64>();
    }
    weights
}

fn dense_degrees(adj: &[f64], n: usize) -> Vec<f64> {
    let mut degrees = vec![0.0_f64; n];
    for i in 0..n {
        let mut sum = 0.0_f64;
        for j in 0..n {
            sum += adj[i * n + j];
        }
        degrees[i] = sum;
    }
    degrees
}

fn csr_degrees(csr: &CsrMatrix) -> Vec<f64> {
    (0..csr.n_rows).map(|i| csr.row_sum(i)).collect()
}

impl StochasticGraphLayer {
    /// Construct from dense adjacency matrix (backwards-compatible API).
    pub fn new(adj_flat: Vec<f64>, n_nodes: usize, n_features: usize, seed: u64) -> Self {
        assert_eq!(
            adj_flat.len(),
            n_nodes * n_nodes,
            "adj_flat must have length n_nodes * n_nodes",
        );
        let degrees = dense_degrees(&adj_flat, n_nodes);
        Self {
            n_nodes,
            n_features,
            storage: AdjStorage::Dense { adj: adj_flat },
            weights: random_weights(n_features, seed),
            degrees,
        }
    }

    /// Construct from pre-built CSR adjacency.
    pub fn new_sparse(csr: CsrMatrix, n_features: usize, seed: u64) -> Result<Self, String> {
        if csr.n_rows != csr.n_cols {
            return Err(format!(
                "CSR must be square, got {}x{}",
                csr.n_rows, csr.n_cols
            ));
        }
        let n_nodes = csr.n_rows;
        let degrees = csr_degrees(&csr);
        Ok(Self {
            n_nodes,
            n_features,
            storage: AdjStorage::Sparse { csr },
            weights: random_weights(n_features, seed),
            degrees,
        })
    }

    /// Convert dense adjacency to CSR if density < `density_threshold` (default 0.3).
    pub fn from_dense_auto(
        adj_flat: Vec<f64>,
        n_nodes: usize,
        n_features: usize,
        seed: u64,
        density_threshold: f64,
    ) -> Self {
        assert_eq!(adj_flat.len(), n_nodes * n_nodes);
        let total = (n_nodes * n_nodes) as f64;
        let nnz = adj_flat.iter().filter(|v| v.abs() > 1e-15).count() as f64;
        let density = nnz / total;

        if density < density_threshold {
            let csr = CsrMatrix::from_dense(&adj_flat, n_nodes, n_nodes, 1e-15);
            let degrees = csr_degrees(&csr);
            Self {
                n_nodes,
                n_features,
                storage: AdjStorage::Sparse { csr },
                weights: random_weights(n_features, seed),
                degrees,
            }
        } else {
            Self::new(adj_flat, n_nodes, n_features, seed)
        }
    }

    /// True if using sparse CSR storage.
    pub fn is_sparse(&self) -> bool {
        matches!(self.storage, AdjStorage::Sparse { .. })
    }

    fn validate_features(&self, node_features: &[f64]) -> Result<(), String> {
        if node_features.len() != self.n_nodes * self.n_features {
            return Err(format!(
                "node_features length mismatch: got {}, expected {}.",
                node_features.len(),
                self.n_nodes * self.n_features
            ));
        }
        Ok(())
    }

    /// Aggregate + weight transform + tanh, dispatched on storage.
    fn aggregate_and_transform(&self, agg_flat: &[f64]) -> Vec<f64> {
        let out_rows: Vec<Vec<f64>> = (0..self.n_nodes)
            .into_par_iter()
            .map(|i| {
                let agg = &agg_flat[i * self.n_features..(i + 1) * self.n_features];
                let mut out = vec![0.0_f64; self.n_features];
                for (f_out, out_val) in out.iter_mut().enumerate().take(self.n_features) {
                    let mut acc = 0.0_f64;
                    for (g, agg_val) in agg.iter().enumerate().take(self.n_features) {
                        acc += *agg_val * self.weights[g * self.n_features + f_out];
                    }
                    *out_val = acc.tanh();
                }
                out
            })
            .collect();
        let mut flat = Vec::with_capacity(self.n_nodes * self.n_features);
        for row in out_rows {
            flat.extend(row);
        }
        flat
    }

    /// Rate-mode graph forward pass.
    pub fn forward(&self, node_features: &[f64]) -> Result<Vec<f64>, String> {
        self.validate_features(node_features)?;

        let mut agg = vec![0.0_f64; self.n_nodes * self.n_features];

        match &self.storage {
            AdjStorage::Dense { adj } => {
                let agg_rows: Vec<Vec<f64>> = (0..self.n_nodes)
                    .into_par_iter()
                    .map(|i| {
                        let mut row = vec![0.0_f64; self.n_features];
                        for f in 0..self.n_features {
                            let mut acc = 0.0_f64;
                            for j in 0..self.n_nodes {
                                acc += adj[i * self.n_nodes + j]
                                    * node_features[j * self.n_features + f];
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
                for (i, row) in agg_rows.into_iter().enumerate() {
                    agg[i * self.n_features..(i + 1) * self.n_features].copy_from_slice(&row);
                }
            }
            AdjStorage::Sparse { csr } => {
                let agg_rows: Vec<Vec<f64>> = (0..self.n_nodes)
                    .into_par_iter()
                    .map(|i| {
                        let mut row = vec![0.0_f64; self.n_features];
                        for idx in csr.row_offsets[i]..csr.row_offsets[i + 1] {
                            let j = csr.col_indices[idx];
                            let a_ij = csr.values[idx];
                            for f in 0..self.n_features {
                                row[f] += a_ij * node_features[j * self.n_features + f];
                            }
                        }
                        if self.degrees[i] != 0.0 {
                            for x in &mut row {
                                *x /= self.degrees[i];
                            }
                        }
                        row
                    })
                    .collect();
                for (i, row) in agg_rows.into_iter().enumerate() {
                    agg[i * self.n_features..(i + 1) * self.n_features].copy_from_slice(&row);
                }
            }
        }

        Ok(self.aggregate_and_transform(&agg))
    }

    /// SC-mode forward pass using AND+popcount message passing.
    ///
    /// Dense: encodes full adjacency matrix.
    /// Sparse: encodes only non-zero CSR entries, avoiding O(n^2) dense expansion.
    pub fn forward_sc(
        &self,
        node_features: &[f64],
        length: usize,
        seed: u64,
    ) -> Result<Vec<f64>, String> {
        self.validate_features(node_features)?;
        if length == 0 {
            return Err("length must be > 0 for SC mode.".to_string());
        }

        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let words = length.div_ceil(64);

        let feat_packed = crate::bitstream::encode_matrix_prob_to_packed(
            node_features,
            self.n_nodes,
            self.n_features,
            length,
            words,
            &mut rng,
        );

        let mut agg = vec![0.0_f64; self.n_nodes * self.n_features];

        match &self.storage {
            AdjStorage::Dense { adj } => {
                let adj_packed = crate::bitstream::encode_matrix_prob_to_packed(
                    adj,
                    self.n_nodes,
                    self.n_nodes,
                    length,
                    words,
                    &mut rng,
                );
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
                }
            }
            AdjStorage::Sparse { csr } => {
                // Encode only non-zero adjacency values (nnz entries, not n^2)
                let nnz = csr.nnz();
                let adj_vals_clamped: Vec<f64> =
                    csr.values.iter().map(|v| v.clamp(0.0, 1.0)).collect();
                let adj_packed = crate::bitstream::encode_matrix_prob_to_packed(
                    &adj_vals_clamped,
                    1,
                    nnz,
                    length,
                    words,
                    &mut rng,
                );
                for i in 0..self.n_nodes {
                    #[allow(clippy::needless_range_loop)]
                    for idx in csr.row_offsets[i]..csr.row_offsets[i + 1] {
                        let j = csr.col_indices[idx];
                        let a = &adj_packed[idx];
                        for f in 0..self.n_features {
                            let b = &feat_packed[j * self.n_features + f];
                            let mut pop = 0_u64;
                            for w in 0..words {
                                pop += crate::bitstream::swar_popcount_word(a[w] & b[w]);
                            }
                            agg[i * self.n_features + f] += pop as f64 / length as f64;
                        }
                    }
                }
            }
        }

        for i in 0..self.n_nodes {
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
