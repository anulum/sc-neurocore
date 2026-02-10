//! # Stochastic Attention
//!
//! Rate-mode and SC-mode attention primitives used by the Python bridge.

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;

pub struct StochasticAttention {
    /// Key/query feature dimension used by callers.
    pub dim_k: usize,
}

impl StochasticAttention {
    /// Create a stochastic attention operator.
    pub fn new(dim_k: usize) -> Self {
        Self { dim_k }
    }

    /// Rate-mode attention forward pass.
    ///
    /// Shapes:
    /// - `q`: `(q_rows, q_cols)`
    /// - `k`: `(k_rows, k_cols)`
    /// - `v`: `(v_rows, v_cols)`
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        q: &[f64],
        q_rows: usize,
        q_cols: usize,
        k: &[f64],
        k_rows: usize,
        k_cols: usize,
        v: &[f64],
        v_rows: usize,
        v_cols: usize,
    ) -> Result<Vec<f64>, String> {
        validate_shapes(q, q_rows, q_cols, k, k_rows, k_cols, v, v_rows, v_cols)?;

        let score_rows: Vec<Vec<f64>> = (0..q_rows)
            .into_par_iter()
            .map(|i| {
                let q_row = &q[i * q_cols..(i + 1) * q_cols];
                let mut row = vec![0.0_f64; k_rows];
                for j in 0..k_rows {
                    let k_row = &k[j * k_cols..(j + 1) * k_cols];
                    let mut dot = 0.0_f64;
                    for d in 0..q_cols {
                        dot += q_row[d] * k_row[d];
                    }
                    row[j] = dot;
                }
                row
            })
            .collect();

        let out_rows: Vec<Vec<f64>> = (0..q_rows)
            .into_par_iter()
            .map(|i| {
                let scores = &score_rows[i];
                let mut row_sum = scores.iter().sum::<f64>();
                if row_sum == 0.0 {
                    row_sum = 1.0;
                }

                let mut out = vec![0.0_f64; v_cols];
                for d in 0..v_cols {
                    let mut acc = 0.0_f64;
                    for j in 0..k_rows {
                        let weight = scores[j] / row_sum;
                        acc += weight * v[j * v_cols + d];
                    }
                    out[d] = acc;
                }
                out
            })
            .collect();

        Ok(flatten_rows(out_rows, q_rows, v_cols))
    }

    /// SC-mode attention forward pass with Bernoulli bitstream encoding.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_sc(
        &self,
        q: &[f64],
        q_rows: usize,
        q_cols: usize,
        k: &[f64],
        k_rows: usize,
        k_cols: usize,
        v: &[f64],
        v_rows: usize,
        v_cols: usize,
        length: usize,
        seed: u64,
    ) -> Result<Vec<f64>, String> {
        validate_shapes(q, q_rows, q_cols, k, k_rows, k_cols, v, v_rows, v_cols)?;
        if length == 0 {
            return Err("length must be > 0 for SC mode.".to_string());
        }

        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let words = length.div_ceil(64);

        let q_packed = crate::bitstream::encode_matrix_prob_to_packed(
            q, q_rows, q_cols, length, words, &mut rng,
        );
        let k_packed = crate::bitstream::encode_matrix_prob_to_packed(
            k, k_rows, k_cols, length, words, &mut rng,
        );
        let v_packed = crate::bitstream::encode_matrix_prob_to_packed(
            v, v_rows, v_cols, length, words, &mut rng,
        );

        let mut score_rows = vec![vec![0.0_f64; k_rows]; q_rows];
        for (i, score_row) in score_rows.iter_mut().enumerate().take(q_rows) {
            for (j, score_value) in score_row.iter_mut().enumerate().take(k_rows) {
                let mut pop_total = 0_u64;
                for d in 0..q_cols {
                    let q_idx = i * q_cols + d;
                    let k_idx = j * k_cols + d;
                    let qa = &q_packed[q_idx];
                    let kb = &k_packed[k_idx];
                    for w in 0..words {
                        pop_total += crate::bitstream::swar_popcount_word(qa[w] & kb[w]);
                    }
                }
                *score_value = pop_total as f64 / length as f64;
            }
        }

        let attn_weights: Vec<Vec<f64>> = score_rows
            .iter()
            .map(|row| {
                let mut row_sum = row.iter().sum::<f64>();
                if row_sum == 0.0 {
                    row_sum = 1.0;
                }
                row.iter().map(|x| x / row_sum).collect()
            })
            .collect();

        let attn_flat: Vec<f64> = attn_weights.into_iter().flatten().collect();
        let attn_packed = crate::bitstream::encode_matrix_prob_to_packed(
            &attn_flat, q_rows, k_rows, length, words, &mut rng,
        );

        let out_rows: Vec<Vec<f64>> = (0..q_rows)
            .into_par_iter()
            .map(|i| {
                let mut out = vec![0.0_f64; v_cols];
                for d in 0..v_cols {
                    let mut pop_total = 0_u64;
                    for j in 0..k_rows {
                        let a = &attn_packed[i * k_rows + j];
                        let b = &v_packed[j * v_cols + d];
                        for w in 0..words {
                            pop_total += crate::bitstream::swar_popcount_word(a[w] & b[w]);
                        }
                    }
                    out[d] = pop_total as f64 / length as f64;
                }
                out
            })
            .collect();

        Ok(flatten_rows(out_rows, q_rows, v_cols))
    }

    /// Multi-head attention: split Q/K/V columns across heads,
    /// run per-head attention, then concatenate outputs.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_multihead(
        &self,
        q: &[f64],
        q_rows: usize,
        q_total_cols: usize,
        k: &[f64],
        k_rows: usize,
        k_total_cols: usize,
        v: &[f64],
        v_rows: usize,
        v_total_cols: usize,
        n_heads: usize,
    ) -> Result<Vec<f64>, String> {
        if n_heads == 0 {
            return Err("n_heads must be > 0.".to_string());
        }
        if !q_total_cols.is_multiple_of(n_heads)
            || !k_total_cols.is_multiple_of(n_heads)
            || !v_total_cols.is_multiple_of(n_heads)
        {
            return Err(format!(
                "Total columns must be divisible by n_heads={}. Got Q={}, K={}, V={}.",
                n_heads, q_total_cols, k_total_cols, v_total_cols
            ));
        }
        if q.len() != q_rows * q_total_cols {
            return Err(format!(
                "Q data length mismatch: got {}, expected {}.",
                q.len(),
                q_rows * q_total_cols
            ));
        }
        if k.len() != k_rows * k_total_cols {
            return Err(format!(
                "K data length mismatch: got {}, expected {}.",
                k.len(),
                k_rows * k_total_cols
            ));
        }
        if v.len() != v_rows * v_total_cols {
            return Err(format!(
                "V data length mismatch: got {}, expected {}.",
                v.len(),
                v_rows * v_total_cols
            ));
        }

        let dk = q_total_cols / n_heads;
        let dk_k = k_total_cols / n_heads;
        let dv = v_total_cols / n_heads;

        if dk != dk_k {
            return Err(format!(
                "Q/K head dimensions must match: Q_head={}, K_head={}.",
                dk, dk_k
            ));
        }

        let head_outputs: Result<Vec<Vec<f64>>, String> = (0..n_heads)
            .into_par_iter()
            .map(|h| {
                let q_head = extract_head_columns(q, q_rows, q_total_cols, h, dk);
                let k_head = extract_head_columns(k, k_rows, k_total_cols, h, dk);
                let v_head = extract_head_columns(v, v_rows, v_total_cols, h, dv);
                self.forward(
                    &q_head, q_rows, dk, &k_head, k_rows, dk, &v_head, v_rows, dv,
                )
            })
            .collect();
        let head_outputs = head_outputs?;

        let out_cols = dv * n_heads;
        let mut out = Vec::with_capacity(q_rows * out_cols);
        for i in 0..q_rows {
            for head in head_outputs.iter().take(n_heads) {
                let head_row = &head[i * dv..(i + 1) * dv];
                out.extend_from_slice(head_row);
            }
        }
        Ok(out)
    }
}

#[allow(clippy::too_many_arguments)]
fn validate_shapes(
    q: &[f64],
    q_rows: usize,
    q_cols: usize,
    k: &[f64],
    k_rows: usize,
    k_cols: usize,
    v: &[f64],
    v_rows: usize,
    v_cols: usize,
) -> Result<(), String> {
    if q_cols != k_cols {
        return Err(format!(
            "Q/K dimension mismatch: q_cols={}, k_cols={}.",
            q_cols, k_cols
        ));
    }
    if k_rows != v_rows {
        return Err(format!(
            "K/V row mismatch: k_rows={}, v_rows={}.",
            k_rows, v_rows
        ));
    }
    if q.len() != q_rows * q_cols {
        return Err(format!(
            "Q data length mismatch: got {}, expected {}.",
            q.len(),
            q_rows * q_cols
        ));
    }
    if k.len() != k_rows * k_cols {
        return Err(format!(
            "K data length mismatch: got {}, expected {}.",
            k.len(),
            k_rows * k_cols
        ));
    }
    if v.len() != v_rows * v_cols {
        return Err(format!(
            "V data length mismatch: got {}, expected {}.",
            v.len(),
            v_rows * v_cols
        ));
    }
    Ok(())
}

fn flatten_rows(rows: Vec<Vec<f64>>, n_rows: usize, n_cols: usize) -> Vec<f64> {
    let mut flat = Vec::with_capacity(n_rows * n_cols);
    for row in rows {
        flat.extend(row);
    }
    flat
}

/// Extract one head slice from a row-major matrix.
fn extract_head_columns(
    matrix: &[f64],
    rows: usize,
    total_cols: usize,
    head_idx: usize,
    head_cols: usize,
) -> Vec<f64> {
    let offset = head_idx * head_cols;
    let mut out = Vec::with_capacity(rows * head_cols);
    for i in 0..rows {
        let row_start = i * total_cols + offset;
        out.extend_from_slice(&matrix[row_start..row_start + head_cols]);
    }
    out
}
