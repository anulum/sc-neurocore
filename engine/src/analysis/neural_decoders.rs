// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Foundation-model neural population decoder primitives

//! Core compute kernels for POYO+, POSSM, NDT3, and CEBRA decoders.
//!
//! These are the hot-path operations that benefit from Rust acceleration:
//! spike tokenisation, sinusoidal position encoding, scaled dot-product
//! attention, diagonal SSM step, and InfoNCE contrastive loss.

use rayon::prelude::*;

/// Token: (unit_id, timestamp_ms).
pub type SpikeToken = (usize, f64);

/// Convert binary spike trains to sorted (unit_id, timestamp) tokens.
///
/// Azabou et al. (2023), NeurIPS; Ryoo et al. (2025), ICLR.
/// Each spike in each train produces one token. Tokens are sorted by time.
pub fn tokenise_spikes(trains: &[&[i32]], dt: f64) -> Vec<SpikeToken> {
    let mut tokens: Vec<SpikeToken> = trains
        .par_iter()
        .enumerate()
        .flat_map_iter(|(uid, train)| {
            train
                .iter()
                .enumerate()
                .filter(|(_, &v)| v != 0)
                .map(move |(idx, _)| (uid, idx as f64 * dt))
        })
        .collect();
    tokens.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    tokens
}

/// Sinusoidal position encoding. Vaswani et al. (2017).
///
/// PE(t, 2i)   = sin(t / 10000^{2i/d})
/// PE(t, 2i+1) = cos(t / 10000^{2i/d})
///
/// Output: flat row-major [n_timestamps × d_model].
pub fn sinusoidal_position_encode(timestamps: &[f64], d_model: usize) -> Vec<f64> {
    let n = timestamps.len();
    let mut pe = vec![0.0_f64; n * d_model];
    let half_d = d_model / 2 + d_model % 2;
    let divisors: Vec<f64> = (0..half_d)
        .map(|i| 10000.0_f64.powf(2.0 * i as f64 / d_model as f64))
        .collect();

    pe.par_chunks_mut(d_model)
        .enumerate()
        .for_each(|(row, pe_row)| {
            let t = timestamps[row];
            for (k, div) in divisors.iter().enumerate() {
                let col_sin = 2 * k;
                let col_cos = 2 * k + 1;
                let angle = t / div;
                pe_row[col_sin] = angle.sin();
                if col_cos < d_model {
                    pe_row[col_cos] = angle.cos();
                }
            }
        });
    pe
}

/// Scaled dot-product attention.
///
/// Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
///
/// All matrices are row-major flat: Q [nq × d], K [nk × d], V [nk × d].
/// Output: [nq × d].
pub fn scaled_dot_product_attention(
    queries: &[f64],
    keys: &[f64],
    values: &[f64],
    nq: usize,
    nk: usize,
    d: usize,
) -> Vec<f64> {
    let inv_sqrt_d = 1.0 / (d as f64).sqrt();
    let mut output = vec![0.0_f64; nq * d];

    output
        .par_chunks_mut(d)
        .enumerate()
        .for_each(|(i, out_row)| {
            let q_row = &queries[i * d..(i + 1) * d];
            // Compute scores
            let mut scores = vec![0.0_f64; nk];
            let mut max_score = f64::NEG_INFINITY;
            for j in 0..nk {
                let k_row = &keys[j * d..(j + 1) * d];
                let mut dot = 0.0;
                for f in 0..d {
                    dot += q_row[f] * k_row[f];
                }
                scores[j] = dot * inv_sqrt_d;
                if scores[j] > max_score {
                    max_score = scores[j];
                }
            }
            // Stable softmax
            let mut sum_exp = 0.0;
            for s in &mut scores {
                *s = (*s - max_score).exp();
                sum_exp += *s;
            }
            let inv_sum = 1.0 / (sum_exp + 1e-30);
            for s in &mut scores {
                *s *= inv_sum;
            }
            // Weighted sum of values
            for j in 0..nk {
                let w = scores[j];
                let v_row = &values[j * d..(j + 1) * d];
                for f in 0..d {
                    out_row[f] += w * v_row[f];
                }
            }
        });
    output
}

/// Gaussian attention. Li et al. (2025), scKGBERT.
///
/// α_ij = exp(-||q_i - k_j||² / (2σ²)) / Σ_m exp(-||q_i - k_m||² / (2σ²))
///
/// Q [nq × d], K [nk × d], V [nk × d]. Output: [nq × d].
pub fn gaussian_attention(
    queries: &[f64],
    keys: &[f64],
    values: &[f64],
    nq: usize,
    nk: usize,
    d: usize,
    sigma: f64,
) -> Vec<f64> {
    let inv_2sigma2 = 1.0 / (2.0 * sigma * sigma);
    let mut output = vec![0.0_f64; nq * d];

    output
        .par_chunks_mut(d)
        .enumerate()
        .for_each(|(i, out_row)| {
            let q_row = &queries[i * d..(i + 1) * d];
            let mut log_weights = vec![0.0_f64; nk];
            let mut max_lw = f64::NEG_INFINITY;
            for j in 0..nk {
                let k_row = &keys[j * d..(j + 1) * d];
                let mut dist_sq = 0.0;
                for f in 0..d {
                    let diff = q_row[f] - k_row[f];
                    dist_sq += diff * diff;
                }
                log_weights[j] = -dist_sq * inv_2sigma2;
                if log_weights[j] > max_lw {
                    max_lw = log_weights[j];
                }
            }
            let mut sum_exp = 0.0;
            for lw in &mut log_weights {
                *lw = (*lw - max_lw).exp();
                sum_exp += *lw;
            }
            let inv_sum = 1.0 / (sum_exp + 1e-30);
            for j in 0..nk {
                let w = log_weights[j] * inv_sum;
                let v_row = &values[j * d..(j + 1) * d];
                for f in 0..d {
                    out_row[f] += w * v_row[f];
                }
            }
        });
    output
}

/// Diagonal SSM step. Gu et al. (2022), S4D; Ryoo et al. (2025), POSSM.
///
/// h_t = A_bar ⊙ h_{t-1} + B_bar x_t
/// y_t = Re(C h_t) + D x_t
///
/// A_bar is complex diagonal [d_state] (re, im interleaved → 2 * d_state floats).
/// B_bar is [d_state × d_model] complex flat (re, im interleaved).
/// C is [d_model × d_state] complex flat.
/// D is [d_model × d_model] real flat.
/// h is [d_state] complex (re, im interleaved).
/// x is [d_model] real input.
///
/// Returns y [d_model] and updates h in place.
pub fn ssm_step_diagonal(
    a_bar_re: &[f64],
    a_bar_im: &[f64],
    b_bar_re: &[f64],
    b_bar_im: &[f64],
    c_re: &[f64],
    c_im: &[f64],
    d_mat: &[f64],
    h_re: &mut [f64],
    h_im: &mut [f64],
    x: &[f64],
    d_state: usize,
    d_model: usize,
) -> Vec<f64> {
    // h_t = A_bar ⊙ h_{t-1} + B_bar x_t
    for s in 0..d_state {
        // Complex multiply: (a_re + i*a_im) * (h_re + i*h_im)
        let new_re = a_bar_re[s] * h_re[s] - a_bar_im[s] * h_im[s];
        let new_im = a_bar_re[s] * h_im[s] + a_bar_im[s] * h_re[s];
        // B_bar @ x: B_bar[s, :] . x
        let mut bx_re = 0.0;
        let mut bx_im = 0.0;
        for m in 0..d_model {
            bx_re += b_bar_re[s * d_model + m] * x[m];
            bx_im += b_bar_im[s * d_model + m] * x[m];
        }
        h_re[s] = new_re + bx_re;
        h_im[s] = new_im + bx_im;
    }

    // y_t = Re(C h_t) + D x_t
    let mut y = vec![0.0_f64; d_model];
    for m in 0..d_model {
        let mut ch_re = 0.0;
        for s in 0..d_state {
            // Re(C[m,s] * h[s]) = C_re*h_re - C_im*h_im
            ch_re += c_re[m * d_state + s] * h_re[s] - c_im[m * d_state + s] * h_im[s];
        }
        let mut dx = 0.0;
        for m2 in 0..d_model {
            dx += d_mat[m * d_model + m2] * x[m2];
        }
        y[m] = ch_re + dx;
    }
    y
}

/// InfoNCE contrastive loss. van den Oord et al. (2018); CEBRA.
///
/// L = -(1/N) Σ_i log( exp(sim(z_i, z_i^+)/τ) / Σ_j exp(sim(z_i, z_j)/τ) )
/// where sim(a, b) = cosine similarity.
///
/// anchors, positives: [N × d] row-major flat.
pub fn infonce_loss(
    anchors: &[f64],
    positives: &[f64],
    n: usize,
    d: usize,
    temperature: f64,
) -> f64 {
    if n == 0 || d == 0 {
        return 0.0;
    }
    let inv_tau = 1.0 / temperature;

    // Normalise
    let norm = |v: &[f64]| -> Vec<f64> {
        let mut out = v.to_vec();
        for i in 0..n {
            let row = &mut out[i * d..(i + 1) * d];
            let nrm: f64 = row.iter().map(|x| x * x).sum::<f64>().sqrt() + 1e-30;
            for x in row.iter_mut() {
                *x /= nrm;
            }
        }
        out
    };

    let a_norm = norm(anchors);
    let p_norm = norm(positives);

    let total_loss: f64 = (0..n)
        .into_par_iter()
        .map(|i| {
            let a_row = &a_norm[i * d..(i + 1) * d];
            // Positive similarity (diagonal)
            let p_row = &p_norm[i * d..(i + 1) * d];
            let pos_sim: f64 = a_row.iter().zip(p_row).map(|(a, p)| a * p).sum();

            // All similarities
            let mut max_sim = f64::NEG_INFINITY;
            let mut sims = vec![0.0_f64; n];
            for j in 0..n {
                let pj = &p_norm[j * d..(j + 1) * d];
                let sim: f64 = a_row.iter().zip(pj).map(|(a, p)| a * p).sum();
                sims[j] = sim * inv_tau;
                if sims[j] > max_sim {
                    max_sim = sims[j];
                }
            }
            let sum_exp: f64 = sims.iter().map(|s| (s - max_sim).exp()).sum();
            let log_softmax = pos_sim * inv_tau - max_sim - sum_exp.ln();
            -log_softmax
        })
        .sum();

    total_loss / n as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenise_empty() {
        let tokens = tokenise_spikes(&[], 1.0);
        assert!(tokens.is_empty());
    }

    #[test]
    fn test_tokenise_single() {
        let train = vec![0, 0, 1, 0, 0];
        let tokens = tokenise_spikes(&[&train], 0.5);
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].0, 0);
        assert!((tokens[0].1 - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_tokenise_sorted() {
        let t0 = vec![0, 0, 0, 0, 1]; // spike at t=4
        let t1 = vec![0, 1, 0, 0, 0]; // spike at t=1
        let tokens = tokenise_spikes(&[&t0, &t1], 1.0);
        assert_eq!(tokens.len(), 2);
        assert!(tokens[0].1 <= tokens[1].1);
    }

    #[test]
    fn test_sinusoidal_pe_shape() {
        let ts = vec![0.0, 1.0, 2.0];
        let pe = sinusoidal_position_encode(&ts, 8);
        assert_eq!(pe.len(), 3 * 8);
    }

    #[test]
    fn test_sinusoidal_pe_zero() {
        let pe = sinusoidal_position_encode(&[0.0], 4);
        assert!((pe[0] - 0.0).abs() < 1e-10); // sin(0)
        assert!((pe[1] - 1.0).abs() < 1e-10); // cos(0)
    }

    #[test]
    fn test_attention_shape() {
        let q = vec![1.0, 0.0, 0.0, 1.0]; // 2×2
        let k = vec![1.0, 0.0, 0.0, 1.0, 0.5, 0.5]; // 3×2
        let v = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 3×2
        let out = scaled_dot_product_attention(&q, &k, &v, 2, 3, 2);
        assert_eq!(out.len(), 4);
    }

    #[test]
    fn test_gaussian_attention_concentrates() {
        // Query at origin, one key at origin, one far away
        let q = vec![0.0, 0.0];
        let k = vec![0.0, 0.0, 100.0, 100.0];
        let v = vec![1.0, 0.0, 0.0, 1.0];
        let out = gaussian_attention(&q, &k, &v, 1, 2, 2, 0.01);
        // Should concentrate on first key (distance=0)
        assert!((out[0] - 1.0).abs() < 1e-3);
        assert!((out[1] - 0.0).abs() < 1e-3);
    }

    #[test]
    fn test_ssm_step_output_size() {
        let d_state = 2;
        let d_model = 3;
        let a_re = vec![0.9, 0.8];
        let a_im = vec![0.1, 0.2];
        let b_re = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]; // 2×3
        let b_im = vec![0.0; 6];
        let c_re = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]; // 3×2
        let c_im = vec![0.0; 6];
        let d_mat = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]; // 3×3 identity
        let mut h_re = vec![0.0; 2];
        let mut h_im = vec![0.0; 2];
        let x = vec![1.0, 0.0, 0.0];
        let y = ssm_step_diagonal(
            &a_re, &a_im, &b_re, &b_im, &c_re, &c_im, &d_mat, &mut h_re, &mut h_im, &x, d_state,
            d_model,
        );
        assert_eq!(y.len(), 3);
    }

    #[test]
    fn test_ssm_state_update() {
        let d_state = 1;
        let d_model = 1;
        let mut h_re = vec![0.0];
        let mut h_im = vec![0.0];
        ssm_step_diagonal(
            &[0.9],
            &[0.0],
            &[1.0],
            &[0.0],
            &[1.0],
            &[0.0],
            &[0.0],
            &mut h_re,
            &mut h_im,
            &[1.0],
            d_state,
            d_model,
        );
        // h_re should be 0.9 * 0 + 1.0 * 1.0 = 1.0
        assert!((h_re[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_infonce_identical_pairs() {
        let d = 4;
        let n = 3;
        let data = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let loss = infonce_loss(&data, &data, n, d, 1.0);
        // Identical pairs: each positive is the anchor itself
        // cosine similarity = 1 for diagonal, < 1 for off-diag
        assert!(loss >= 0.0);
    }

    #[test]
    fn test_infonce_temperature() {
        let d = 2;
        let n = 4;
        let a = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0, -1.0, 1.0];
        let p = a.clone();
        let loss_cold = infonce_loss(&a, &p, n, d, 0.1);
        let loss_hot = infonce_loss(&a, &p, n, d, 10.0);
        assert!(loss_cold < loss_hot);
    }
}
