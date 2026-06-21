// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — GPFA: Gaussian Process Factor Analysis
//
// Yu, Cunningham et al. (2009) J. Neurophysiol. 102:614-635.

use super::basic;

/// EM output: `(trajectories, C, d, R_diag, log_likelihoods)`.
pub type GpfaEmOutput = (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>);

// ── helpers ─────────────────────────────────────────────────────────

/// Squared-exponential GP kernel for `n` time points.
fn gp_kernel(n: usize, tau: f64, sigma: f64) -> Vec<f64> {
    let mut k = vec![0.0f64; n * n];
    let tau_sq = tau * tau + 1e-12;
    let sigma_sq = sigma * sigma;
    for i in 0..n {
        for j in 0..n {
            let diff = i as f64 - j as f64;
            k[i * n + j] = sigma_sq * (-0.5 * diff * diff / tau_sq).exp();
        }
    }
    k
}

/// Gauss-Jordan inverse for n x n matrix.
fn mat_inv(a: &[f64], n: usize) -> Vec<f64> {
    let mut aug = vec![0.0f64; n * 2 * n];
    for i in 0..n {
        for j in 0..n {
            aug[i * 2 * n + j] = a[i * n + j];
        }
        aug[i * 2 * n + n + i] = 1.0;
    }
    for col in 0..n {
        let mut max_row = col;
        let mut max_val = aug[col * 2 * n + col].abs();
        for row in col + 1..n {
            let v = aug[row * 2 * n + col].abs();
            if v > max_val {
                max_val = v;
                max_row = row;
            }
        }
        if max_val < 1e-30 {
            continue;
        }
        if max_row != col {
            for k in 0..2 * n {
                aug.swap(col * 2 * n + k, max_row * 2 * n + k);
            }
        }
        let pivot = aug[col * 2 * n + col];
        for k in 0..2 * n {
            aug[col * 2 * n + k] /= pivot;
        }
        for row in 0..n {
            if row == col {
                continue;
            }
            let factor = aug[row * 2 * n + col];
            for k in 0..2 * n {
                aug[row * 2 * n + k] -= factor * aug[col * 2 * n + k];
            }
        }
    }
    let mut inv = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            inv[i * n + j] = aug[i * 2 * n + n + j];
        }
    }
    inv
}

/// Solve A x = b via Gauss-Jordan (A is n x n, b is n x m, returns x as n x m).
fn mat_solve(a: &[f64], b: &[f64], n: usize, m: usize) -> Vec<f64> {
    let mut aug = vec![0.0f64; n * (n + m)];
    let w = n + m;
    for i in 0..n {
        for j in 0..n {
            aug[i * w + j] = a[i * n + j];
        }
        for j in 0..m {
            aug[i * w + n + j] = b[i * m + j];
        }
    }
    for col in 0..n {
        let mut max_row = col;
        let mut max_val = aug[col * w + col].abs();
        for row in col + 1..n {
            let v = aug[row * w + col].abs();
            if v > max_val {
                max_val = v;
                max_row = row;
            }
        }
        if max_val < 1e-30 {
            continue;
        }
        if max_row != col {
            for k in 0..w {
                aug.swap(col * w + k, max_row * w + k);
            }
        }
        let pivot = aug[col * w + col];
        for k in 0..w {
            aug[col * w + k] /= pivot;
        }
        for row in 0..n {
            if row == col {
                continue;
            }
            let factor = aug[row * w + col];
            for k in 0..w {
                aug[row * w + k] -= factor * aug[col * w + k];
            }
        }
    }
    let mut x = vec![0.0f64; n * m];
    for i in 0..n {
        for j in 0..m {
            x[i * m + j] = aug[i * w + n + j];
        }
    }
    x
}

/// Sign and natural log of the absolute determinant of an `n x n` matrix, via LU
/// with partial pivoting. Matches `numpy.linalg.slogdet`.
fn mat_slogdet(a: &[f64], n: usize) -> (f64, f64) {
    let mut m = a.to_vec();
    let mut sign = 1.0f64;
    let mut log_abs = 0.0f64;
    for col in 0..n {
        let mut max_row = col;
        let mut max_val = m[col * n + col].abs();
        for row in col + 1..n {
            let v = m[row * n + col].abs();
            if v > max_val {
                max_val = v;
                max_row = row;
            }
        }
        if max_val < 1e-300 {
            return (0.0, f64::NEG_INFINITY);
        }
        if max_row != col {
            for k in 0..n {
                m.swap(col * n + k, max_row * n + k);
            }
            sign = -sign;
        }
        let pivot = m[col * n + col];
        if pivot < 0.0 {
            sign = -sign;
        }
        log_abs += pivot.abs().ln();
        for row in col + 1..n {
            let factor = m[row * n + col] / pivot;
            for k in col..n {
                m[row * n + k] -= factor * m[col * n + k];
            }
        }
    }
    (sign, log_abs)
}

/// E-step: compute posterior p(x|y).
fn gpfa_e_step(
    y: &[f64],          // n_neurons x n_bins (row-major)
    c: &[f64],          // n_neurons x n_latents
    d: &[f64],          // n_neurons
    r_diag: &[f64],     // n_neurons
    k_all: &[Vec<f64>], // n_latents kernels, each n_bins x n_bins
    n_neurons: usize,
    n_bins: usize,
    n_latents: usize,
) -> (Vec<f64>, Vec<f64>) {
    // x_post (n_latents x n_bins), xx_post (n_latents x n_latents)
    let kt = n_latents * n_bins;

    // R^{-1}
    let r_inv: Vec<f64> = r_diag.iter().map(|&r| 1.0 / (r + 1e-10)).collect();

    // C^T R^{-1} C (n_latents x n_latents)
    let mut ct_rinv_c = vec![0.0f64; n_latents * n_latents];
    for i in 0..n_latents {
        for j in 0..n_latents {
            let mut s = 0.0;
            for k in 0..n_neurons {
                s += c[k * n_latents + i] * r_inv[k] * c[k * n_latents + j];
            }
            ct_rinv_c[i * n_latents + j] = s;
        }
    }

    // C^T R^{-1} (n_latents x n_neurons)
    let mut ct_rinv = vec![0.0f64; n_latents * n_neurons];
    for i in 0..n_latents {
        for k in 0..n_neurons {
            ct_rinv[i * n_neurons + k] = c[k * n_latents + i] * r_inv[k];
        }
    }

    // Build precision (kt x kt)
    let mut prec = vec![0.0f64; kt * kt];
    for j in 0..n_latents {
        let slj = j * n_bins;
        // K_j^{-1}
        let mut k_reg = k_all[j].clone();
        for i in 0..n_bins {
            k_reg[i * n_bins + i] += 1e-6;
        }
        let k_eye = vec![0.0f64; n_bins * n_bins]
            .iter()
            .enumerate()
            .map(|(idx, _)| {
                if idx / n_bins == idx % n_bins {
                    1.0
                } else {
                    0.0
                }
            })
            .collect::<Vec<f64>>();
        let k_inv = mat_solve(&k_reg, &k_eye, n_bins, n_bins);

        for i in 0..n_bins {
            for jj in 0..n_bins {
                prec[(slj + i) * kt + (slj + jj)] = k_inv[i * n_bins + jj]
                    + ct_rinv_c[j * n_latents + j] * if i == jj { 1.0 } else { 0.0 };
            }
        }
        for k in 0..n_latents {
            if k != j {
                let slk = k * n_bins;
                for i in 0..n_bins {
                    prec[(slj + i) * kt + (slk + i)] = ct_rinv_c[j * n_latents + k];
                }
            }
        }
    }

    // RHS
    let mut rhs = vec![0.0f64; kt];
    // Y_centered = Y - d[:, None]
    for t in 0..n_bins {
        // v = C^T R^{-1} (y_t - d)
        for j in 0..n_latents {
            let mut s = 0.0;
            for k in 0..n_neurons {
                s += ct_rinv[j * n_neurons + k] * (y[k * n_bins + t] - d[k]);
            }
            rhs[j * n_bins + t] = s;
        }
    }

    // Regularise precision
    for i in 0..kt {
        prec[i * kt + i] += 1e-8;
    }

    // Solve prec * x_vec = rhs
    let rhs_col: Vec<f64> = rhs.clone();
    let x_vec = mat_solve(&prec, &rhs_col, kt, 1);

    // Posterior covariance (for E[xx^T])
    let eye_kt: Vec<f64> = (0..kt * kt)
        .map(|idx| if idx / kt == idx % kt { 1.0 } else { 0.0 })
        .collect();
    let sigma_post = mat_solve(&prec, &eye_kt, kt, kt);

    // E[xx^T] per timepoint
    let mut xx_post = vec![0.0f64; n_latents * n_latents];
    for t in 0..n_bins {
        for j in 0..n_latents {
            let xj = x_vec[j * n_bins + t];
            for k in 0..n_latents {
                let xk = x_vec[k * n_bins + t];
                xx_post[j * n_latents + k] +=
                    xj * xk + sigma_post[(j * n_bins + t) * kt + (k * n_bins + t)];
            }
        }
    }

    (x_vec, xx_post)
}

/// M-step: update C, d, R.
fn gpfa_m_step(
    y: &[f64],
    x_post: &[f64],
    xx_post: &[f64],
    n_neurons: usize,
    n_bins: usize,
    n_latents: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    // d_new = Y.mean(axis=1)
    let mut d_new = vec![0.0f64; n_neurons];
    for i in 0..n_neurons {
        let s: f64 = (0..n_bins).map(|t| y[i * n_bins + t]).sum();
        d_new[i] = s / n_bins as f64;
    }

    // Y_centered
    // Yx = Y_centered @ x_post^T (n_neurons x n_latents)
    let mut yx = vec![0.0f64; n_neurons * n_latents];
    for i in 0..n_neurons {
        for j in 0..n_latents {
            let mut s = 0.0;
            for t in 0..n_bins {
                s += (y[i * n_bins + t] - d_new[i]) * x_post[j * n_bins + t];
            }
            yx[i * n_latents + j] = s;
        }
    }

    // C_new = Yx @ inv(xx_post + eps*I)
    let mut xx_reg = xx_post.to_vec();
    for i in 0..n_latents {
        xx_reg[i * n_latents + i] += 1e-8;
    }
    let xx_inv = mat_inv(&xx_reg, n_latents);
    let mut c_new = vec![0.0f64; n_neurons * n_latents];
    for i in 0..n_neurons {
        for j in 0..n_latents {
            let mut s = 0.0;
            for k in 0..n_latents {
                s += yx[i * n_latents + k] * xx_inv[k * n_latents + j];
            }
            c_new[i * n_latents + j] = s;
        }
    }

    // R_new = diag(YY^T/T - C E[x]Y^T/T)
    let mut r_new = vec![0.0f64; n_neurons];
    for i in 0..n_neurons {
        let yyt: f64 = (0..n_bins)
            .map(|t| {
                let v = y[i * n_bins + t] - d_new[i];
                v * v
            })
            .sum::<f64>()
            / n_bins as f64;
        // C[i,:] @ x_post @ Y_centered[i,:]^T / T
        let mut cxy = 0.0;
        for j in 0..n_latents {
            for t in 0..n_bins {
                cxy += c_new[i * n_latents + j]
                    * x_post[j * n_bins + t]
                    * (y[i * n_bins + t] - d_new[i]);
            }
        }
        cxy /= n_bins as f64;
        r_new[i] = (yyt - cxy).max(1e-6);
    }

    (c_new, d_new, r_new)
}

/// Exact marginal Gaussian log-likelihood for the GPFA observation model.
///
/// Matches the Python reference: builds the dense `(n_obs x n_obs)` marginal
/// covariance `A K A^T + (I_T ⊗ R) + 1e-8 I` and returns
/// `-0.5 (yᵀ Σ⁻¹ y + log|Σ| + n_obs log 2π)`.
fn gpfa_log_likelihood(
    y: &[f64],
    c: &[f64],
    d: &[f64],
    r_diag: &[f64],
    k_all: &[Vec<f64>],
    n_neurons: usize,
    n_bins: usize,
    n_latents: usize,
) -> f64 {
    let n_obs = n_neurons * n_bins;
    let n_state = n_latents * n_bins;

    // A: (n_obs x n_state) with A[t*n_neurons + i, j*n_bins + t] = C[i, j].
    let mut a = vec![0.0f64; n_obs * n_state];
    for t in 0..n_bins {
        for j in 0..n_latents {
            let col = j * n_bins + t;
            for i in 0..n_neurons {
                a[(t * n_neurons + i) * n_state + col] = c[i * n_latents + j];
            }
        }
    }

    // K_big: block-diagonal (n_state x n_state).
    let mut k_big = vec![0.0f64; n_state * n_state];
    for (j, kernel) in k_all.iter().enumerate() {
        for ii in 0..n_bins {
            for jj in 0..n_bins {
                k_big[(j * n_bins + ii) * n_state + (j * n_bins + jj)] = kernel[ii * n_bins + jj];
            }
        }
    }

    // AK = A @ K_big.
    let mut ak = vec![0.0f64; n_obs * n_state];
    for i in 0..n_obs {
        for j in 0..n_state {
            let mut s = 0.0;
            for k in 0..n_state {
                s += a[i * n_state + k] * k_big[k * n_state + j];
            }
            ak[i * n_state + j] = s;
        }
    }

    // cov = AK @ Aᵀ + (I_T ⊗ R) + 1e-8 I.
    let mut cov = vec![0.0f64; n_obs * n_obs];
    for i in 0..n_obs {
        for j in 0..n_obs {
            let mut s = 0.0;
            for k in 0..n_state {
                s += ak[i * n_state + k] * a[j * n_state + k];
            }
            cov[i * n_obs + j] = s;
        }
    }
    for t in 0..n_bins {
        for i in 0..n_neurons {
            let idx = t * n_neurons + i;
            cov[idx * n_obs + idx] += r_diag[i];
        }
    }
    for i in 0..n_obs {
        cov[i * n_obs + i] += 1e-8;
    }

    // y_centered = (Y - d)ᵀ flattened time-major.
    let mut yc = vec![0.0f64; n_obs];
    for t in 0..n_bins {
        for i in 0..n_neurons {
            yc[t * n_neurons + i] = y[i * n_bins + t] - d[i];
        }
    }

    let sol = mat_solve(&cov, &yc, n_obs, 1);
    let quad: f64 = (0..n_obs).map(|i| yc[i] * sol[i]).sum();
    let (_sign, logdet) = mat_slogdet(&cov, n_obs);
    -0.5 * (quad + logdet + n_obs as f64 * (2.0 * std::f64::consts::PI).ln())
}

/// Run the GPFA EM loop from a fixed initialisation (the dispatchable kernel).
///
/// Returns `(trajectories, C, d, R_diag, log_likelihoods)`. The GP timescales
/// `tau` are held fixed, so the kernels are constant across iterations.
#[allow(clippy::too_many_arguments)]
pub fn gpfa_em_from_init(
    y: &[f64],
    c0: &[f64],
    d0: &[f64],
    r0_diag: &[f64],
    tau: &[f64],
    n_neurons: usize,
    n_bins: usize,
    n_latents: usize,
    max_iter: usize,
    tol: f64,
) -> GpfaEmOutput {
    let k_all: Vec<Vec<f64>> = (0..n_latents)
        .map(|j| gp_kernel(n_bins, tau[j], 1.0))
        .collect();
    let mut c = c0.to_vec();
    let mut d = d0.to_vec();
    let mut r = r0_diag.to_vec();
    let mut log_liks: Vec<f64> = Vec::new();
    let mut x_post = vec![0.0f64; n_latents * n_bins];

    for _ in 0..max_iter {
        let (xp, xx_post) = gpfa_e_step(y, &c, &d, &r, &k_all, n_neurons, n_bins, n_latents);
        x_post = xp;
        let (c_new, d_new, r_new) = gpfa_m_step(y, &x_post, &xx_post, n_neurons, n_bins, n_latents);
        c = c_new;
        d = d_new;
        r = r_new;
        let ll = gpfa_log_likelihood(y, &c, &d, &r, &k_all, n_neurons, n_bins, n_latents);
        log_liks.push(ll);
        if log_liks.len() > 1 {
            let prev = log_liks[log_liks.len() - 2];
            if (ll - prev).abs() < tol {
                break;
            }
        }
    }

    (x_post, c, d, r, log_liks)
}

// ── public API ──────────────────────────────────────────────────────

/// GPFA result.
pub struct GpfaResult {
    /// Latent trajectories, row-major `(n_latents, n_bins)`.
    pub trajectories: Vec<f64>,
    /// Loading matrix, row-major `(n_neurons, n_latents)`.
    pub c: Vec<f64>,
    /// Mean vector `(n_neurons)`.
    pub d: Vec<f64>,
    /// Noise diagonal `(n_neurons)`.
    pub r: Vec<f64>,
    /// GP timescales `(n_latents)`.
    pub tau: Vec<f64>,
    /// Log-likelihoods per iteration.
    pub log_likelihoods: Vec<f64>,
    pub n_latents: usize,
    pub n_bins: usize,
    pub n_neurons: usize,
}

/// Extract smooth latent trajectories from parallel spike trains via EM.
pub fn gpfa(
    trains: &[&[i32]],
    n_latents: usize,
    bin_ms: f64,
    dt: f64,
    max_iter: usize,
    tol: f64,
    seed: u64,
) -> GpfaResult {
    let n_neurons = trains.len();
    if n_neurons == 0 {
        return GpfaResult {
            trajectories: vec![],
            c: vec![],
            d: vec![],
            r: vec![],
            tau: vec![],
            log_likelihoods: vec![],
            n_latents: 0,
            n_bins: 0,
            n_neurons: 0,
        };
    }
    let bin_steps = (bin_ms / (dt * 1000.0)).round().max(1.0) as usize;
    let binned: Vec<Vec<f64>> = trains
        .iter()
        .map(|t| {
            basic::bin_spike_train(t, bin_steps)
                .into_iter()
                .map(|c| c as f64)
                .collect()
        })
        .collect();
    let n_bins = binned.iter().map(|b| b.len()).min().unwrap_or(0);
    if n_bins == 0 {
        return GpfaResult {
            trajectories: vec![],
            c: vec![],
            d: vec![],
            r: vec![],
            tau: vec![],
            log_likelihoods: vec![],
            n_latents: 0,
            n_bins: 0,
            n_neurons,
        };
    }
    // Y: n_neurons x n_bins
    let mut y = vec![0.0f64; n_neurons * n_bins];
    for i in 0..n_neurons {
        for j in 0..n_bins {
            y[i * n_bins + j] = binned[i][j];
        }
    }
    let nl = n_latents.min(n_neurons).min(n_bins);

    // Initialise
    let mut rng = seed;
    let mut c = vec![0.0f64; n_neurons * nl];
    for v in &mut c {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        *v = ((rng >> 33) as f64 / (1u64 << 31) as f64 - 0.5) * 0.2;
    }
    let mut d_vec = vec![0.0f64; n_neurons];
    for i in 0..n_neurons {
        d_vec[i] = y[i * n_bins..i * n_bins + n_bins].iter().sum::<f64>() / n_bins as f64;
    }
    let mut r_diag = vec![0.0f64; n_neurons];
    for i in 0..n_neurons {
        let mean = d_vec[i];
        let var: f64 = (0..n_bins)
            .map(|t| (y[i * n_bins + t] - mean).powi(2))
            .sum::<f64>()
            / n_bins as f64;
        r_diag[i] = var + 1e-4;
    }
    let tau = vec![bin_ms * 2.0; nl];

    let (x_post, c, d_vec, r_diag, log_liks) = gpfa_em_from_init(
        &y, &c, &d_vec, &r_diag, &tau, n_neurons, n_bins, nl, max_iter, tol,
    );

    GpfaResult {
        trajectories: x_post,
        c,
        d: d_vec,
        r: r_diag,
        tau,
        log_likelihoods: log_liks,
        n_latents: nl,
        n_bins,
        n_neurons,
    }
}

/// Project new spike trains using learned GPFA parameters.
pub fn gpfa_transform(
    new_trains: &[&[i32]],
    c: &[f64],
    d: &[f64],
    r_diag: &[f64],
    tau: &[f64],
    n_latents: usize,
    bin_ms: f64,
    dt: f64,
) -> Vec<f64> {
    let n_neurons = new_trains.len();
    if n_neurons == 0 || c.is_empty() {
        return vec![];
    }
    let bin_steps = (bin_ms / (dt * 1000.0)).round().max(1.0) as usize;
    let binned: Vec<Vec<f64>> = new_trains
        .iter()
        .map(|t| {
            basic::bin_spike_train(t, bin_steps)
                .into_iter()
                .map(|v| v as f64)
                .collect()
        })
        .collect();
    let n_bins = binned.iter().map(|b| b.len()).min().unwrap_or(0);
    if n_bins == 0 {
        return vec![];
    }
    let mut y = vec![0.0f64; n_neurons * n_bins];
    for i in 0..n_neurons {
        for j in 0..n_bins {
            y[i * n_bins + j] = binned[i][j];
        }
    }
    let k_all: Vec<Vec<f64>> = (0..n_latents)
        .map(|j| gp_kernel(n_bins, tau[j], 1.0))
        .collect();
    let (x_post, _) = gpfa_e_step(&y, c, d, r_diag, &k_all, n_neurons, n_bins, n_latents);
    x_post
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_trains() -> Vec<Vec<i32>> {
        let mut trains = Vec::new();
        for n in 0..4 {
            let mut t = vec![0i32; 100];
            let step = 3 + n * 2;
            for i in (0..100).step_by(step) {
                t[i] = 1;
            }
            trains.push(t);
        }
        trains
    }

    #[test]
    fn test_gpfa_basic() {
        let trains = make_trains();
        let refs: Vec<&[i32]> = trains.iter().map(|t| t.as_slice()).collect();
        let result = gpfa(&refs, 2, 10.0, 0.001, 5, 1e-4, 42);
        assert_eq!(result.n_neurons, 4);
        assert_eq!(result.n_latents, 2);
        assert!(!result.trajectories.is_empty());
        assert!(!result.log_likelihoods.is_empty());
    }

    #[test]
    fn test_gpfa_empty() {
        let result = gpfa(&[], 2, 10.0, 0.001, 5, 1e-4, 42);
        assert_eq!(result.n_neurons, 0);
        assert!(result.trajectories.is_empty());
    }

    #[test]
    fn test_gpfa_single_neuron() {
        let train = vec![1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0];
        let refs = vec![train.as_slice()];
        let result = gpfa(&refs, 1, 5.0, 0.001, 3, 1e-4, 42);
        assert_eq!(result.n_neurons, 1);
        assert_eq!(result.n_latents, 1);
    }

    #[test]
    fn test_gpfa_convergence() {
        let trains = make_trains();
        let refs: Vec<&[i32]> = trains.iter().map(|t| t.as_slice()).collect();
        let result = gpfa(&refs, 2, 10.0, 0.001, 20, 1e-4, 42);
        // Log-likelihood should generally increase
        if result.log_likelihoods.len() > 2 {
            let last = result.log_likelihoods[result.log_likelihoods.len() - 1];
            let second = result.log_likelihoods[1];
            assert!(
                last >= second - 1.0,
                "LL should generally increase: {second} -> {last}"
            );
        }
    }

    #[test]
    fn test_gpfa_transform() {
        let trains = make_trains();
        let refs: Vec<&[i32]> = trains.iter().map(|t| t.as_slice()).collect();
        let result = gpfa(&refs, 2, 10.0, 0.001, 5, 1e-4, 42);

        let new_trains = make_trains();
        let new_refs: Vec<&[i32]> = new_trains.iter().map(|t| t.as_slice()).collect();
        let projected = gpfa_transform(
            &new_refs,
            &result.c,
            &result.d,
            &result.r,
            &result.tau,
            result.n_latents,
            10.0,
            0.001,
        );
        assert!(!projected.is_empty());
        assert_eq!(projected.len(), result.n_latents * result.n_bins);
    }

    #[test]
    fn test_gpfa_transform_empty() {
        let proj = gpfa_transform(&[], &[], &[], &[], &[], 0, 10.0, 0.001);
        assert!(proj.is_empty());
    }

    #[test]
    fn test_gp_kernel_shape() {
        let k = gp_kernel(10, 5.0, 1.0);
        assert_eq!(k.len(), 100);
        // Diagonal should be sigma^2
        for i in 0..10 {
            assert!((k[i * 10 + i] - 1.0).abs() < 1e-10);
        }
        // Should be symmetric
        for i in 0..10 {
            for j in 0..10 {
                assert!((k[i * 10 + j] - k[j * 10 + i]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn test_gp_kernel_decay() {
        let k = gp_kernel(20, 3.0, 1.0);
        // Off-diagonal should decay with distance
        assert!(k[1] > k[10]);
    }
}
