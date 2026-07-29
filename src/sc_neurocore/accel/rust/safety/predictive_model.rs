// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety for world_model/predictive_model

use std::f64::consts::PI;

#[derive(Debug, Clone, PartialEq)]
pub struct KalmanFilterInput {
    pub observations: Vec<Vec<f64>>,
    pub controls: Vec<Vec<f64>>,
    pub a: Vec<Vec<f64>>,
    pub b: Vec<Vec<f64>>,
    pub c: Vec<Vec<f64>>,
    pub d: Vec<Vec<f64>>,
    pub q: Vec<Vec<f64>>,
    pub r: Vec<Vec<f64>>,
    pub mu_0: Vec<f64>,
    pub sigma_0: Vec<Vec<f64>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct KalmanFilterOutput {
    pub means: Vec<Vec<f64>>,
    pub covariances: Vec<Vec<Vec<f64>>>,
    pub pred_means: Vec<Vec<f64>>,
    pub pred_covs: Vec<Vec<Vec<f64>>>,
    pub log_lik: f64,
    pub backend: &'static str,
}

pub fn kalman_filter(input: &KalmanFilterInput) -> Result<KalmanFilterOutput, String> {
    validate_input(input)?;
    let t_steps = input.observations.len();
    let obs_dim = input.c.len();
    let state_dim = input.a.len();
    let control_dim = if input.b.is_empty() {
        0
    } else {
        input.b[0].len()
    };
    let has_control = control_dim > 0;

    let mut means = vec![vec![0.0; state_dim]; t_steps];
    let mut covariances = vec![vec![vec![0.0; state_dim]; state_dim]; t_steps];
    let mut pred_means = vec![vec![0.0; state_dim]; t_steps];
    let mut pred_covs = vec![vec![vec![0.0; state_dim]; state_dim]; t_steps];

    let mut x_pred = input.mu_0.clone();
    let mut p_pred = input.sigma_0.clone();
    let identity = eye(state_dim);
    let mut log_lik = 0.0;

    for t in 0..t_steps {
        pred_means[t] = x_pred.clone();
        pred_covs[t] = p_pred.clone();

        let y_hat_base = mat_vec(&input.c, &x_pred);
        let y_hat = if has_control {
            vec_add(&y_hat_base, &mat_vec(&input.d, &input.controls[t]))
        } else {
            y_hat_base
        };
        let innovation = vec_sub(&input.observations[t], &y_hat);
        let s_mat = mat_add(
            &mat_mul(&mat_mul(&input.c, &p_pred), &transpose(&input.c)),
            &input.r,
        );
        let s_sym = symmetrize(&s_mat);
        let s_chol = cholesky(&s_sym)?;
        let s_inv_innov = cholesky_solve(&s_chol, &innovation)?;
        let logdet_s = 2.0
            * s_chol
                .iter()
                .enumerate()
                .map(|(idx, row)| row[idx].ln())
                .sum::<f64>();
        let quad = dot(&innovation, &s_inv_innov);
        log_lik += -0.5 * (obs_dim as f64 * (2.0 * PI).ln() + logdet_s + quad);

        let p_ct = mat_mul(&p_pred, &transpose(&input.c));
        let k_gain = transpose(&cholesky_solve_matrix(&s_chol, &transpose(&p_ct))?);
        let x_filt = vec_add(&x_pred, &mat_vec(&k_gain, &innovation));
        let i_minus_kc = mat_sub(&identity, &mat_mul(&k_gain, &input.c));
        let p_filt = mat_add(
            &mat_mul(&mat_mul(&i_minus_kc, &p_pred), &transpose(&i_minus_kc)),
            &mat_mul(&mat_mul(&k_gain, &input.r), &transpose(&k_gain)),
        );

        means[t] = x_filt.clone();
        covariances[t] = symmetrize(&p_filt);

        let x_next_base = mat_vec(&input.a, &x_filt);
        x_pred = if has_control {
            vec_add(&x_next_base, &mat_vec(&input.b, &input.controls[t]))
        } else {
            x_next_base
        };
        p_pred = mat_add(
            &mat_mul(&mat_mul(&input.a, &p_filt), &transpose(&input.a)),
            &input.q,
        );
        p_pred = symmetrize(&p_pred);
    }

    Ok(KalmanFilterOutput {
        means,
        covariances,
        pred_means,
        pred_covs,
        log_lik,
        backend: "rust",
    })
}

pub fn validate_predictive_model(input: &KalmanFilterInput) -> bool {
    validate_input(input).is_ok()
}

fn validate_input(input: &KalmanFilterInput) -> Result<(), String> {
    let t_steps = input.observations.len();
    if t_steps == 0 {
        return Err("observations must contain at least one timestep".to_string());
    }
    let obs_dim = rectangular(&input.observations, "observations")?;
    let state_dim = square(&input.a, "A")?;
    let control_dim = if input.b.is_empty() {
        0
    } else {
        matrix_shape(&input.b, "B")?.1
    };

    expect_shape(&input.c, obs_dim, state_dim, "C")?;
    expect_shape(&input.q, state_dim, state_dim, "Q")?;
    expect_shape(&input.r, obs_dim, obs_dim, "R")?;
    expect_shape(&input.sigma_0, state_dim, state_dim, "Sigma_0")?;
    if input.mu_0.len() != state_dim {
        return Err("mu_0 length must match state dimension".to_string());
    }
    if control_dim > 0 {
        expect_shape(&input.b, state_dim, control_dim, "B")?;
        expect_shape(&input.d, obs_dim, control_dim, "D")?;
        expect_shape(&input.controls, t_steps, control_dim, "controls")?;
    } else {
        expect_shape(&input.b, state_dim, 0, "B")?;
        expect_shape(&input.d, obs_dim, 0, "D")?;
        if !input.controls.is_empty() {
            expect_shape(&input.controls, t_steps, 0, "controls")?;
        }
    }
    for (name, matrix) in [
        ("observations", &input.observations),
        ("controls", &input.controls),
        ("A", &input.a),
        ("B", &input.b),
        ("C", &input.c),
        ("D", &input.d),
        ("Q", &input.q),
        ("R", &input.r),
        ("Sigma_0", &input.sigma_0),
    ] {
        for row in matrix {
            for value in row {
                if !value.is_finite() {
                    return Err(format!("{name} contains non-finite value"));
                }
            }
        }
    }
    if input.mu_0.iter().any(|value| !value.is_finite()) {
        return Err("mu_0 contains non-finite value".to_string());
    }
    cholesky(&input.r).map_err(|_| "R must be positive definite".to_string())?;
    cholesky(&input.sigma_0).map_err(|_| "Sigma_0 must be positive definite".to_string())?;
    Ok(())
}

fn matrix_shape(matrix: &[Vec<f64>], name: &str) -> Result<(usize, usize), String> {
    if matrix.is_empty() {
        return Ok((0, 0));
    }
    let cols = matrix[0].len();
    if matrix.iter().any(|row| row.len() != cols) {
        return Err(format!("{name} must be rectangular"));
    }
    Ok((matrix.len(), cols))
}

fn rectangular(matrix: &[Vec<f64>], name: &str) -> Result<usize, String> {
    let (_, cols) = matrix_shape(matrix, name)?;
    if cols == 0 {
        Err(format!("{name} must not have zero columns"))
    } else {
        Ok(cols)
    }
}

fn square(matrix: &[Vec<f64>], name: &str) -> Result<usize, String> {
    let (rows, cols) = matrix_shape(matrix, name)?;
    if rows == 0 || rows != cols {
        Err(format!("{name} must be square"))
    } else {
        Ok(rows)
    }
}

fn expect_shape(matrix: &[Vec<f64>], rows: usize, cols: usize, name: &str) -> Result<(), String> {
    let shape = matrix_shape(matrix, name)?;
    if shape == (rows, cols) {
        Ok(())
    } else {
        Err(format!(
            "{name} shape mismatch: expected ({rows}, {cols}), got {shape:?}"
        ))
    }
}

fn eye(n: usize) -> Vec<Vec<f64>> {
    let mut out = vec![vec![0.0; n]; n];
    for (idx, row) in out.iter_mut().enumerate() {
        row[idx] = 1.0;
    }
    out
}

fn transpose(matrix: &[Vec<f64>]) -> Vec<Vec<f64>> {
    if matrix.is_empty() {
        return Vec::new();
    }
    let mut out = vec![vec![0.0; matrix.len()]; matrix[0].len()];
    for row in 0..matrix.len() {
        for col in 0..matrix[0].len() {
            out[col][row] = matrix[row][col];
        }
    }
    out
}

fn mat_mul(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }
    let mut out = vec![vec![0.0; b[0].len()]; a.len()];
    for i in 0..a.len() {
        for k in 0..b.len() {
            for j in 0..b[0].len() {
                out[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    out
}

fn mat_vec(a: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
    a.iter().map(|row| dot(row, x)).collect()
}

fn mat_add(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    a.iter()
        .zip(b)
        .map(|(left, right)| vec_add(left, right))
        .collect()
}

fn mat_sub(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    a.iter()
        .zip(b)
        .map(|(left, right)| vec_sub(left, right))
        .collect()
}

fn vec_add(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b).map(|(left, right)| left + right).collect()
}

fn vec_sub(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b).map(|(left, right)| left - right).collect()
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(left, right)| left * right).sum()
}

fn symmetrize(matrix: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let mut out = matrix.to_vec();
    for i in 0..out.len() {
        for j in 0..out.len() {
            out[i][j] = 0.5 * (matrix[i][j] + matrix[j][i]);
        }
    }
    out
}

fn cholesky(matrix: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, String> {
    let n = square(matrix, "matrix")?;
    let mut l = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..=i {
            let mut sum = matrix[i][j];
            for (left, right) in l[i][..j].iter().zip(&l[j][..j]) {
                sum -= left * right;
            }
            if i == j {
                if sum <= 0.0 || !sum.is_finite() {
                    return Err("matrix must be positive definite".to_string());
                }
                l[i][j] = sum.sqrt();
            } else {
                l[i][j] = sum / l[j][j];
            }
        }
    }
    Ok(l)
}

fn cholesky_solve(l: &[Vec<f64>], rhs: &[f64]) -> Result<Vec<f64>, String> {
    let n = l.len();
    if rhs.len() != n {
        return Err("rhs dimension mismatch".to_string());
    }
    let mut y = vec![0.0; n];
    for i in 0..n {
        let mut sum = rhs[i];
        for k in 0..i {
            sum -= l[i][k] * y[k];
        }
        y[i] = sum / l[i][i];
    }
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut sum = y[i];
        for k in (i + 1)..n {
            sum -= l[k][i] * x[k];
        }
        x[i] = sum / l[i][i];
    }
    Ok(x)
}

fn cholesky_solve_matrix(l: &[Vec<f64>], rhs: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, String> {
    let rhs_t = transpose(rhs);
    let solved_cols = rhs_t
        .iter()
        .map(|col| cholesky_solve(l, col))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(transpose(&solved_cols))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scalar_input() -> KalmanFilterInput {
        KalmanFilterInput {
            observations: vec![vec![1.0], vec![0.8], vec![1.2]],
            controls: Vec::new(),
            a: vec![vec![1.0]],
            b: vec![Vec::new()],
            c: vec![vec![1.0]],
            d: vec![Vec::new()],
            q: vec![vec![0.01]],
            r: vec![vec![0.04]],
            mu_0: vec![0.0],
            sigma_0: vec![vec![1.0]],
        }
    }

    #[test]
    fn test_predictive_model_validates_shapes_and_pd_covariances() {
        let input = scalar_input();
        assert!(validate_predictive_model(&input));
        let mut bad = input.clone();
        bad.r = vec![vec![0.0]];
        assert!(kalman_filter(&bad).is_err());
        let mut bad_shape = input;
        bad_shape.observations[0] = vec![1.0, 2.0];
        assert!(kalman_filter(&bad_shape).is_err());
    }

    #[test]
    fn test_scalar_kalman_filter_tracks_observations_and_log_likelihood() {
        let output = kalman_filter(&scalar_input()).unwrap();
        assert_eq!(output.backend, "rust");
        assert_eq!(output.means.len(), 3);
        assert_eq!(output.covariances.len(), 3);
        assert!(output.log_lik.is_finite());
        assert!(output.means[0][0] > 0.9);
        assert!(output.means[2][0] > output.means[1][0]);
        assert!(output.covariances[2][0][0] > 0.0);
    }

    #[test]
    fn test_controlled_kalman_filter_uses_b_and_d_terms() {
        let input = KalmanFilterInput {
            observations: vec![vec![1.0], vec![1.5]],
            controls: vec![vec![1.0], vec![1.0]],
            a: vec![vec![1.0]],
            b: vec![vec![0.5]],
            c: vec![vec![1.0]],
            d: vec![vec![0.1]],
            q: vec![vec![0.01]],
            r: vec![vec![0.05]],
            mu_0: vec![0.0],
            sigma_0: vec![vec![0.5]],
        };
        let output = kalman_filter(&input).unwrap();
        assert!(output.pred_means[1][0] > output.means[0][0]);
        assert!(output.means[1][0] > output.means[0][0]);
    }
}
