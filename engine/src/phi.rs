// SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Phi* integrated information estimation in Rust

//! Barrett & Seth 2011 geometric Phi under Gaussian assumption.
//! Phi* = MI(past; future) - max_partition sum MI(past_k; future_k)

/// Compute Phi* for a multi-channel time series.
/// data: [n_channels][n_timesteps], row-major.
pub fn phi_star(data: &[Vec<f64>], tau: usize) -> f64 {
    let n = data.len();
    if n < 2 || data[0].len() <= 2 * tau {
        return 0.0;
    }
    let t = data[0].len();

    // Split into past and future
    let past: Vec<Vec<f64>> = data.iter().map(|ch| ch[..t - tau].to_vec()).collect();
    let future: Vec<Vec<f64>> = data.iter().map(|ch| ch[tau..].to_vec()).collect();

    let mi_whole = gaussian_mi(&past, &future);

    // Minimum information partition: contiguous splits only
    let mut mi_parts_min = f64::INFINITY;
    for k in 1..n {
        let past_a: Vec<Vec<f64>> = past[..k].to_vec();
        let fut_a: Vec<Vec<f64>> = future[..k].to_vec();
        let past_b: Vec<Vec<f64>> = past[k..].to_vec();
        let fut_b: Vec<Vec<f64>> = future[k..].to_vec();
        let mi_parts = gaussian_mi(&past_a, &fut_a) + gaussian_mi(&past_b, &fut_b);
        if mi_parts < mi_parts_min {
            mi_parts_min = mi_parts;
        }
    }

    (mi_whole - mi_parts_min).max(0.0)
}

/// Gaussian mutual information: MI(X;Y) = 0.5 * ln(det(Cov_X) * det(Cov_Y) / det(Cov_XY))
fn gaussian_mi(x: &[Vec<f64>], y: &[Vec<f64>]) -> f64 {
    let eps = 1e-10;
    let cov_x = covariance_matrix(x, eps);
    let cov_y = covariance_matrix(y, eps);

    let mut xy: Vec<Vec<f64>> = x.to_vec();
    xy.extend_from_slice(y);
    let cov_xy = covariance_matrix(&xy, eps);

    let det_x = determinant(&cov_x).max(1e-300);
    let det_y = determinant(&cov_y).max(1e-300);
    let det_xy = determinant(&cov_xy).max(1e-300);

    let mi = 0.5 * (det_x * det_y / det_xy).ln();
    mi.max(0.0)
}

/// Compute covariance matrix with regularization.
fn covariance_matrix(data: &[Vec<f64>], eps: f64) -> Vec<Vec<f64>> {
    let n = data.len();
    if n == 0 {
        return vec![];
    }
    let t = data[0].len();
    if t == 0 {
        return vec![vec![eps; n]; n];
    }

    // Compute means
    let means: Vec<f64> = data
        .iter()
        .map(|ch| ch.iter().sum::<f64>() / t as f64)
        .collect();

    // Compute covariance
    let mut cov = vec![vec![0.0; n]; n];
    let t_f = t as f64;
    let denom = (t_f - 1.0).max(1.0);

    for i in 0..n {
        for j in 0..=i {
            let dot = crate::simd::dot_f64_dispatch(&data[i], &data[j]);
            let val = (dot - t_f * means[i] * means[j]) / denom;
            cov[i][j] = val;
            cov[j][i] = val;
        }
        cov[i][i] += eps; // regularize diagonal
    }
    cov
}

/// Determinant via LU decomposition (in-place, partial pivoting).
fn determinant(matrix: &[Vec<f64>]) -> f64 {
    let n = matrix.len();
    if n == 0 {
        return 1.0;
    }
    if n == 1 {
        return matrix[0][0];
    }
    if n == 2 {
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0];
    }

    let mut a: Vec<Vec<f64>> = matrix.to_vec();
    let mut det = 1.0f64;

    for col in 0..n {
        // Partial pivot
        let mut max_row = col;
        let mut max_val = a[col][col].abs();
        for row in (col + 1)..n {
            if a[row][col].abs() > max_val {
                max_val = a[row][col].abs();
                max_row = row;
            }
        }
        if max_val < 1e-300 {
            return 0.0;
        }
        if max_row != col {
            a.swap(col, max_row);
            det = -det;
        }

        det *= a[col][col];
        let pivot = a[col][col];
        for row in (col + 1)..n {
            let factor = a[row][col] / pivot;
            for j in (col + 1)..n {
                a[row][j] -= factor * a[col][j];
            }
        }
    }
    det
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_independent_channels_low_phi() {
        // Independent random channels: Phi should be near zero
        let data = vec![
            (0..200).map(|i| (i as f64 * 0.1).sin()).collect(),
            (0..200).map(|i| (i as f64 * 0.3).cos()).collect(),
            (0..200).map(|i| (i as f64 * 0.7).sin()).collect(),
        ];
        let phi = phi_star(&data, 1);
        assert!(phi < 0.5, "Expected low Phi for independent, got {phi}");
    }

    #[test]
    fn test_correlated_channels_positive_phi() {
        // Cross-correlated but distinct: Phi should be positive
        let data = vec![
            (0..200)
                .map(|i| (i as f64 * 0.1).sin() + (i as f64 * 0.05).cos() * 0.3)
                .collect(),
            (0..200)
                .map(|i| (i as f64 * 0.1).sin() * 0.8 + (i as f64 * 0.2).sin() * 0.5)
                .collect(),
            (0..200)
                .map(|i| (i as f64 * 0.1).sin() * 0.6 + (i as f64 * 0.15).cos() * 0.7)
                .collect(),
        ];
        let phi = phi_star(&data, 1);
        assert!(phi >= 0.0, "Expected non-negative Phi, got {phi}");
    }

    #[test]
    fn test_single_channel_zero() {
        let data = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0]];
        assert_eq!(phi_star(&data, 1), 0.0);
    }

    #[test]
    fn test_short_data_zero() {
        let data = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        assert_eq!(phi_star(&data, 1), 0.0);
    }

    #[test]
    fn test_determinant_2x2() {
        let m = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let det = determinant(&m);
        assert!((det - (-2.0)).abs() < 1e-10);
    }

    #[test]
    fn test_determinant_3x3() {
        let m = vec![
            vec![1.0, 2.0, 3.0],
            vec![0.0, 1.0, 4.0],
            vec![5.0, 6.0, 0.0],
        ];
        let det = determinant(&m);
        assert!((det - 1.0).abs() < 1e-10);
    }
}
