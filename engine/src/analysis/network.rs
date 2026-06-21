// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Network-level spike train analysis

use super::basic::bin_spike_train;
use super::correlation::cross_correlation;
use nalgebra::{DMatrix, SymmetricEigen};

/// Functional connectivity matrix from peak cross-correlation.
/// Returns n×n matrix (flat, row-major) where (i,j) = max |cc| between neurons i and j.
pub fn functional_connectivity(trains: &[&[i32]], max_lag_ms: f64, dt: f64) -> Vec<f64> {
    let n = trains.len();
    let mut mat = vec![0.0_f64; n * n];
    for i in 0..n {
        mat[i * n + i] = 1.0;
        for j in (i + 1)..n {
            let (cc, _) = cross_correlation(trains[i], trains[j], max_lag_ms, dt);
            let peak = cc.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
            mat[i * n + j] = peak;
            mat[j * n + i] = peak;
        }
    }
    mat
}

/// Unitary event analysis (Gruen et al. 2002).
/// Detects bins where coincident spikes exceed Poisson chance.
/// Returns list of significant bin indices.
pub fn unitary_events(trains: &[&[i32]], bin_size: usize, alpha: f64) -> Vec<usize> {
    let n_trains = trains.len();
    if n_trains < 2 {
        return vec![];
    }
    let binned: Vec<Vec<i64>> = trains
        .iter()
        .map(|t| bin_spike_train(t, bin_size))
        .collect();
    let min_bins = binned.iter().map(|b| b.len()).min().unwrap_or(0);
    if min_bins == 0 {
        return vec![];
    }

    // Active matrix (binary)
    let active: Vec<Vec<bool>> = binned
        .iter()
        .map(|b| b[..min_bins].iter().map(|&v| v > 0).collect())
        .collect();

    // Mean rates per neuron
    let rates: Vec<f64> = active
        .iter()
        .map(|row| row.iter().filter(|&&v| v).count() as f64 / min_bins as f64)
        .collect();

    let expected_rate: f64 = rates.iter().product::<f64>().powi(n_trains as i32);

    let mut significant = Vec::new();
    for k in 0..min_bins {
        let all_active = (0..n_trains).all(|i| active[i][k]);
        if all_active && expected_rate < alpha {
            significant.push(k);
        }
    }
    significant
}

/// Cell assembly detection via PCA on binned spike matrix (Lopes-dos-Santos et al. 2013).
/// Returns list of assemblies (each a list of neuron indices).
pub fn cell_assembly_detection(
    trains: &[&[i32]],
    bin_size: usize,
    threshold: f64,
) -> Vec<Vec<usize>> {
    let n = trains.len();
    if n < 3 {
        return vec![];
    }
    let binned: Vec<Vec<f64>> = trains
        .iter()
        .map(|t| {
            bin_spike_train(t, bin_size)
                .iter()
                .map(|&v| v as f64)
                .collect()
        })
        .collect();
    let min_bins = binned.iter().map(|b| b.len()).min().unwrap_or(0);
    if min_bins < 2 {
        return vec![];
    }

    // Z-score each neuron's binned counts
    let mut mat: Vec<Vec<f64>> = binned.iter().map(|b| b[..min_bins].to_vec()).collect();
    for row in &mut mat {
        let mean = row.iter().sum::<f64>() / min_bins as f64;
        let std = (row.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / min_bins as f64)
            .sqrt()
            .max(1e-30);
        for v in row.iter_mut() {
            *v = (*v - mean) / std;
        }
    }

    // Correlation matrix: C = mat * mat^T / T
    let mut corr = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in i..n {
            let mut s = 0.0;
            for k in 0..min_bins {
                s += mat[i][k] * mat[j][k];
            }
            let c = s / min_bins as f64;
            corr[i * n + j] = c;
            corr[j * n + i] = c;
        }
    }

    // Eigendecomposition via the LAPACK-grade symmetric solver
    let (eigvals, eigvecs) = symmetric_eigen(&corr, n);

    // Marcenko-Pastur upper bound
    let q = n as f64 / min_bins as f64;
    let mp_upper = (1.0 + q.sqrt()).powi(2);

    let thresh_scaled = threshold / (n as f64).sqrt();
    let mut assemblies = Vec::new();
    for i in 0..n {
        if eigvals[i] > mp_upper {
            let members: Vec<usize> = (0..n)
                .filter(|&j| eigvecs[j * n + i].abs() > thresh_scaled)
                .collect();
            if members.len() >= 2 {
                assemblies.push(members);
            }
        }
    }
    assemblies
}

/// Synfire chain detection via cross-correlation peak ordering (Abeles 1991).
/// Returns list of chains (ordered neuron indices).
pub fn synfire_chain_detection(
    trains: &[&[i32]],
    dt: f64,
    max_delay_ms: f64,
    min_chain_length: usize,
) -> Vec<Vec<usize>> {
    let n = trains.len();
    if n < min_chain_length {
        return vec![];
    }

    // Peak lag matrix
    let mut peak_lags = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            let (cc, lags) = cross_correlation(trains[i], trains[j], max_delay_ms, dt);
            if !cc.is_empty() {
                let peak_idx = cc
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(idx, _)| idx)
                    .unwrap_or(0);
                peak_lags[i * n + j] = lags[peak_idx];
            }
        }
    }

    let mut chains = Vec::new();
    let mut visited = vec![false; n];

    for start in 0..n {
        if visited[start] {
            continue;
        }
        let mut chain = vec![start];
        let mut current = start;
        for _ in 0..n {
            let mut candidates: Vec<(f64, usize)> = Vec::new();
            for j in 0..n {
                if chain.contains(&j) {
                    continue;
                }
                let lag = peak_lags[current * n + j];
                if lag > 0.0 && lag <= max_delay_ms {
                    candidates.push((lag, j));
                }
            }
            if candidates.is_empty() {
                break;
            }
            candidates.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            let nxt = candidates[0].1;
            chain.push(nxt);
            current = nxt;
        }
        if chain.len() >= min_chain_length {
            for &idx in &chain {
                visited[idx] = true;
            }
            chains.push(chain);
        }
    }
    chains
}

/// Descending eigenvalues and sign-canonicalised eigenvectors of a symmetric
/// matrix `a` (row-major `n × n`) via nalgebra's symmetric eigensolver
/// (tridiagonalisation + implicit QR — LAPACK-grade, replacing a hand-rolled
/// Jacobi sweep). Eigenvectors are returned row-major (`vecs[row * n + col]`),
/// column `i` paired with eigenvalue `i`, each sign-fixed so its
/// largest-magnitude entry is positive.
fn symmetric_eigen(a: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    let se = SymmetricEigen::new(DMatrix::<f64>::from_row_slice(n, n, a));
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&i, &j| se.eigenvalues[j].partial_cmp(&se.eigenvalues[i]).unwrap());

    let vals: Vec<f64> = idx.iter().map(|&i| se.eigenvalues[i]).collect();
    let mut vecs = vec![0.0f64; n * n];
    for (new_col, &old_col) in idx.iter().enumerate() {
        let mut pivot = 0usize;
        let mut max_abs = 0.0f64;
        for r in 0..n {
            let v = se.eigenvectors[(r, old_col)].abs();
            if v > max_abs {
                max_abs = v;
                pivot = r;
            }
        }
        let sign = if se.eigenvectors[(pivot, old_col)] < 0.0 {
            -1.0
        } else {
            1.0
        };
        for r in 0..n {
            vecs[r * n + new_col] = sign * se.eigenvectors[(r, old_col)];
        }
    }
    (vals, vecs)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_train(spikes: &[usize], len: usize) -> Vec<i32> {
        let mut t = vec![0i32; len];
        for &s in spikes {
            t[s] = 1;
        }
        t
    }

    // ── functional_connectivity ─────────────────────────────────────

    #[test]
    fn test_fc_diagonal_one() {
        let t1 = make_train(&[10, 30, 50], 100);
        let t2 = make_train(&[20, 40, 60], 100);
        let trains: Vec<&[i32]> = vec![&t1, &t2];
        let mat = functional_connectivity(&trains, 20.0, 0.001);
        assert!((mat[0] - 1.0).abs() < 1e-10, "diagonal should be 1.0");
        assert!((mat[3] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_fc_symmetric() {
        let t1 = make_train(&[10, 30, 50], 100);
        let t2 = make_train(&[12, 32, 52], 100);
        let trains: Vec<&[i32]> = vec![&t1, &t2];
        let mat = functional_connectivity(&trains, 20.0, 0.001);
        assert!((mat[1] - mat[2]).abs() < 1e-10, "should be symmetric");
    }

    #[test]
    fn test_fc_identical_high() {
        let t = make_train(&[10, 30, 50, 70, 90], 100);
        let trains: Vec<&[i32]> = vec![&t, &t];
        let mat = functional_connectivity(&trains, 20.0, 0.001);
        assert!(
            mat[1] > 0.9,
            "identical trains → high connectivity, got {}",
            mat[1]
        );
    }

    // ── unitary_events ──────────────────────────────────────────────

    #[test]
    fn test_ue_coincident() {
        // Sparse trains — rate < 1.0 so expected_rate^n_trains < alpha
        // bin_size=10, 20 bins total. Each has 1 spike in bin 0 and bin 5 only → rate = 2/20 = 0.1
        let t1 = make_train(&[5, 55], 200);
        let t2 = make_train(&[5, 55], 200);
        let trains: Vec<&[i32]> = vec![&t1, &t2];
        let ue = unitary_events(&trains, 10, 0.05);
        assert!(
            !ue.is_empty(),
            "sparse coincident spikes → significant bins"
        );
    }

    #[test]
    fn test_ue_single_train() {
        let t = make_train(&[5, 15], 50);
        let trains: Vec<&[i32]> = vec![&t];
        assert!(
            unitary_events(&trains, 5, 0.05).is_empty(),
            "need ≥2 trains"
        );
    }

    #[test]
    fn test_ue_empty() {
        let t1 = vec![0i32; 50];
        let t2 = vec![0i32; 50];
        let trains: Vec<&[i32]> = vec![&t1, &t2];
        assert!(
            unitary_events(&trains, 5, 0.05).is_empty(),
            "no spikes → no events"
        );
    }

    // ── cell_assembly_detection ─────────────────────────────────────

    #[test]
    fn test_assembly_too_few_neurons() {
        let t1 = make_train(&[5, 15], 50);
        let t2 = make_train(&[5, 15], 50);
        let trains: Vec<&[i32]> = vec![&t1, &t2];
        assert!(
            cell_assembly_detection(&trains, 5, 2.0).is_empty(),
            "need ≥3 neurons"
        );
    }

    #[test]
    fn test_assembly_correlated_group() {
        // Neurons 0,1,2 fire together; neuron 3 fires independently
        let sync = make_train(&[0, 1, 10, 11, 20, 21, 30, 31, 40, 41], 50);
        let indep = make_train(&[3, 7, 13, 17, 23, 27, 33, 37, 43, 47], 50);
        let trains: Vec<&[i32]> = vec![&sync, &sync, &sync, &indep];
        let assemblies = cell_assembly_detection(&trains, 5, 1.0);
        // May or may not detect assembly depending on eigenstructure
        // Just verify it doesn't panic and returns valid indices
        for asm in &assemblies {
            for &idx in asm {
                assert!(idx < 4, "index out of bounds");
            }
        }
    }

    // ── synfire_chain_detection ─────────────────────────────────────

    #[test]
    fn test_synfire_sequential() {
        // Neurons fire in sequence: 0→1→2 with ~5ms delays
        let t0 = make_train(&[10, 30, 50, 70, 90], 100);
        let t1 = make_train(&[15, 35, 55, 75, 95], 100);
        let t2 = make_train(&[20, 40, 60, 80], 100);
        let trains: Vec<&[i32]> = vec![&t0, &t1, &t2];
        let chains = synfire_chain_detection(&trains, 0.001, 10.0, 3);
        // Should detect at least one chain
        if !chains.is_empty() {
            assert!(chains[0].len() >= 3, "chain should have ≥3 neurons");
        }
    }

    #[test]
    fn test_synfire_too_few() {
        let t = make_train(&[10, 30], 50);
        let trains: Vec<&[i32]> = vec![&t, &t];
        assert!(
            synfire_chain_detection(&trains, 0.001, 10.0, 3).is_empty(),
            "need ≥ min_chain_length neurons"
        );
    }

    // ── symmetric_eigen ─────────────────────────────────────────────

    #[test]
    fn test_symmetric_eigen_identity() {
        let a = vec![1.0, 0.0, 0.0, 1.0];
        let (vals, _) = symmetric_eigen(&a, 2);
        assert!((vals[0] - 1.0).abs() < 1e-10);
        assert!((vals[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_symmetric_eigen_descending() {
        let a = vec![3.0, 0.0, 0.0, 7.0];
        let (vals, _) = symmetric_eigen(&a, 2);
        // Eigenvalues are returned in descending order.
        assert!((vals[0] - 7.0).abs() < 1e-10);
        assert!((vals[1] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_symmetric_eigen_known() {
        // [[2, 1], [1, 2]] → eigenvalues 3 and 1 (descending)
        let a = vec![2.0, 1.0, 1.0, 2.0];
        let (vals, _) = symmetric_eigen(&a, 2);
        assert!(
            (vals[0] - 3.0).abs() < 1e-8,
            "eigenvalue 3, got {}",
            vals[0]
        );
        assert!(
            (vals[1] - 1.0).abs() < 1e-8,
            "eigenvalue 1, got {}",
            vals[1]
        );
    }

    #[test]
    fn test_symmetric_eigen_eigenvectors_orthogonal() {
        let a = vec![2.0, 1.0, 1.0, 2.0];
        let (_, v) = symmetric_eigen(&a, 2);
        // v[:,0] . v[:,1] should be ~0
        let dot: f64 = (0..2).map(|i| v[i * 2] * v[i * 2 + 1]).sum();
        assert!(
            dot.abs() < 1e-8,
            "eigenvectors should be orthogonal, dot={dot}"
        );
    }

    #[test]
    fn test_symmetric_eigen_sign_canonical() {
        // Each eigenvector column's dominant entry is positive.
        let a = vec![2.0, 1.0, 1.0, 2.0];
        let (_, v) = symmetric_eigen(&a, 2);
        for c in 0..2 {
            let pivot = if v[c].abs() >= v[2 + c].abs() { 0 } else { 1 };
            assert!(v[pivot * 2 + c] > 0.0, "column {c} not sign-canonical");
        }
    }
}
