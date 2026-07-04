// SPDX-License-Identifier: AGPL-3.0-or-later
use sc_neurocore_engine::graph::{CsrMatrix, StochasticGraphLayer};

fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() < tol
}

#[test]
fn csr_dense_roundtrip() {
    // 3x3 identity
    let dense = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    let csr = CsrMatrix::from_dense(&dense, 3, 3, 1e-15);
    assert_eq!(csr.nnz(), 3);
    let back = csr.to_dense();
    for (a, b) in dense.iter().zip(back.iter()) {
        assert!(approx_eq(*a, *b, 1e-15));
    }
}

#[test]
fn csr_ring_graph() {
    // Ring: 0→1, 1→2, 2→0
    let row_offsets = vec![0, 1, 2, 3];
    let col_indices = vec![1, 2, 0];
    let values = vec![1.0, 1.0, 1.0];
    let csr = CsrMatrix::new(row_offsets, col_indices, values, 3, 3).unwrap();
    assert_eq!(csr.nnz(), 3);

    let dense = csr.to_dense();
    assert!(approx_eq(dense[1], 1.0, 1e-15)); // 0→1
    assert!(approx_eq(dense[5], 1.0, 1e-15)); // 1→2
    assert!(approx_eq(dense[6], 1.0, 1e-15)); // 2→0
    assert!(approx_eq(dense[0], 0.0, 1e-15)); // no self-loop
}

#[test]
fn sparse_forward_matches_dense() {
    let adj = vec![0.0, 0.9, 0.0, 0.9, 0.0, 0.9, 0.0, 0.9, 0.0];
    let n = 3;
    let nf = 2;
    let seed = 42;
    let features = vec![1.0, 0.5, 0.3, 0.7, 0.8, 0.2];

    let dense_layer = StochasticGraphLayer::new(adj.clone(), n, nf, seed);
    let csr = CsrMatrix::from_dense(&adj, n, n, 1e-15);
    let sparse_layer = StochasticGraphLayer::new_sparse(csr, nf, seed).unwrap();

    let out_dense = dense_layer.forward(&features).unwrap();
    let out_sparse = sparse_layer.forward(&features).unwrap();

    for (a, b) in out_dense.iter().zip(out_sparse.iter()) {
        assert!(approx_eq(*a, *b, 1e-12), "dense={} sparse={}", a, b);
    }
}

#[test]
fn auto_detect_sparse_threshold() {
    // 10x10 with only 5 non-zero entries → density = 5/100 = 5% < 30%
    let mut adj = vec![0.0_f64; 100];
    adj[1] = 1.0;
    adj[12] = 1.0;
    adj[23] = 1.0;
    adj[34] = 1.0;
    adj[40] = 1.0;

    let layer = StochasticGraphLayer::from_dense_auto(adj, 10, 3, 42, 0.3);
    assert!(layer.is_sparse());
}

#[test]
fn auto_detect_dense_threshold() {
    // 3x3 fully connected → density = 100% > 30%
    let adj = vec![1.0; 9];
    let layer = StochasticGraphLayer::from_dense_auto(adj, 3, 2, 42, 0.3);
    assert!(!layer.is_sparse());
}

#[test]
fn empty_graph_sparse() {
    let row_offsets = vec![0, 0, 0, 0];
    let csr = CsrMatrix::new(row_offsets, vec![], vec![], 3, 3).unwrap();
    let layer = StochasticGraphLayer::new_sparse(csr, 2, 42).unwrap();
    let features = vec![1.0, 0.5, 0.3, 0.7, 0.8, 0.2];
    let out = layer.forward(&features).unwrap();
    // Aggregation is zero (no edges), weight transform still applies, tanh(0) = 0
    for val in &out {
        assert!(approx_eq(*val, 0.0, 1e-15));
    }
}

#[test]
fn csr_rejects_mismatched_lengths() {
    let result = CsrMatrix::new(vec![0, 2], vec![0], vec![1.0, 2.0], 1, 3);
    assert!(result.is_err());
}

#[test]
fn sc_sparse_matches_dense() {
    let adj = vec![0.0, 0.8, 0.0, 0.8, 0.0, 0.8, 0.0, 0.8, 0.0];
    let n = 3;
    let nf = 2;
    let seed = 42;
    let features = vec![0.5, 0.5, 0.5, 0.5, 0.5, 0.5];

    let dense_layer = StochasticGraphLayer::new(adj.clone(), n, nf, seed);
    let csr = CsrMatrix::from_dense(&adj, n, n, 1e-15);
    let sparse_layer = StochasticGraphLayer::new_sparse(csr, nf, seed).unwrap();

    let out_dense = dense_layer.forward_sc(&features, 4096, 99).unwrap();
    let out_sparse = sparse_layer.forward_sc(&features, 4096, 99).unwrap();

    // Sparse encodes only nnz entries (different RNG draws), so tolerance is
    // governed by stochastic approximation error ~O(1/sqrt(length)).
    for (a, b) in out_dense.iter().zip(out_sparse.iter()) {
        assert!(approx_eq(*a, *b, 0.1), "dense={} sparse={}", a, b);
    }
}

#[test]
fn csr_new_rejects_out_of_range_column_index() {
    // A column index >= n_cols would index out of bounds in to_dense; new must reject it.
    let err = CsrMatrix::new(vec![0, 1], vec![5], vec![1.0], 1, 3).unwrap_err();
    assert!(err.contains("col_index"), "unexpected error: {err}");
}

#[test]
fn csr_new_rejects_non_monotonic_row_offsets() {
    // Lengths and nnz (= last offset = 0) are consistent, so the monotonicity check is
    // what rejects the 1 -> 0 drop.
    let err = CsrMatrix::new(vec![0, 1, 0], vec![], vec![], 2, 2).unwrap_err();
    assert!(err.contains("non-decreasing"), "unexpected error: {err}");
}

#[test]
fn csr_new_rejects_nonzero_first_offset() {
    // Lengths and nnz (= last = 1 = one column index) are consistent, so the first-offset
    // check is what rejects it.
    let err = CsrMatrix::new(vec![1, 1], vec![0], vec![1.0], 1, 2).unwrap_err();
    assert!(err.contains("start at 0"), "unexpected error: {err}");
}

#[test]
fn csr_new_accepts_valid_matrix() {
    // A well-formed CSR still constructs and expands without error.
    let csr = CsrMatrix::new(vec![0, 1, 2], vec![1, 0], vec![2.0, 3.0], 2, 2).unwrap();
    let dense = csr.to_dense();
    assert!(approx_eq(dense[1], 2.0, 1e-15) && approx_eq(dense[2], 3.0, 1e-15));
}
