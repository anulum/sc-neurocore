// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Per-row-parallel CSR sparse mat-vec for the Potjans CorticalColumn block path

//! Per-row-parallel CSR sparse matrix-vector add for the Potjans
//! `CorticalColumn` block-CSR injection path.
//!
//! `parallel_csr_spmv_add(indptr, indices, data, x, y)` computes
//! `y += W @ x` where `W` is a CSR matrix described by `(indptr,
//! indices, data)`. Rows are processed in parallel via rayon.
//!
//! This is the kernel that lets `CorticalColumn` use the per-(source-
//! type, global-bin) block matrices at scale ≥ 0.5: a single block
//! mat-vec at scale=0.1 is ≈ 18 ms scipy-single-threaded; with rayon
//! over 8 cores it is ≈ 2-3 ms. At scale=0.5 the savings extrapolate
//! linearly with `nnz`, bringing 600 ms simulation wall-time from
//! ~50 minutes (single-threaded scipy block) into the
//! ~10-minute range and unlocking the full-scale (~77 000-cell)
//! convergence regime documented by van Albada et al. 2015 Fig 5.
//!
//! Determinism: per-row reductions are LOCAL to each row, so the
//! parallel order does not affect the result. Bit-identical to the
//! scipy single-threaded reference for matching inputs.

use rayon::prelude::*;

/// `y[r] += sum_k data[k] * x[indices[k]]` for `k in indptr[r]..indptr[r+1]`,
/// processing rows in **chunks** in parallel via rayon.
///
/// Per-row work is tiny (≈ 500 nnz × ≈ 1 ns/op = ~500 ns) — too
/// small for rayon's per-iteration scheduler overhead to amortise.
/// Chunking groups ~`CHUNK_SIZE` rows into one task so each task
/// runs ~250 µs of work, well above rayon's break-even point.
/// Measured 2026-04-18 on a 12-core box: per-row `par_iter_mut`
/// gave 0× speedup vs scipy single-threaded; `par_chunks_mut(512)`
/// gives ~3× speedup at scale=0.1 and scales further at larger N.
const CHUNK_SIZE: usize = 512;

pub fn parallel_csr_spmv_add(
    indptr: &[i32],
    indices: &[i32],
    data: &[f64],
    x: &[f64],
    y: &mut [f64],
) {
    y.par_chunks_mut(CHUNK_SIZE)
        .enumerate()
        .for_each(|(chunk_idx, chunk)| {
            let row_start = chunk_idx * CHUNK_SIZE;
            for (i, yi) in chunk.iter_mut().enumerate() {
                let r = row_start + i;
                let start = indptr[r] as usize;
                let end = indptr[r + 1] as usize;
                let mut sum: f64 = 0.0;
                for k in start..end {
                    let col = indices[k] as usize;
                    sum += data[k] * x[col];
                }
                *yi += sum;
            }
        });
}

/// Batched per-row-parallel CSR spmv add: `y += sum_b W_b @ x_b`
/// across `n_blocks` (matrix, vector) pairs, all sharing the same
/// row dimension. Used by `CorticalColumn._inject_block(dt)` to do
/// `2 × n_delay_bins` (= 10) spmv calls in one FFI call instead of
/// 10 separate FFI calls per step. The per-row reduction is local
/// so chunking still parallelises cleanly.
///
/// Layout: each block's slice into the flat `indptrs / indices /
/// data` arrays is given by `block_offsets`. Spike vectors are
/// flat-concatenated `xs` indexed by `x_offsets`.
#[allow(clippy::too_many_arguments)]
pub fn parallel_csr_multi_spmv_add(
    indptr_blocks: &[&[i32]],
    indices_blocks: &[&[i32]],
    data_blocks: &[&[f64]],
    x_blocks: &[&[f64]],
    y: &mut [f64],
) {
    let n_blocks = indptr_blocks.len();
    debug_assert_eq!(n_blocks, indices_blocks.len());
    debug_assert_eq!(n_blocks, data_blocks.len());
    debug_assert_eq!(n_blocks, x_blocks.len());

    y.par_chunks_mut(CHUNK_SIZE)
        .enumerate()
        .for_each(|(chunk_idx, chunk)| {
            let row_start = chunk_idx * CHUNK_SIZE;
            for (i, yi) in chunk.iter_mut().enumerate() {
                let r = row_start + i;
                let mut sum: f64 = 0.0;
                for b in 0..n_blocks {
                    let indptr = indptr_blocks[b];
                    let indices = indices_blocks[b];
                    let data = data_blocks[b];
                    let x = x_blocks[b];
                    let start = indptr[r] as usize;
                    let end = indptr[r + 1] as usize;
                    for k in start..end {
                        let col = indices[k] as usize;
                        sum += data[k] * x[col];
                    }
                }
                *yi += sum;
            }
        });
}

#[cfg(test)]
mod tests {
    use super::*;

    /// CSR matrix [[1, 0, 2], [0, 3, 0], [4, 0, 5]] @ [1, 1, 1] = [3, 3, 9].
    #[test]
    fn test_basic_csr_spmv() {
        let indptr: Vec<i32> = vec![0, 2, 3, 5];
        let indices: Vec<i32> = vec![0, 2, 1, 0, 2];
        let data: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let x: Vec<f64> = vec![1.0, 1.0, 1.0];
        let mut y: Vec<f64> = vec![0.0, 0.0, 0.0];
        parallel_csr_spmv_add(&indptr, &indices, &data, &x, &mut y);
        assert_eq!(y, vec![3.0, 3.0, 9.0]);
    }

    /// Empty rows (rows with zero nnz) leave `y` unchanged at that row.
    #[test]
    fn test_empty_row() {
        let indptr: Vec<i32> = vec![0, 1, 1, 2];
        let indices: Vec<i32> = vec![0, 1];
        let data: Vec<f64> = vec![10.0, 20.0];
        let x: Vec<f64> = vec![1.0, 2.0];
        let mut y: Vec<f64> = vec![100.0, 100.0, 100.0];
        parallel_csr_spmv_add(&indptr, &indices, &data, &x, &mut y);
        assert_eq!(y, vec![110.0, 100.0, 140.0]);
    }

    /// Result accumulates: calling twice doubles the contribution.
    #[test]
    fn test_accumulates_into_y() {
        let indptr: Vec<i32> = vec![0, 1, 2];
        let indices: Vec<i32> = vec![0, 0];
        let data: Vec<f64> = vec![3.0, 5.0];
        let x: Vec<f64> = vec![2.0];
        let mut y: Vec<f64> = vec![0.0, 0.0];
        parallel_csr_spmv_add(&indptr, &indices, &data, &x, &mut y);
        assert_eq!(y, vec![6.0, 10.0]);
        parallel_csr_spmv_add(&indptr, &indices, &data, &x, &mut y);
        assert_eq!(y, vec![12.0, 20.0]);
    }

    /// Larger matrix to exercise rayon's parallelism.
    #[test]
    fn test_large_dense_diagonal() {
        let n = 1024;
        let indptr: Vec<i32> = (0..=n).map(|i| i as i32).collect();
        let indices: Vec<i32> = (0..n).map(|i| i as i32).collect();
        let data: Vec<f64> = (0..n).map(|i| (i as f64) + 1.0).collect();
        let x: Vec<f64> = vec![1.0; n];
        let mut y: Vec<f64> = vec![0.0; n];
        parallel_csr_spmv_add(&indptr, &indices, &data, &x, &mut y);
        for i in 0..n {
            assert_eq!(y[i], (i as f64) + 1.0);
        }
    }

    /// Batched multi-spmv equals sequential single-spmv, accumulated.
    #[test]
    fn test_multi_spmv_matches_sequential() {
        // 3 blocks, each 4×3 with different sparsity patterns
        let indptr0: Vec<i32> = vec![0, 1, 2, 3, 4];
        let indices0: Vec<i32> = vec![0, 1, 2, 0];
        let data0: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
        let x0: Vec<f64> = vec![10.0, 20.0, 30.0];

        let indptr1: Vec<i32> = vec![0, 0, 1, 1, 2];
        let indices1: Vec<i32> = vec![1, 2];
        let data1: Vec<f64> = vec![5.0, 6.0];
        let x1: Vec<f64> = vec![100.0, 200.0, 300.0];

        let indptr2: Vec<i32> = vec![0, 1, 1, 2, 3];
        let indices2: Vec<i32> = vec![2, 0, 1];
        let data2: Vec<f64> = vec![7.0, 8.0, 9.0];
        let x2: Vec<f64> = vec![1000.0, 2000.0, 3000.0];

        // Sequential reference
        let mut y_seq = vec![0.0_f64; 4];
        parallel_csr_spmv_add(&indptr0, &indices0, &data0, &x0, &mut y_seq);
        parallel_csr_spmv_add(&indptr1, &indices1, &data1, &x1, &mut y_seq);
        parallel_csr_spmv_add(&indptr2, &indices2, &data2, &x2, &mut y_seq);

        // Batched
        let mut y_batched = vec![0.0_f64; 4];
        let indptrs: Vec<&[i32]> = vec![&indptr0, &indptr1, &indptr2];
        let indices_b: Vec<&[i32]> = vec![&indices0, &indices1, &indices2];
        let data_b: Vec<&[f64]> = vec![&data0, &data1, &data2];
        let xs: Vec<&[f64]> = vec![&x0, &x1, &x2];
        parallel_csr_multi_spmv_add(&indptrs, &indices_b, &data_b, &xs, &mut y_batched);

        assert_eq!(y_seq, y_batched);
    }
}
