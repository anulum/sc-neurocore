// SPDX-License-Identifier: AGPL-3.0-or-later
//! Fuzz sparse-matrix (CSR) construction and expansion.
//!
//! `CsrMatrix::new` validates a caller-supplied CSR triple (it is reachable from Python
//! via the sparse graph constructor), so a CSR it accepts must be safe to use: expanding
//! it with `to_dense` must not panic on a non-monotonic row offset or an out-of-range
//! column index. This target builds a CSR from `arbitrary` vectors and, if `new` accepts
//! it, expands it — any index-out-of-bounds, slice or overflow panic aborts under
//! libFuzzer. `n_rows`/`n_cols` are `u8` so the dense expansion and `n_rows + 1` stay
//! bounded (the CSR-validation invariants are exercised at small dimensions). Build without
//! Z3: `cargo +nightly fuzz build csr_matrix`.
#![no_main]

use libfuzzer_sys::fuzz_target;
use sc_neurocore_engine::graph::CsrMatrix;

fuzz_target!(|input: (Vec<usize>, Vec<usize>, Vec<f64>, u8, u8)| {
    let (row_offsets, col_indices, values, n_rows, n_cols) = input;
    if let Ok(csr) = CsrMatrix::new(
        row_offsets,
        col_indices,
        values,
        n_rows as usize,
        n_cols as usize,
    ) {
        // A CSR that `new` validated must expand without panicking.
        let _ = csr.to_dense();
    }
});
