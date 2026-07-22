// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — hierarchical partition PyO3 binding

//! Python binding for correlation-aware Kernighan-Lin partition refinement.

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use crate::partition;

/// Register hierarchical partition refinement with the extension module.
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(py_kl_refine, module)?)?;
    Ok(())
}

// Caller passes flat numpy arrays (CSR adjacency + flat scc weights +
// flat vertex_weights + initial part_map). The kernel mutates a copy
// of part_map in-place and returns (new_part_map, num_moves).
#[pyfunction]
#[pyo3(signature = (
    adj_offsets, adj_neighbours, adj_scc_abs, vertex_weights,
    part_map, parts_concat, parts_offsets,
    n_parts, kl_iterations, correlation_penalty,
))]
#[allow(clippy::too_many_arguments)]
fn py_kl_refine<'py>(
    py: Python<'py>,
    adj_offsets: PyReadonlyArray1<'_, i64>,
    adj_neighbours: PyReadonlyArray1<'_, i32>,
    adj_scc_abs: PyReadonlyArray1<'_, f64>,
    vertex_weights: PyReadonlyArray1<'_, f64>,
    part_map: PyReadonlyArray1<'_, i32>,
    parts_concat: PyReadonlyArray1<'_, i32>,
    parts_offsets: PyReadonlyArray1<'_, i64>,
    n_parts: i32,
    kl_iterations: i32,
    correlation_penalty: f64,
) -> PyResult<(Py<PyArray1<i32>>, u64)> {
    let mut pm = part_map.as_slice()?.to_vec();
    let moves = partition::kl_refine(
        adj_offsets.as_slice()?,
        adj_neighbours.as_slice()?,
        adj_scc_abs.as_slice()?,
        vertex_weights.as_slice()?,
        &mut pm,
        parts_concat.as_slice()?,
        parts_offsets.as_slice()?,
        n_parts,
        kl_iterations,
        correlation_penalty,
    );
    Ok((pm.into_pyarray(py).into(), moves))
}
