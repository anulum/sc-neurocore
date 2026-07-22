// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — cortical injection PyO3 bindings

//! Python bindings for per-row-parallel cortical-column CSR injection.

use numpy::{PyReadonlyArray1, PyReadwriteArray1};
use pyo3::prelude::*;

use crate::cortical_inject;

/// Register single- and multi-block cortical injection kernels.
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(py_parallel_csr_spmv_add, module)?)?;
    module.add_function(wrap_pyfunction!(py_parallel_csr_multi_spmv_add, module)?)?;
    Ok(())
}

/// Add one sparse cortical block multiplication to a mutable output vector.
///
/// `y += W @ x`, where `W` is the CSR matrix described by `indptr`, `indices`
/// and `data`. Per-row reductions preserve SciPy-compatible arithmetic order.
#[pyfunction]
#[pyo3(signature = (indptr, indices, data, x, y))]
fn py_parallel_csr_spmv_add(
    indptr: PyReadonlyArray1<'_, i32>,
    indices: PyReadonlyArray1<'_, i32>,
    data: PyReadonlyArray1<'_, f64>,
    x: PyReadonlyArray1<'_, f64>,
    y: PyReadwriteArray1<'_, f64>,
) -> PyResult<()> {
    let mut y = y;
    cortical_inject::parallel_csr_spmv_add(
        indptr.as_slice()?,
        indices.as_slice()?,
        data.as_slice()?,
        x.as_slice()?,
        y.as_slice_mut()?,
    );
    Ok(())
}

/// Add several sparse cortical block multiplications in one FFI call.
#[pyfunction]
#[pyo3(signature = (indptrs, indices_list, data_list, xs, y))]
fn py_parallel_csr_multi_spmv_add(
    indptrs: Vec<PyReadonlyArray1<'_, i32>>,
    indices_list: Vec<PyReadonlyArray1<'_, i32>>,
    data_list: Vec<PyReadonlyArray1<'_, f64>>,
    xs: Vec<PyReadonlyArray1<'_, f64>>,
    y: PyReadwriteArray1<'_, f64>,
) -> PyResult<()> {
    let mut y = y;
    let indptr_slices: Vec<&[i32]> = indptrs
        .iter()
        .map(|a| a.as_slice())
        .collect::<Result<_, _>>()?;
    let indices_slices: Vec<&[i32]> = indices_list
        .iter()
        .map(|a| a.as_slice())
        .collect::<Result<_, _>>()?;
    let data_slices: Vec<&[f64]> = data_list
        .iter()
        .map(|a| a.as_slice())
        .collect::<Result<_, _>>()?;
    let x_slices: Vec<&[f64]> = xs.iter().map(|a| a.as_slice()).collect::<Result<_, _>>()?;
    cortical_inject::parallel_csr_multi_spmv_add(
        &indptr_slices,
        &indices_slices,
        &data_slices,
        &x_slices,
        y.as_slice_mut()?,
    );
    Ok(())
}
