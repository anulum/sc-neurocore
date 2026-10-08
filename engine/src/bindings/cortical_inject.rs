// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — cortical injection PyO3 bindings

//! Python bindings for per-row-parallel cortical-column CSR injection.

use numpy::{Element, PyArray1, PyArrayMethods, PyReadonlyArray1, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::{cortical_inject, runtime_control_binding::ensure_parallel_runtime};

/// Register single- and multi-block cortical injection kernels.
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(py_parallel_csr_spmv_add, module)?)?;
    module.add_function(wrap_pyfunction!(py_parallel_csr_multi_spmv_add, module)?)?;
    Ok(())
}

/// Borrow a typed input with an authored refusal instead of a NumPy borrow panic.
fn readonly<'py, T: Element>(
    array: &Bound<'py, PyArray1<T>>,
    name: &str,
) -> PyResult<PyReadonlyArray1<'py, T>> {
    array
        .try_readonly()
        .map_err(|_| PyValueError::new_err(format!("{name} cannot be borrowed for reading")))
}

/// Add one sparse cortical block multiplication to a mutable output vector.
///
/// `y += W @ x`, where `W` is the CSR matrix described by `indptr`, `indices`
/// and `data`. Per-row reductions preserve SciPy-compatible arithmetic order.
/// Validate every used CSR row and admit workers before changing `y`.
/// Empty output is a no-op; nonempty output must be writable and cannot alias inputs.
#[pyfunction]
#[pyo3(signature = (indptr, indices, data, x, y))]
fn py_parallel_csr_spmv_add(
    indptr: &Bound<'_, PyArray1<i32>>,
    indices: &Bound<'_, PyArray1<i32>>,
    data: &Bound<'_, PyArray1<f64>>,
    x: &Bound<'_, PyArray1<f64>>,
    y: &Bound<'_, PyArray1<f64>>,
) -> PyResult<()> {
    if y.shape()[0] == 0 {
        return Ok(());
    }
    let indptr = readonly(indptr, "indptr")?;
    let indices = readonly(indices, "indices")?;
    let data = readonly(data, "data")?;
    let x = readonly(x, "x")?;
    let mut y = y
        .try_readwrite()
        .map_err(|_| PyValueError::new_err("y must be writable and must not alias input arrays"))?;
    let indptr = indptr.as_slice()?;
    let indices = indices.as_slice()?;
    let data = data.as_slice()?;
    let x = x.as_slice()?;
    let output = y.as_slice_mut()?;
    cortical_inject::validate_csr_rows(indptr, indices, data, x, output.len())
        .map_err(PyValueError::new_err)?;
    if !output.is_empty() {
        ensure_parallel_runtime()?;
    }
    cortical_inject::parallel_csr_spmv_add(indptr, indices, data, x, output);
    Ok(())
}

/// Add several sparse cortical block multiplications in one FFI call.
/// All blocks must match, validate and admit workers before any output changes.
/// Empty output is a no-op; nonempty output must be writable and cannot alias inputs.
#[pyfunction]
#[pyo3(signature = (indptrs, indices_list, data_list, xs, y))]
fn py_parallel_csr_multi_spmv_add(
    indptrs: Vec<Bound<'_, PyArray1<i32>>>,
    indices_list: Vec<Bound<'_, PyArray1<i32>>>,
    data_list: Vec<Bound<'_, PyArray1<f64>>>,
    xs: Vec<Bound<'_, PyArray1<f64>>>,
    y: &Bound<'_, PyArray1<f64>>,
) -> PyResult<()> {
    if indptrs.len() != indices_list.len()
        || indptrs.len() != data_list.len()
        || indptrs.len() != xs.len()
    {
        return Err(PyValueError::new_err(
            "CSR block lists must have matching lengths",
        ));
    }
    if y.shape()[0] == 0 {
        return Ok(());
    }
    let indptrs = indptrs
        .iter()
        .map(|a| readonly(a, "indptrs"))
        .collect::<PyResult<Vec<_>>>()?;
    let indices_list = indices_list
        .iter()
        .map(|a| readonly(a, "indices_list"))
        .collect::<PyResult<Vec<_>>>()?;
    let data_list = data_list
        .iter()
        .map(|a| readonly(a, "data_list"))
        .collect::<PyResult<Vec<_>>>()?;
    let xs = xs
        .iter()
        .map(|a| readonly(a, "xs"))
        .collect::<PyResult<Vec<_>>>()?;
    let mut y = y
        .try_readwrite()
        .map_err(|_| PyValueError::new_err("y must be writable and must not alias input arrays"))?;
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
    let output = y.as_slice_mut()?;
    for block in 0..indptr_slices.len() {
        cortical_inject::validate_csr_rows(
            indptr_slices[block],
            indices_slices[block],
            data_slices[block],
            x_slices[block],
            output.len(),
        )
        .map_err(PyValueError::new_err)?;
    }
    if !output.is_empty() && !indptr_slices.is_empty() {
        ensure_parallel_runtime()?;
    }
    cortical_inject::parallel_csr_multi_spmv_add(
        &indptr_slices,
        &indices_slices,
        &data_slices,
        &x_slices,
        output,
    );
    Ok(())
}
