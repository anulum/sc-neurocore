// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Covariance matrix array and worker contracts

//! Preserve covariance binning and empty shapes with checked output allocation.

use numpy::{PyArray1, PyArray2, PyArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::arrays::{contiguous, matrix_size, readonly};
use crate::runtime_control_binding::ensure_parallel_runtime;
use crate::{analysis::correlation, fixed_point_lif_binding::array_allocation::zeros_2d};

/// Register both existing covariance entry points.
pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(py_covariance_matrix, module)?)?;
    module.add_function(wrap_pyfunction!(py_spike_count_covariance, module)?)?;
    Ok(())
}

fn covariance(
    py: Python<'_>,
    trains: Vec<Bound<'_, PyArray1<i32>>>,
    bin_size: usize,
) -> PyResult<Py<PyArray2<f64>>> {
    let borrowed: Vec<_> = trains
        .iter()
        .map(|t| readonly(t, "trains"))
        .collect::<PyResult<_>>()?;
    let refs: Vec<_> = borrowed
        .iter()
        .map(|t| contiguous(t, "trains"))
        .collect::<PyResult<_>>()?;
    let n = refs.len();
    if n > 0 && bin_size == 0 {
        return Err(PyValueError::new_err("bin_size must be positive"));
    }
    matrix_size(n, n)?;
    let output = zeros_2d::<f64>(py, n.max(1), n)?;
    if n == 0 {
        return Ok(output.unbind());
    }
    if n > 1 {
        ensure_parallel_runtime()?;
    }
    let rows = correlation::covariance_matrix(&refs, bin_size);
    let mut writer = output.try_readwrite()?;
    let target = writer.as_slice_mut()?;
    for (destination, row) in target.chunks_exact_mut(n).zip(rows.iter()) {
        destination.copy_from_slice(row);
    }
    drop(writer);
    Ok(output.unbind())
}

/// Return covariance of contiguous int32 trains using the original shared bins.
/// No trains return shape `(1, 0)`; one train remains serial, including empty data.
/// Allocate the final NumPy matrix fallibly before admitting multiple-row workers.
#[pyfunction]
#[pyo3(signature = (trains, bin_size=10))]
fn py_covariance_matrix(
    py: Python<'_>,
    trains: Vec<Bound<'_, PyArray1<i32>>>,
    bin_size: usize,
) -> PyResult<Py<PyArray2<f64>>> {
    covariance(py, trains, bin_size)
}

/// Return windowed spike-count covariance with the original covariance kernel.
/// Preserve the same shapes, short-train sum bins and checked worker admission.
#[pyfunction]
#[pyo3(signature = (trains, window=50))]
fn py_spike_count_covariance(
    py: Python<'_>,
    trains: Vec<Bound<'_, PyArray1<i32>>>,
    window: usize,
) -> PyResult<Py<PyArray2<f64>>> {
    covariance(py, trains, window)
}
