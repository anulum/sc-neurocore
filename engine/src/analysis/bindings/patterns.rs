// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Spike pattern array and worker contracts

//! Checked directionality and matrix outputs without zero-width Rayon chunks.

use numpy::{PyArray1, PyArray2, PyArrayMethods};
use pyo3::prelude::*;

use super::arrays::{contiguous, matrix_size, readonly};
use crate::runtime_control_binding::ensure_parallel_runtime;
use crate::{analysis::patterns, fixed_point_lif_binding::array_allocation::zeros_2d};

/// Register the existing directionality, order and cumulant functions.
pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(py_spike_directionality, module)?)?;
    module.add_function(wrap_pyfunction!(py_spike_train_order, module)?)?;
    module.add_function(wrap_pyfunction!(py_cubic_higher_order, module)?)?;
    Ok(())
}

/// Return the serial original nearest-before/after directionality statistic.
/// Contiguous float64 timestamps are filtered inclusively by the given bounds.
/// Readonly input and unsorted timestamps retain the original acceptance.
#[pyfunction]
#[pyo3(signature = (times_a, times_b, t_start=0.0, t_end=1.0))]
fn py_spike_directionality(
    times_a: &Bound<'_, PyArray1<f64>>,
    times_b: &Bound<'_, PyArray1<f64>>,
    t_start: f64,
    t_end: f64,
) -> PyResult<f64> {
    let a = readonly(times_a, "times_a")?;
    let b = readonly(times_b, "times_b")?;
    Ok(patterns::spike_directionality(
        contiguous(&a, "times_a")?,
        contiguous(&b, "times_b")?,
        t_start,
        t_end,
    ))
}

/// Return the original antisymmetric train-order matrix from float64 timestamps.
/// Empty and single-train matrices stay lazy; multiple rows use shared workers.
/// Allocate final output before workers and preserve original upper-pair ordering.
#[pyfunction]
#[pyo3(signature = (times_list, t_start=0.0, t_end=1.0))]
fn py_spike_train_order(
    py: Python<'_>,
    times_list: Vec<Bound<'_, PyArray1<f64>>>,
    t_start: f64,
    t_end: f64,
) -> PyResult<Py<PyArray2<f64>>> {
    let borrowed: Vec<_> = times_list
        .iter()
        .map(|t| readonly(t, "times_list"))
        .collect::<PyResult<_>>()?;
    let refs: Vec<_> = borrowed
        .iter()
        .map(|t| contiguous(t, "times_list"))
        .collect::<PyResult<_>>()?;
    let n = refs.len();
    matrix_size(n, n)?;
    let output = zeros_2d::<f64>(py, n, n)?;
    if n <= 1 {
        return Ok(output.unbind());
    }
    ensure_parallel_runtime()?;
    let values = patterns::spike_train_order(&refs, t_start, t_end);
    output
        .try_readwrite()?
        .as_slice_mut()?
        .copy_from_slice(&values);
    Ok(output.unbind())
}

/// Return the original third-order cumulant matrix for contiguous int32 samples.
/// Zero lag width and empty samples return zero matrices without workers.
/// The existing kernel ignores dt and uses the full sample mean and valid lags.
#[pyfunction]
#[pyo3(signature = (binary_train, dt=0.001, max_lag=20))]
fn py_cubic_higher_order(
    py: Python<'_>,
    binary_train: &Bound<'_, PyArray1<i32>>,
    dt: f64,
    max_lag: usize,
) -> PyResult<Py<PyArray2<f64>>> {
    let borrowed = readonly(binary_train, "binary_train")?;
    let data = contiguous(&borrowed, "binary_train")?;
    matrix_size(max_lag, max_lag)?;
    let output = zeros_2d::<f64>(py, max_lag, max_lag)?;
    if max_lag == 0 || data.is_empty() {
        return Ok(output.unbind());
    }
    ensure_parallel_runtime()?;
    let values = patterns::cubic_higher_order(data, dt, max_lag);
    output
        .try_readwrite()?
        .as_slice_mut()?
        .copy_from_slice(&values);
    Ok(output.unbind())
}
