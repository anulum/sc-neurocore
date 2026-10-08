// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Causality array shapes and VAR worker admission

//! Serial Granger and parallel VAR boundary contracts, preserving core mathematics.

use numpy::{IntoPyArray, PyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::arrays::{contiguous, matrix_size, readonly};
use crate::{analysis::causality, runtime_control_binding::ensure_parallel_runtime};

/// Register the five existing causality functions with their original signatures.
pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(py_pairwise_granger_causality, module)?)?;
    module.add_function(wrap_pyfunction!(py_conditional_granger_causality, module)?)?;
    module.add_function(wrap_pyfunction!(py_spectral_granger_causality, module)?)?;
    module.add_function(wrap_pyfunction!(py_partial_directed_coherence, module)?)?;
    module.add_function(wrap_pyfunction!(py_directed_transfer_function, module)?)?;
    Ok(())
}

fn bins(length: usize, bin_size: usize) -> PyResult<usize> {
    if bin_size == 0 {
        return Err(PyValueError::new_err("bin_size must be positive"));
    }
    Ok((length / bin_size).max(1))
}

fn serial_fit(lengths: &[usize], bin_size: usize, order: usize, factor: usize) -> PyResult<bool> {
    let n = bins(lengths.iter().copied().min().unwrap_or(0), bin_size)?;
    if order == 0 || order >= n.div_ceil(2) {
        return Ok(false);
    }
    let columns = matrix_size(order, factor)?;
    matrix_size(n - order, columns)?;
    matrix_size(columns, columns)?;
    Ok(true)
}

/// Return serial pairwise Granger causality for contiguous int32 trains.
/// Zero order or insufficient common bins return zero without workers.
/// Positive bin sizes and representable used OLS dimensions are required.
#[pyfunction]
#[pyo3(signature = (source, target, bin_size=10, order=5))]
fn py_pairwise_granger_causality(
    source: &Bound<'_, PyArray1<i32>>,
    target: &Bound<'_, PyArray1<i32>>,
    bin_size: usize,
    order: usize,
) -> PyResult<f64> {
    let source = readonly(source, "source")?;
    let target = readonly(target, "target")?;
    let s = contiguous(&source, "source")?;
    let t = contiguous(&target, "target")?;
    if !serial_fit(&[s.len(), t.len()], bin_size, order, 2)? {
        return Ok(0.0);
    }
    Ok(causality::pairwise_granger_causality(s, t, bin_size, order))
}

/// Return serial conditional Granger causality for contiguous int32 trains.
/// Common-bin truncation, zero-order and insufficient-data results remain zero.
/// Validate bin size and used OLS storage before the original serial computation.
#[pyfunction]
#[pyo3(signature = (source, target, condition, bin_size=10, order=5))]
fn py_conditional_granger_causality(
    source: &Bound<'_, PyArray1<i32>>,
    target: &Bound<'_, PyArray1<i32>>,
    condition: &Bound<'_, PyArray1<i32>>,
    bin_size: usize,
    order: usize,
) -> PyResult<f64> {
    let source = readonly(source, "source")?;
    let target = readonly(target, "target")?;
    let condition = readonly(condition, "condition")?;
    let s = contiguous(&source, "source")?;
    let t = contiguous(&target, "target")?;
    let c = contiguous(&condition, "condition")?;
    if !serial_fit(&[s.len(), t.len(), c.len()], bin_size, order, 3)? {
        return Ok(0.0);
    }
    Ok(causality::conditional_granger_causality(
        s, t, c, bin_size, order,
    ))
}

type SpectralKernel = fn(&[&[i32]], usize, usize, usize) -> (Vec<f64>, usize);

fn spectral(
    py: Python<'_>,
    trains: Vec<Bound<'_, PyArray1<i32>>>,
    bin_size: usize,
    order: usize,
    n_freqs: usize,
    kernel: SpectralKernel,
) -> PyResult<(Py<PyArray1<f64>>, usize, usize)> {
    let trains: Vec<_> = trains
        .iter()
        .map(|train| readonly(train, "trains"))
        .collect::<PyResult<_>>()?;
    let refs: Vec<_> = trains
        .iter()
        .map(|train| contiguous(train, "trains"))
        .collect::<PyResult<_>>()?;
    let d = refs.len();
    if d == 0 {
        return Ok((Vec::<f64>::new().into_pyarray(py).into(), 0, n_freqs));
    }
    let t = bins(refs[0].len(), bin_size)?;
    let square = matrix_size(d, d)?;
    matrix_size(square, n_freqs)?;
    matrix_size(square, 2)?;
    matrix_size(square, order)?;
    if t > order.saturating_add(1) {
        if order == 0 {
            return Err(PyValueError::new_err(
                "order must be positive for a fitted VAR model",
            ));
        }
        for train in &refs {
            if bins(train.len(), bin_size)? < t {
                return Err(PyValueError::new_err(
                    "each train must contain at least the first train's binned length",
                ));
            }
        }
        let columns = matrix_size(order, d)?;
        matrix_size(columns, columns)?;
        matrix_size(columns, t - order)?;
        matrix_size(d, t - order)?;
        ensure_parallel_runtime()?;
    }
    let (output, size) = kernel(&refs, bin_size, order, n_freqs);
    Ok((output.into_pyarray(py).into(), size, n_freqs))
}

/// Return flat spectral Granger values in neuron-by-neuron-by-frequency order.
/// Actual VAR fitting requires positive order and each train's used prefix.
/// Short first-train data retain the serial identity-covariance fallback.
#[pyfunction]
#[pyo3(signature = (trains, bin_size=10, order=5, n_freqs=64))]
fn py_spectral_granger_causality(
    py: Python<'_>,
    trains: Vec<Bound<'_, PyArray1<i32>>>,
    bin_size: usize,
    order: usize,
    n_freqs: usize,
) -> PyResult<(Py<PyArray1<f64>>, usize, usize)> {
    spectral(
        py,
        trains,
        bin_size,
        order,
        n_freqs,
        causality::spectral_granger_causality,
    )
}

/// Return flat partial directed coherence using the original VAR spectral matrix.
/// Admit workers only for the actual fit; zero frequencies still perform that fit.
/// Empty train lists return empty output with the supplied frequency metadata.
#[pyfunction]
#[pyo3(signature = (trains, bin_size=10, order=5, n_freqs=64))]
fn py_partial_directed_coherence(
    py: Python<'_>,
    trains: Vec<Bound<'_, PyArray1<i32>>>,
    bin_size: usize,
    order: usize,
    n_freqs: usize,
) -> PyResult<(Py<PyArray1<f64>>, usize, usize)> {
    spectral(
        py,
        trains,
        bin_size,
        order,
        n_freqs,
        causality::partial_directed_coherence,
    )
}

/// Return flat directed transfer values using the original complex LU inversion.
/// Validate used dimensions and train lengths before admitting the shared workers.
/// Readonly inputs, unused longer tails and no-fit fallback remain accepted.
#[pyfunction]
#[pyo3(signature = (trains, bin_size=10, order=5, n_freqs=64))]
fn py_directed_transfer_function(
    py: Python<'_>,
    trains: Vec<Bound<'_, PyArray1<i32>>>,
    bin_size: usize,
    order: usize,
    n_freqs: usize,
) -> PyResult<(Py<PyArray1<f64>>, usize, usize)> {
    spectral(
        py,
        trains,
        bin_size,
        order,
        n_freqs,
        causality::directed_transfer_function,
    )
}
