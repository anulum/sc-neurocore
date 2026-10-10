// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Neural decoder array and runtime binding contracts

//! Typed, contiguous neural-decoder inputs and recoverable worker admission.

use numpy::{IntoPyArray, PyArray1};
use pyo3::prelude::*;

use super::arrays::{contiguous, matrix_size, readonly, require_prefix};
use crate::{analysis::neural_decoders, runtime_control_binding::ensure_parallel_runtime};

type SpikeTokenArrays = (Py<PyArray1<i64>>, Py<PyArray1<f64>>);

/// Register the existing five neural-decoder functions without changing their names.
pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(py_tokenise_spikes, module)?)?;
    module.add_function(wrap_pyfunction!(py_sinusoidal_position_encode, module)?)?;
    module.add_function(wrap_pyfunction!(py_scaled_dot_product_attention, module)?)?;
    module.add_function(wrap_pyfunction!(py_gaussian_attention, module)?)?;
    module.add_function(wrap_pyfunction!(py_infonce_loss, module)?)?;
    Ok(())
}

/// Return stable time-sorted unit IDs and millisecond timestamps for nonzero spikes.
/// Inputs are one-dimensional contiguous int32 arrays; readonly and aliased inputs
/// are accepted. Empty trains return empty arrays without starting workers.
#[pyfunction]
#[pyo3(signature = (trains, dt=1.0))]
fn py_tokenise_spikes(
    py: Python<'_>,
    trains: Vec<Bound<'_, PyArray1<i32>>>,
    dt: f64,
) -> PyResult<SpikeTokenArrays> {
    let borrowed: Vec<_> = trains
        .iter()
        .map(|train| readonly(train, "trains"))
        .collect::<PyResult<_>>()?;
    let refs: Vec<_> = borrowed
        .iter()
        .map(|train| contiguous(train, "trains"))
        .collect::<PyResult<_>>()?;
    if refs.iter().all(|train| train.is_empty()) {
        return Ok((
            Vec::<i64>::new().into_pyarray(py).into(),
            Vec::<f64>::new().into_pyarray(py).into(),
        ));
    }
    ensure_parallel_runtime()?;
    let tokens = neural_decoders::tokenise_spikes(&refs, dt);
    let uids: Vec<i64> = tokens.iter().map(|token| token.0 as i64).collect();
    let times: Vec<f64> = tokens.iter().map(|token| token.1).collect();
    Ok((uids.into_pyarray(py).into(), times.into_pyarray(py).into()))
}

/// Encode contiguous float64 timestamps as flat row-major sinusoidal positions.
/// The output has `len(timestamps) * d_model` elements, including odd widths.
/// Zero-width or empty encodings return empty output without starting workers.
#[pyfunction]
fn py_sinusoidal_position_encode(
    py: Python<'_>,
    timestamps: &Bound<'_, PyArray1<f64>>,
    d_model: usize,
) -> PyResult<Py<PyArray1<f64>>> {
    let timestamps = readonly(timestamps, "timestamps")?;
    let ts = contiguous(&timestamps, "timestamps")?;
    if matrix_size(ts.len(), d_model)? == 0 {
        return Ok(Vec::<f64>::new().into_pyarray(py).into());
    }
    ensure_parallel_runtime()?;
    Ok(neural_decoders::sinusoidal_position_encode(ts, d_model)
        .into_pyarray(py)
        .into())
}

/// Compute scaled dot-product attention on flat float64 Q, K and V buffers.
/// Used prefixes have shapes `(nq, d)`, `(nk, d)` and `(nk, d)`; tails are ignored.
/// Reject short or noncontiguous inputs before worker admission or computation.
#[pyfunction]
fn py_scaled_dot_product_attention(
    py: Python<'_>,
    queries: &Bound<'_, PyArray1<f64>>,
    keys: &Bound<'_, PyArray1<f64>>,
    values: &Bound<'_, PyArray1<f64>>,
    nq: usize,
    nk: usize,
    d: usize,
) -> PyResult<Py<PyArray1<f64>>> {
    let queries = readonly(queries, "queries")?;
    let keys = readonly(keys, "keys")?;
    let values = readonly(values, "values")?;
    let q = contiguous(&queries, "queries")?;
    let k = contiguous(&keys, "keys")?;
    let v = contiguous(&values, "values")?;
    let output_size = matrix_size(nq, d)?;
    if output_size == 0 {
        return Ok(Vec::<f64>::new().into_pyarray(py).into());
    }
    require_prefix(q, output_size, "queries")?;
    let key_size = matrix_size(nk, d)?;
    require_prefix(k, key_size, "keys")?;
    require_prefix(v, key_size, "values")?;
    ensure_parallel_runtime()?;
    Ok(
        neural_decoders::scaled_dot_product_attention(q, k, v, nq, nk, d)
            .into_pyarray(py)
            .into(),
    )
}

/// Compute Gaussian attention using flat float64 buffers and the supplied sigma.
/// Used prefixes have shapes `(nq, d)`, `(nk, d)` and `(nk, d)`; tails are ignored.
/// Empty keys retain zero output; zero output avoids worker admission.
#[pyfunction]
fn py_gaussian_attention(
    py: Python<'_>,
    queries: &Bound<'_, PyArray1<f64>>,
    keys: &Bound<'_, PyArray1<f64>>,
    values: &Bound<'_, PyArray1<f64>>,
    nq: usize,
    nk: usize,
    d: usize,
    sigma: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let queries = readonly(queries, "queries")?;
    let keys = readonly(keys, "keys")?;
    let values = readonly(values, "values")?;
    let q = contiguous(&queries, "queries")?;
    let k = contiguous(&keys, "keys")?;
    let v = contiguous(&values, "values")?;
    let output_size = matrix_size(nq, d)?;
    if output_size == 0 {
        return Ok(Vec::<f64>::new().into_pyarray(py).into());
    }
    require_prefix(q, output_size, "queries")?;
    let key_size = matrix_size(nk, d)?;
    require_prefix(k, key_size, "keys")?;
    require_prefix(v, key_size, "values")?;
    ensure_parallel_runtime()?;
    Ok(
        neural_decoders::gaussian_attention(q, k, v, nq, nk, d, sigma)
            .into_pyarray(py)
            .into(),
    )
}

/// Return cosine InfoNCE loss for the first `n * d` float64 embedding elements.
/// Both arrays must be contiguous and contain the full used prefix.
/// Zero rows or width retain the original zero loss without starting workers.
#[pyfunction]
fn py_infonce_loss(
    anchors: &Bound<'_, PyArray1<f64>>,
    positives: &Bound<'_, PyArray1<f64>>,
    n: usize,
    d: usize,
    temperature: f64,
) -> PyResult<f64> {
    let anchors = readonly(anchors, "anchors")?;
    let positives = readonly(positives, "positives")?;
    let a = contiguous(&anchors, "anchors")?;
    let p = contiguous(&positives, "positives")?;
    let size = matrix_size(n, d)?;
    if size == 0 {
        return Ok(0.0);
    }
    require_prefix(a, size, "anchors")?;
    require_prefix(p, size, "positives")?;
    ensure_parallel_runtime()?;
    Ok(neural_decoders::infonce_loss(a, p, n, d, temperature))
}
