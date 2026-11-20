// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Predictive-coding PyO3 bindings

//! Python bindings for packed prediction error and lossless spike prediction codecs.

use numpy::{
    PyArray1, PyReadonlyArray1, PyReadwriteArray1, PyReadwriteArray2, PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Register predictive-coding functions with the extension module.
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(py_prediction_error, module)?)?;
    module.add_function(wrap_pyfunction!(py_predict_xor_ema, module)?)?;
    module.add_function(wrap_pyfunction!(py_predict_xor_lfsr, module)?)?;
    module.add_function(wrap_pyfunction!(py_recover_xor_ema, module)?)?;
    module.add_function(wrap_pyfunction!(py_recover_xor_lfsr, module)?)?;
    module.add_function(wrap_pyfunction!(py_predict_xor_context, module)?)?;
    module.add_function(wrap_pyfunction!(py_recover_xor_context, module)?)?;
    module.add_function(wrap_pyfunction!(py_update_spike_world_model, module)?)?;
    Ok(())
}

/// Return normalized XOR error between packed prediction and observation streams.
#[pyfunction]
fn py_prediction_error(
    _py: Python<'_>,
    predicted: PyReadonlyArray1<'_, u64>,
    actual: PyReadonlyArray1<'_, u64>,
    length: usize,
) -> PyResult<f64> {
    let predicted = predicted.as_slice().map_err(|error| {
        PyValueError::new_err(format!("predicted array must be contiguous: {error}"))
    })?;
    let actual = actual.as_slice().map_err(|error| {
        PyValueError::new_err(format!("actual array must be contiguous: {error}"))
    })?;
    Ok(crate::predictive_coding::prediction_error_packed(
        predicted, actual, length,
    ))
}

/// Encode a spike matrix as EMA-prediction XOR errors.
#[pyfunction]
fn py_predict_xor_ema(
    py: Python<'_>,
    spikes: PyReadonlyArray1<'_, i8>,
    n_channels: usize,
    alpha: f64,
    threshold: f64,
) -> PyResult<(Py<PyArray1<i8>>, usize)> {
    let spikes = spikes
        .as_slice()
        .map_err(|error| PyValueError::new_err(format!("spikes must be contiguous: {error}")))?;
    let (errors, correct) =
        crate::predictive_coding::predict_and_xor_ema(spikes, n_channels, alpha, threshold);
    Ok((PyArray1::from_vec(py, errors).into(), correct))
}

/// Recover a spike matrix from EMA-prediction XOR errors.
#[pyfunction]
fn py_recover_xor_ema(
    py: Python<'_>,
    errors: PyReadonlyArray1<'_, i8>,
    n_channels: usize,
    alpha: f64,
    threshold: f64,
) -> PyResult<Py<PyArray1<i8>>> {
    let errors = errors
        .as_slice()
        .map_err(|error| PyValueError::new_err(format!("errors must be contiguous: {error}")))?;
    let spikes =
        crate::predictive_coding::xor_and_recover_ema(errors, n_channels, alpha, threshold);
    Ok(PyArray1::from_vec(py, spikes).into())
}

/// Encode a spike matrix as seeded LFSR-prediction XOR errors.
#[pyfunction]
fn py_predict_xor_lfsr(
    py: Python<'_>,
    spikes: PyReadonlyArray1<'_, i8>,
    n_channels: usize,
    alpha_q8: i32,
    seed: u16,
) -> PyResult<(Py<PyArray1<i8>>, usize)> {
    let spikes = spikes
        .as_slice()
        .map_err(|error| PyValueError::new_err(format!("spikes must be contiguous: {error}")))?;
    let (errors, correct) =
        crate::predictive_coding::predict_and_xor_lfsr(spikes, n_channels, alpha_q8, seed);
    Ok((PyArray1::from_vec(py, errors).into(), correct))
}

/// Recover a spike matrix from seeded LFSR-prediction XOR errors.
#[pyfunction]
fn py_recover_xor_lfsr(
    py: Python<'_>,
    errors: PyReadonlyArray1<'_, i8>,
    n_channels: usize,
    alpha_q8: i32,
    seed: u16,
) -> PyResult<Py<PyArray1<i8>>> {
    let errors = errors
        .as_slice()
        .map_err(|error| PyValueError::new_err(format!("errors must be contiguous: {error}")))?;
    let spikes = crate::predictive_coding::xor_and_recover_lfsr(errors, n_channels, alpha_q8, seed);
    Ok(PyArray1::from_vec(py, spikes).into())
}

/// Encode row-major binary samples using per-channel context statistics.
#[pyfunction]
fn py_predict_xor_context(
    py: Python<'_>,
    spikes: PyReadonlyArray1<'_, i8>,
    n_channels: usize,
    history_bits: u8,
) -> PyResult<(Py<PyArray1<i8>>, usize)> {
    let input = spikes
        .as_slice()
        .map_err(|error| PyValueError::new_err(format!("spikes must be contiguous: {error}")))?
        .to_vec();
    let (errors, correct) = py
        .detach(move || {
            crate::predictive_coding::predict_and_xor_context(&input, n_channels, history_bits)
        })
        .map_err(PyValueError::new_err)?;
    Ok((PyArray1::from_vec(py, errors).into(), correct))
}

/// Recover row-major binary samples from per-channel context errors.
#[pyfunction]
fn py_recover_xor_context(
    py: Python<'_>,
    errors: PyReadonlyArray1<'_, i8>,
    n_channels: usize,
    history_bits: u8,
) -> PyResult<Py<PyArray1<i8>>> {
    let input = errors
        .as_slice()
        .map_err(|error| PyValueError::new_err(format!("errors must be contiguous: {error}")))?
        .to_vec();
    let spikes = py
        .detach(move || {
            crate::predictive_coding::xor_and_recover_context(&input, n_channels, history_bits)
        })
        .map_err(PyValueError::new_err)?;
    Ok(PyArray1::from_vec(py, spikes).into())
}

/// Update the autoregressive model after one observation using its exact prediction.
#[pyfunction]
fn py_update_spike_world_model(
    mut weights: PyReadwriteArray2<'_, f64>,
    mut bias: PyReadwriteArray1<'_, f64>,
    features: PyReadonlyArray1<'_, f64>,
    probabilities: PyReadonlyArray1<'_, f64>,
    samples: PyReadonlyArray1<'_, f64>,
    learning_rate: f64,
) -> PyResult<()> {
    let history = features.as_slice()?;
    let expected_shape = [bias.len(), history.len()];
    if weights.shape() != expected_shape {
        return Err(PyValueError::new_err(
            "world-model update dimensions must match",
        ));
    }
    // Keep the GIL while mutating storage owned by NumPy.
    crate::predictive_coding::update_world_model(
        weights.as_slice_mut()?,
        bias.as_slice_mut()?,
        history,
        probabilities.as_slice()?,
        samples.as_slice()?,
        learning_rate,
    )
    .map_err(PyValueError::new_err)
}
