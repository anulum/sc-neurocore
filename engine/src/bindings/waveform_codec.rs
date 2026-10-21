// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — PyO3 bindings of the WaveformCodec kernels

use numpy::{
    PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use crate::codecs::waveform_codec as kernels;

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(waveform_codec_noise_thresholds, module)?)?;
    module.add_function(wrap_pyfunction!(waveform_codec_extract_background, module)?)?;
    module.add_function(wrap_pyfunction!(waveform_codec_detect_spikes, module)?)?;
    module.add_function(wrap_pyfunction!(waveform_codec_template_match, module)?)?;
    module.add_function(wrap_pyfunction!(
        waveform_codec_quantize_background,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        waveform_codec_reconstruct_background,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        waveform_codec_lossless_encode_channel,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        waveform_codec_lossless_decode_channel,
        module
    )?)?;
    Ok(())
}

fn contiguous<'a, T: numpy::Element, D: numpy::ndarray::Dimension>(
    array: &'a numpy::PyReadonlyArray<'_, T, D>,
    name: &str,
) -> PyResult<&'a [T]> {
    array
        .as_slice()
        .map_err(|error| PyValueError::new_err(format!("{name} must be C-contiguous: {error}")))
}

/// Float32 negative spike thresholds per channel of a ``(T, N)`` waveform.
#[pyfunction]
fn waveform_codec_noise_thresholds(
    py: Python<'_>,
    waveform: PyReadonlyArray2<'_, f32>,
    negative_sigma_factor: f32,
) -> PyResult<Py<PyArray1<f32>>> {
    let shape = waveform.shape();
    let (t, n) = (shape[0], shape[1]);
    if t == 0 {
        return Err(PyValueError::new_err(
            "the waveform needs at least one sample",
        ));
    }
    let thresholds = kernels::noise_thresholds(
        contiguous(&waveform, "waveform")?,
        t,
        n,
        negative_sigma_factor,
    );
    Ok(PyArray1::from_vec(py, thresholds).unbind())
}

/// Float32 ``(rows, N)`` background with spike windows zeroed and 16-row averages.
#[pyfunction]
fn waveform_codec_extract_background(
    py: Python<'_>,
    waveform: PyReadonlyArray2<'_, f32>,
    times_per_channel: Vec<Vec<usize>>,
    half: usize,
) -> PyResult<Py<PyArray2<f32>>> {
    let shape = waveform.shape();
    let (t, n) = (shape[0], shape[1]);
    if times_per_channel.len() != n || times_per_channel.iter().flatten().any(|&time| time >= t) {
        return Err(PyValueError::new_err(
            "spike times must list one channel per column and lie inside the waveform",
        ));
    }
    let (rows, background) = kernels::extract_background(
        contiguous(&waveform, "waveform")?,
        t,
        n,
        &times_per_channel,
        half,
    );
    Ok(PyArray1::from_vec(py, background)
        .reshape([rows, n])?
        .unbind())
}

/// Spike times per channel of a C-contiguous float32 ``(T, N)`` waveform.
#[pyfunction]
fn waveform_codec_detect_spikes(
    waveform: PyReadonlyArray2<'_, f32>,
    thresholds: PyReadonlyArray1<'_, f32>,
    refractory: usize,
) -> PyResult<Vec<Vec<usize>>> {
    let shape = waveform.shape();
    let (t, n) = (shape[0], shape[1]);
    let thresholds = contiguous(&thresholds, "thresholds")?;
    if thresholds.len() != n {
        return Err(PyValueError::new_err(
            "one threshold per channel is required",
        ));
    }
    Ok(kernels::detect_spikes(
        contiguous(&waveform, "waveform")?,
        t,
        n,
        thresholds,
        refractory,
    ))
}

/// Template sources, template IDs, scales and ``(M, S)`` float32 residuals.
type TemplateMatchResult = (Vec<usize>, Vec<usize>, Vec<f64>, Py<PyArray2<f32>>);

/// Template sources, IDs, scales and float32 residuals of ``(M, S)`` snippets.
#[pyfunction]
fn waveform_codec_template_match(
    py: Python<'_>,
    snippets: PyReadonlyArray2<'_, f32>,
    max_templates: usize,
    threshold: f64,
) -> PyResult<TemplateMatchResult> {
    let shape = snippets.shape();
    let (m, samples) = (shape[0], shape[1]);
    let result = kernels::template_match(
        contiguous(&snippets, "snippets")?,
        m,
        samples,
        max_templates,
        threshold,
    );
    let residuals = PyArray1::from_vec(py, result.residuals).reshape([m, samples])?;
    Ok((
        result.template_sources,
        result.template_ids,
        result.scales,
        residuals.unbind(),
    ))
}

/// Closed-loop int8 codes of a C-contiguous float32 ``(rows, cols)`` background.
#[pyfunction]
fn waveform_codec_quantize_background(
    py: Python<'_>,
    background: PyReadonlyArray2<'_, f32>,
    step: f32,
    half: i32,
) -> PyResult<Py<PyArray2<i8>>> {
    let shape = background.shape();
    let (rows, cols) = (shape[0], shape[1]);
    let codes = kernels::quantize_background(
        contiguous(&background, "background")?,
        rows,
        cols,
        step,
        half,
    );
    Ok(PyArray1::from_vec(py, codes)
        .reshape([rows, cols])?
        .unbind())
}

/// Float32 ``(rows, cols)`` background rebuilt from its first row and codes.
#[pyfunction]
fn waveform_codec_reconstruct_background(
    py: Python<'_>,
    codes: PyReadonlyArray2<'_, i8>,
    first: PyReadonlyArray1<'_, f32>,
    step: f32,
) -> PyResult<Py<PyArray2<f32>>> {
    let shape = codes.shape();
    let (rows, cols) = (shape[0], shape[1]);
    let first = contiguous(&first, "first")?;
    if first.len() != cols {
        return Err(PyValueError::new_err(
            "the first row needs one value per column",
        ));
    }
    let recon =
        kernels::reconstruct_background(contiguous(&codes, "codes")?, rows, cols, first, step);
    Ok(PyArray1::from_vec(py, recon)
        .reshape([rows, cols])?
        .unbind())
}

/// Lossless section of one channel of int64 samples (at least one sample).
#[pyfunction]
fn waveform_codec_lossless_encode_channel<'py>(
    py: Python<'py>,
    samples: PyReadonlyArray1<'_, i64>,
) -> PyResult<Bound<'py, PyBytes>> {
    let samples = contiguous(&samples, "samples")?;
    if samples.is_empty() {
        return Err(PyValueError::new_err("a channel needs at least one sample"));
    }
    Ok(PyBytes::new(py, &kernels::lossless_encode_channel(samples)))
}

/// Int64 samples of one lossless channel section; malformed sections raise ValueError.
#[pyfunction]
fn waveform_codec_lossless_decode_channel(
    py: Python<'_>,
    section: &[u8],
    samples: usize,
) -> PyResult<Py<PyArray1<i64>>> {
    let values =
        kernels::lossless_decode_channel(section, samples).map_err(PyValueError::new_err)?;
    Ok(PyArray1::from_vec(py, values).unbind())
}
