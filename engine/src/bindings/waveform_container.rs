// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Python bindings for complete waveform packets

use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use crate::codecs::waveform_container::{
    self, EncodedWaveform, SectionCompression, WaveformConfig, WaveformMode, WaveformSamples,
};

/// Register whole-container operations independently of the existing kernel exports.
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(waveform_container_encode, module)?)?;
    module.add_function(wrap_pyfunction!(waveform_container_decode, module)?)?;
    module.add("WAVEFORM_CONTAINER_ZSTD", cfg!(feature = "waveform-zstd"))?;
    Ok(())
}

#[derive(FromPyObject)]
struct PyWaveformConfig {
    mode: String,
    threshold_sigma: f64,
    snippet_samples: usize,
    max_templates: usize,
    template_threshold: f64,
    quantize_bits: u8,
    adc_bits: Option<u32>,
    zstandard: bool,
    denoise_background: bool,
}

impl TryFrom<PyWaveformConfig> for WaveformConfig {
    type Error = String;

    fn try_from(options: PyWaveformConfig) -> Result<Self, String> {
        Ok(Self {
            mode: WaveformMode::parse(&options.mode)?,
            threshold_sigma: options.threshold_sigma,
            snippet_samples: options.snippet_samples,
            max_templates: options.max_templates,
            template_threshold: options.template_threshold,
            quantize_bits: options.quantize_bits,
            adc_bits: options.adc_bits,
            compression: if options.zstandard {
                SectionCompression::Zstandard
            } else {
                SectionCompression::Zlib
            },
            denoise_background: options.denoise_background,
        })
    }
}

type EncodedParts<'py> = (Bound<'py, PyBytes>, usize, usize, usize, usize, usize);
type DecodedParts<'py> = (
    String,
    Option<Bound<'py, PyArray2<i8>>>,
    Option<Bound<'py, PyAny>>,
);

/// Encode the complete container from a contiguous int16 or float32 matrix.
#[pyfunction]
fn waveform_container_encode<'py>(
    py: Python<'py>,
    waveform: &Bound<'py, PyAny>,
    options: PyWaveformConfig,
) -> PyResult<EncodedParts<'py>> {
    let config = WaveformConfig::try_from(options).map_err(PyValueError::new_err)?;
    let encoded = if config.mode.is_lossless() {
        let (rows, channels, values) =
            copy_matrix(waveform.extract::<PyReadonlyArray2<'_, i16>>()?)?;
        py.detach(|| waveform_container::encode_lossless(&values, rows, channels, &config))
    } else {
        let (rows, channels, values) =
            copy_matrix(waveform.extract::<PyReadonlyArray2<'_, f32>>()?)?;
        py.detach(|| waveform_container::encode_lossy(&values, rows, channels, &config))
    }
    .map_err(PyValueError::new_err)?;
    Ok(encoding_parts(py, encoded))
}

fn copy_matrix<T: numpy::Element + Copy>(
    matrix: PyReadonlyArray2<'_, T>,
) -> PyResult<(usize, usize, Vec<T>)> {
    let shape = matrix.shape();
    let samples = matrix
        .as_slice()
        .map_err(|_| PyValueError::new_err("waveform must be C-contiguous"))?;
    Ok((shape[0], shape[1], samples.to_vec()))
}

fn encoding_parts(py: Python<'_>, encoded: EncodedWaveform) -> EncodedParts<'_> {
    (
        PyBytes::new(py, &encoded.packet),
        encoded.spikes,
        encoded.templates,
        encoded.spike_bytes,
        encoded.snippet_bytes,
        encoded.background_bytes,
    )
}

/// Decode the complete buffer with the supplied maximum number of sample values.
#[pyfunction]
fn waveform_container_decode<'py>(
    py: Python<'py>,
    packet: &[u8],
    max_samples: usize,
) -> PyResult<DecodedParts<'py>> {
    let bytes = packet.to_vec();
    let decoded = py
        .detach(|| waveform_container::decode(&bytes, max_samples))
        .map_err(PyValueError::new_err)?;
    let shape = [decoded.rows, decoded.channels];
    let raster = decoded
        .raster
        .map(|values| PyArray1::from_vec(py, values).reshape(shape))
        .transpose()?;
    let samples = match decoded.samples {
        Some(WaveformSamples::Exact(values)) => {
            Some(PyArray1::from_vec(py, values).reshape(shape)?.into_any())
        }
        Some(WaveformSamples::Approximate(values)) => {
            Some(PyArray1::from_vec(py, values).reshape(shape)?.into_any())
        }
        None => None,
    };
    Ok((decoded.mode.as_str().to_owned(), raster, samples))
}
