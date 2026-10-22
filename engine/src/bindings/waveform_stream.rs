// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — PyO3 bindings of the streaming lossless waveform coder

use numpy::{
    PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use crate::codecs::waveform_stream::{detect_adc_bits, StreamDecoder, StreamEncoder};

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyWaveformLosslessEncoder>()?;
    module.add_class::<PyWaveformLosslessDecoder>()?;
    module.add_function(wrap_pyfunction!(waveform_codec_detect_adc_bits, module)?)?;
    Ok(())
}

/// Stateful lossless encoder: ``encode(window)`` turns an int16 ``(samples, channels)`` window into a packet.
#[pyclass(
    name = "WaveformLosslessEncoder",
    module = "sc_neurocore_engine.sc_neurocore_engine"
)]
pub struct PyWaveformLosslessEncoder {
    inner: StreamEncoder,
    channels: usize,
}

#[pymethods]
impl PyWaveformLosslessEncoder {
    #[new]
    fn new(channels: usize, adc_bits: u32, groups: usize) -> PyResult<Self> {
        let inner =
            StreamEncoder::new(channels, adc_bits, groups).map_err(PyValueError::new_err)?;
        Ok(Self { inner, channels })
    }

    /// Encode one C-contiguous int16 window; the GIL is released while coding.
    fn encode<'py>(
        &mut self,
        py: Python<'py>,
        window: PyReadonlyArray2<'_, i16>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let shape = window.shape();
        if shape[1] != self.channels {
            return Err(PyValueError::new_err(
                "window has the wrong number of channels",
            ));
        }
        let samples = shape[0];
        let values = window
            .as_slice()
            .map_err(|error| {
                PyValueError::new_err(format!("window must be C-contiguous: {error}"))
            })?
            .to_vec();
        let inner = &mut self.inner;
        let packet = py
            .detach(|| inner.encode(&values, samples))
            .map_err(PyValueError::new_err)?;
        Ok(PyBytes::new(py, &packet))
    }
}

/// Stateful lossless decoder: ``decode(packet, samples)`` returns the int16 ``(samples, channels)`` window.
#[pyclass(
    name = "WaveformLosslessDecoder",
    module = "sc_neurocore_engine.sc_neurocore_engine"
)]
pub struct PyWaveformLosslessDecoder {
    inner: StreamDecoder,
    channels: usize,
}

#[pymethods]
impl PyWaveformLosslessDecoder {
    #[new]
    fn new(channels: usize, adc_bits: u32, groups: usize) -> PyResult<Self> {
        let inner =
            StreamDecoder::new(channels, adc_bits, groups).map_err(PyValueError::new_err)?;
        Ok(Self { inner, channels })
    }

    /// Decode one packet of ``samples`` rows; malformed packets raise ValueError.
    fn decode(
        &mut self,
        py: Python<'_>,
        packet: &[u8],
        samples: usize,
    ) -> PyResult<Py<PyArray2<i16>>> {
        let bytes = packet.to_vec();
        let inner = &mut self.inner;
        let values = py
            .detach(|| inner.decode(&bytes, samples))
            .map_err(PyValueError::new_err)?;
        Ok(PyArray1::from_vec(py, values)
            .reshape([samples, self.channels])?
            .unbind())
    }
}

/// Smallest ADC width (1..16) whose level mapping is one-to-one on the int16 samples.
#[pyfunction]
fn waveform_codec_detect_adc_bits(samples: PyReadonlyArray1<'_, i16>) -> PyResult<u32> {
    let samples = samples
        .as_slice()
        .map_err(|error| PyValueError::new_err(format!("samples must be contiguous: {error}")))?;
    Ok(detect_adc_bits(samples))
}
