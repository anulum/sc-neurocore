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
#[cfg(feature = "gpu")]
use crate::codecs::waveform_stream_gpu::GpuStreamEncoder;

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyWaveformLosslessEncoder>()?;
    module.add_class::<PyWaveformLosslessDecoder>()?;
    #[cfg(feature = "gpu")]
    module.add_class::<PyWaveformLosslessGpuEncoder>()?;
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

    /// Start an independent stream with the same configuration.
    fn reset(&mut self, py: Python<'_>) {
        py.detach(|| self.inner.reset());
    }

    /// Encode one C-contiguous int16 window; the GIL is released while coding.
    fn encode<'py>(
        &mut self,
        py: Python<'py>,
        window: PyReadonlyArray2<'_, i16>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let (samples, values) = copy_window(&window, self.channels)?;
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

    /// Start an independent stream and clear an earlier decoding error.
    fn reset(&mut self, py: Python<'_>) {
        py.detach(|| self.inner.reset());
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

/// Copy a contiguous Python-owned window for CPU or GPU work outside the GIL.
fn copy_window(window: &PyReadonlyArray2<'_, i16>, channels: usize) -> PyResult<(usize, Vec<i16>)> {
    let shape = window.shape();
    if shape[1] != channels {
        return Err(PyValueError::new_err(
            "window has the wrong number of channels",
        ));
    }
    let values = window
        .as_slice()
        .map_err(|error| PyValueError::new_err(format!("window must be C-contiguous: {error}")))?;
    Ok((shape[0], values.to_vec()))
}

/// Device-backed lossless packet encoder; packets use a CPU decoder and one group per channel.
#[cfg(feature = "gpu")]
#[pyclass(
    name = "WaveformLosslessGpuEncoder",
    module = "sc_neurocore_engine.sc_neurocore_engine"
)]
pub struct PyWaveformLosslessGpuEncoder {
    inner: GpuStreamEncoder,
    channels: usize,
}

#[cfg(feature = "gpu")]
#[pymethods]
impl PyWaveformLosslessGpuEncoder {
    #[new]
    fn new(py: Python<'_>, channels: usize, adc_bits: u32, max_samples: usize) -> PyResult<Self> {
        let inner = py
            .detach(|| GpuStreamEncoder::new(channels, adc_bits, max_samples))
            .map_err(PyValueError::new_err)?;
        Ok(Self { inner, channels })
    }

    /// Name of the physical adapter selected at construction.
    #[getter]
    fn device_name(&self) -> &str {
        &self.inner.adapter_name
    }

    /// Graphics API used by the adapter (for example Vulkan).
    #[getter]
    fn device_api(&self) -> &str {
        self.inner.device_api()
    }

    /// Start a fresh stream on the original device.
    fn reset(&mut self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| self.inner.reset())
            .map_err(PyValueError::new_err)
    }

    /// Transfer, encode and read back one C-contiguous int16 window outside the GIL.
    fn encode<'py>(
        &mut self,
        py: Python<'py>,
        window: PyReadonlyArray2<'_, i16>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let (samples, values) = copy_window(&window, self.channels)?;
        let inner = &mut self.inner;
        let packet = py
            .detach(|| inner.encode(&values, samples))
            .map_err(PyValueError::new_err)?;
        Ok(PyBytes::new(py, &packet))
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
