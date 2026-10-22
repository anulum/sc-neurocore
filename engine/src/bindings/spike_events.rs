// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — PyO3 bindings of the spike-event layer

use numpy::{PyArray2, PyArrayMethods, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use crate::codecs::spike_events::{
    highpass, lowpass, threshold_factor, SpikeEventConfig, SpikeEventDecoder, SpikeEventEncoder,
};

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PySpikeEventEncoder>()?;
    module.add_class::<PySpikeEventDecoder>()?;
    module.add_function(wrap_pyfunction!(spike_events_filter_coefficients, module)?)?;
    Ok(())
}

/// Stateful spike-event encoder: ``encode(window)`` turns an int16 ``(samples, channels)`` window into a packet.
#[pyclass(
    name = "SpikeEventEncoder",
    module = "sc_neurocore_engine.sc_neurocore_engine"
)]
pub struct PySpikeEventEncoder {
    inner: SpikeEventEncoder,
    channels: usize,
}

#[pymethods]
impl PySpikeEventEncoder {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        channels: usize,
        sample_rate_hz: f64,
        bin_samples: usize,
        refractory_samples: usize,
        threshold_sigma: f64,
        highpass_hz: f64,
        lowpass_hz: f64,
        warmup_samples: u64,
        sample_timing: bool,
        power_bins: usize,
        power_shift: u32,
        groups: usize,
    ) -> PyResult<Self> {
        let inner = SpikeEventEncoder::new(SpikeEventConfig {
            channels,
            sample_rate_hz,
            bin_samples,
            refractory_samples,
            threshold_sigma,
            highpass_hz,
            lowpass_hz,
            warmup_samples,
            sample_timing,
            power_bins,
            power_shift,
            groups,
        })
        .map_err(PyValueError::new_err)?;
        Ok(PySpikeEventEncoder { inner, channels })
    }

    /// Encode a C-contiguous int16 ``(samples, channels)`` window into a packet.
    fn encode<'py>(
        &mut self,
        py: Python<'py>,
        window: PyReadonlyArray2<'py, i16>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let shape = window.shape();
        if shape[1] != self.channels {
            return Err(PyValueError::new_err(format!(
                "window must be (samples, {})",
                self.channels
            )));
        }
        let samples = shape[0];
        let data = window.as_slice()?;
        let inner = &mut self.inner;
        let packet = py
            .detach(|| inner.encode(data, samples))
            .map_err(PyValueError::new_err)?;
        Ok(PyBytes::new(py, &packet))
    }
}

/// Decoded packet: ``(first_bin, events, offsets or None, first_period, power)``.
type DecodedWindow<'py> = (
    u64,
    Bound<'py, PyArray2<u8>>,
    Option<Bound<'py, PyArray2<i16>>>,
    u64,
    Bound<'py, PyArray2<i32>>,
);

/// Stateful spike-event decoder matching ``SpikeEventEncoder``.
#[pyclass(
    name = "SpikeEventDecoder",
    module = "sc_neurocore_engine.sc_neurocore_engine"
)]
pub struct PySpikeEventDecoder {
    inner: SpikeEventDecoder,
    channels: usize,
    sample_timing: bool,
}

#[pymethods]
impl PySpikeEventDecoder {
    #[new]
    fn new(
        channels: usize,
        bin_samples: usize,
        sample_timing: bool,
        power_bins: usize,
        groups: usize,
    ) -> PyResult<Self> {
        let inner =
            SpikeEventDecoder::new(channels, bin_samples, sample_timing, power_bins, groups)
                .map_err(PyValueError::new_err)?;
        Ok(PySpikeEventDecoder {
            inner,
            channels,
            sample_timing,
        })
    }

    /// Decode the packet of a window of ``samples`` samples.
    fn decode<'py>(
        &mut self,
        py: Python<'py>,
        packet: &[u8],
        samples: usize,
    ) -> PyResult<DecodedWindow<'py>> {
        let inner = &mut self.inner;
        let window = py
            .detach(|| inner.decode(packet, samples))
            .map_err(PyValueError::new_err)?;
        let channels = self.channels;
        let bins = window.events.len() / channels;
        let periods = window.power.len() / channels;
        let events = numpy::PyArray1::from_vec(py, window.events).reshape([bins, channels])?;
        let offsets = if self.sample_timing {
            Some(numpy::PyArray1::from_vec(py, window.offsets).reshape([bins, channels])?)
        } else {
            None
        };
        let power = numpy::PyArray1::from_vec(py, window.power).reshape([periods, channels])?;
        Ok((
            window.first_bin,
            events,
            offsets,
            window.first_period,
            power,
        ))
    }
}

/// Q14 high-pass and low-pass coefficients ``(b0, b1, b2, a1, a2)`` and the Q12 threshold factor.
#[pyfunction]
#[allow(clippy::type_complexity)]
fn spike_events_filter_coefficients(
    sample_rate_hz: f64,
    highpass_hz: f64,
    lowpass_hz: f64,
    threshold_sigma: f64,
) -> PyResult<([i64; 5], [i64; 5], i64)> {
    let h = highpass(highpass_hz, sample_rate_hz).map_err(PyValueError::new_err)?;
    let l = lowpass(lowpass_hz, sample_rate_hz).map_err(PyValueError::new_err)?;
    Ok((
        [h.b0, h.b1, h.b2, h.a1, h.a2],
        [l.b0, l.b1, l.b2, l.a1, l.a2],
        threshold_factor(threshold_sigma),
    ))
}
