// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Address-event Python bindings

//! Bind complete AER packets to binary NumPy rasters.

use numpy::{ndarray::Array2, IntoPyArray, PyArray2, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use super::{decode_raster, encode_raster};

type EncodedPacket<'py> = (Bound<'py, PyBytes>, usize, usize);

/// Encode a contiguous binary int8 raster as AERX or AERI.
#[pyfunction]
fn address_event_encode_raster<'py>(
    py: Python<'py>,
    samples: PyReadonlyArray2<'py, i8>,
    timestamp_bits: u8,
    neuron_bits: u16,
) -> PyResult<EncodedPacket<'py>> {
    let shape = samples.shape();
    let slice = samples
        .as_slice()
        .map_err(|_| PyValueError::new_err("AER raster must be C-contiguous"))?;
    let result = encode_raster(slice, shape[0], shape[1], timestamp_bits, neuron_bits)
        .map_err(PyValueError::new_err)?;
    Ok((
        PyBytes::new(py, &result.packet),
        result.spikes,
        result.events,
    ))
}

/// Validate a complete packet before decoding to a NumPy raster.
#[pyfunction]
fn address_event_decode_raster<'py>(
    py: Python<'py>,
    packet: &[u8],
    rows: usize,
    channels: usize,
) -> PyResult<Bound<'py, PyArray2<i8>>> {
    let decoded = py
        .detach(|| decode_raster(packet, rows, channels))
        .map_err(PyValueError::new_err)?;
    let raster = Array2::from_shape_vec((decoded.rows, decoded.channels), decoded.samples)
        .map_err(|_| PyValueError::new_err("invalid AER raster array shape"))?;
    Ok(raster.into_pyarray(py))
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(address_event_encode_raster, module)?)?;
    module.add_function(wrap_pyfunction!(address_event_decode_raster, module)?)
}
