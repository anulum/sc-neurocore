// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Framed spike-raster Python bindings

//! Expose complete raster containers and their standalone frames.

use numpy::{ndarray::Array2, IntoPyArray, PyArray2, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use super::{decode_frame, decode_raster, encode_frame, encode_raster, DecodedRaster};

type BoundPacket<'py> = (Bound<'py, PyBytes>, usize, usize, usize);

fn as_array(py: Python<'_>, raster: DecodedRaster) -> PyResult<Bound<'_, PyArray2<i8>>> {
    let array = Array2::from_shape_vec((raster.rows, raster.channels), raster.samples)
        .map_err(|_| PyValueError::new_err("invalid raster array shape"))?;
    Ok(array.into_pyarray(py))
}

/// Encode a contiguous binary int8 array as an independent frame.
#[pyfunction]
fn raster_stream_encode_frame<'py>(
    py: Python<'py>,
    samples: PyReadonlyArray2<'py, i8>,
) -> PyResult<Bound<'py, PyBytes>> {
    let shape = samples.shape();
    let raster = samples
        .as_slice()
        .map_err(|_| PyValueError::new_err("raster must be C-contiguous"))?;
    let packet = encode_frame(raster, shape[0], shape[1]).map_err(PyValueError::new_err)?;
    Ok(PyBytes::new(py, &packet))
}

/// Decode exactly one standalone frame into a binary int8 array.
#[pyfunction]
fn raster_stream_decode_frame<'py>(
    py: Python<'py>,
    packet: &[u8],
) -> PyResult<Bound<'py, PyArray2<i8>>> {
    let raster = py
        .detach(|| decode_frame(packet))
        .map_err(PyValueError::new_err)?;
    as_array(py, raster)
}

/// Encode an SSCF container and return its spike count, active count and largest frame.
#[pyfunction]
fn raster_stream_encode_raster<'py>(
    py: Python<'py>,
    samples: PyReadonlyArray2<'py, i8>,
    window_size: usize,
) -> PyResult<BoundPacket<'py>> {
    let shape = samples.shape();
    let raster = samples
        .as_slice()
        .map_err(|_| PyValueError::new_err("raster must be C-contiguous"))?;
    let output =
        encode_raster(raster, shape[0], shape[1], window_size).map_err(PyValueError::new_err)?;
    Ok((
        PyBytes::new(py, &output.packet),
        output.spikes,
        output.active_channels,
        output.max_frame_bytes,
    ))
}

/// Decode an SSCF packet, with an optional row prefix and matching channel count.
#[pyfunction]
fn raster_stream_decode_raster<'py>(
    py: Python<'py>,
    packet: &[u8],
    rows: usize,
    channels: usize,
) -> PyResult<Bound<'py, PyArray2<i8>>> {
    let raster = py
        .detach(|| decode_raster(packet, rows, channels))
        .map_err(PyValueError::new_err)?;
    as_array(py, raster)
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(raster_stream_encode_frame, module)?)?;
    module.add_function(wrap_pyfunction!(raster_stream_decode_frame, module)?)?;
    module.add_function(wrap_pyfunction!(raster_stream_encode_raster, module)?)?;
    module.add_function(wrap_pyfunction!(raster_stream_decode_raster, module)?)
}
