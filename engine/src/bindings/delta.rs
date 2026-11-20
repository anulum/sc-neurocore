// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Delta codec Python bindings

//! Bind complete DSCX packets to binary NumPy rasters.

use numpy::{ndarray::Array2, IntoPyArray, PyArray2, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use crate::codecs::delta::{decode_raster, encode_raster};
use crate::codecs::isi::Entropy;

/// Encode a contiguous binary int8 raster and return bytes and pre-quantisation counts.
#[pyfunction]
fn delta_encode_raster<'py>(
    py: Python<'py>,
    raster: PyReadonlyArray2<'py, i8>,
    group_size: usize,
    precision: usize,
    entropy: &str,
) -> PyResult<(Bound<'py, PyBytes>, usize, usize)> {
    let entropy = match entropy {
        "auto" => Entropy::Auto,
        "varint" => Entropy::Varint,
        "huffman" => Entropy::Huffman,
        _ => {
            return Err(PyValueError::new_err(
                "entropy must be auto, varint or huffman",
            ))
        }
    };
    let shape = raster.shape();
    let values = raster
        .as_slice()
        .map_err(|_| PyValueError::new_err("DSCX raster must be C-contiguous"))?;
    let result = encode_raster(values, shape[0], shape[1], group_size, precision, entropy)
        .map_err(PyValueError::new_err)?;
    Ok((
        PyBytes::new(py, &result.packet),
        result.spike_count,
        result.residual_count,
    ))
}

/// Decode a DSCX packet into an int8 raster or raise ValueError for invalid input.
#[pyfunction]
fn delta_decode_raster<'py>(
    py: Python<'py>,
    packet: &[u8],
    rows: usize,
    channels: usize,
    precision: usize,
) -> PyResult<Bound<'py, PyArray2<i8>>> {
    let values = py
        .detach(|| decode_raster(packet, rows, channels, precision))
        .map_err(PyValueError::new_err)?;
    let raster = Array2::from_shape_vec((rows, channels), values)
        .map_err(|_| PyValueError::new_err("DSCX raster allocation shape is invalid"))?;
    Ok(raster.into_pyarray(py))
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(delta_encode_raster, module)?)?;
    module.add_function(wrap_pyfunction!(delta_decode_raster, module)?)
}
