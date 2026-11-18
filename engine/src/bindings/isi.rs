// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — ISI codec Python bindings

//! Bind the inter-spike interval codec to NumPy rasters.

use numpy::{ndarray::Array2, IntoPyArray, PyArray2, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use crate::codecs::isi::{decode_raster, encode_raster, Entropy};

/// Encode a C-contiguous int8 raster and return the packet and encoded event count.
#[pyfunction]
fn isi_encode_raster<'py>(
    py: Python<'py>,
    raster: PyReadonlyArray2<'py, i8>,
    precision: usize,
    entropy: &str,
) -> PyResult<(Bound<'py, PyBytes>, usize)> {
    let entropy = match entropy {
        "auto" => Entropy::Auto,
        "varint" => Entropy::Varint,
        "huffman" => Entropy::Huffman,
        _ => {
            return Err(PyValueError::new_err(
                "entropy must be auto, varint or huffman",
            ));
        }
    };
    let shape = raster.shape();
    let values = raster
        .as_slice()
        .map_err(|_| PyValueError::new_err("ISI raster must be C-contiguous"))?;
    let (packet, events) = encode_raster(values, shape[0], shape[1], precision, entropy)
        .map_err(PyValueError::new_err)?;
    Ok((PyBytes::new(py, &packet), events))
}

/// Decode an ISI packet into a binary int8 raster with the requested dimensions.
#[pyfunction]
fn isi_decode_raster<'py>(
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
        .map_err(|_| PyValueError::new_err("ISI raster allocation shape is invalid"))?;
    Ok(raster.into_pyarray(py))
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(isi_encode_raster, module)?)?;
    module.add_function(wrap_pyfunction!(isi_decode_raster, module)?)
}
