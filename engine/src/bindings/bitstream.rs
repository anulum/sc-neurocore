// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Bitstream PyO3 binding

//! Python bindings for packing, unpacking, counting, and encoding bitstreams.

use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1};
use pyo3::conversion::FromPyObjectOwned;
use pyo3::exceptions::{PyMemoryError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyList};
use pyo3::Borrowed;

use crate::{encoder, neuron, simd};

#[path = "bitstream/packing.rs"]
mod packing;

#[path = "bitstream/encoding.rs"]
mod encoding;

/// Preserve the packed input sequence protocol while reserving nested storage fallibly.
struct PackedSequence<T>(Vec<T>);

impl<'py, T: FromPyObjectOwned<'py>> FromPyObject<'_, 'py> for PackedSequence<T> {
    type Error = PyErr;

    fn extract(obj: Borrowed<'_, 'py, PyAny>) -> PyResult<Self> {
        crate::hdc_binding::extract_sequence(obj, "cannot allocate packed input sequence").map(Self)
    }
}

/// Refuse an output dimension that cannot fit Python or NumPy storage.
fn validate_unpack_length(length: usize) -> PyResult<()> {
    if length > isize::MAX as usize {
        return Err(PyMemoryError::new_err(
            "unpacked bitstream cannot be allocated",
        ));
    }
    Ok(())
}

/// Refuse unrepresentable output dimensions and truncated packed storage before allocation.
fn validate_unpack(words: usize, length: usize) -> PyResult<()> {
    validate_unpack_length(length)?;
    if words < length.div_ceil(64) {
        return Err(PyValueError::new_err(
            "packed words do not cover original_length",
        ));
    }
    Ok(())
}

/// Unpack directly into fallibly allocated Python bytes without intermediate native storage.
fn unpack_bytes<'py>(
    py: Python<'py>,
    words: &[u64],
    length: usize,
) -> PyResult<Bound<'py, PyBytes>> {
    validate_unpack(words.len(), length)?;
    PyBytes::new_with(py, length, |bits| {
        for (index, bit) in bits.iter_mut().enumerate() {
            *bit = ((words[index / 64] >> (index % 64)) & 1) as u8;
        }
        Ok(())
    })
}

/// Register bitstream bindings without adding implementation code to the crate root.
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<Lfsr16>()?;
    module.add_class::<BitstreamEncoder>()?;
    module.add_class::<PyBitstreamAverager>()?;
    module.add_function(wrap_pyfunction!(packing::pack_bitstream, module)?)?;
    module.add_function(wrap_pyfunction!(unpack_bitstream, module)?)?;
    module.add_function(wrap_pyfunction!(popcount, module)?)?;
    module.add_function(wrap_pyfunction!(packing::pack_bitstream_numpy, module)?)?;
    module.add_function(wrap_pyfunction!(popcount_numpy, module)?)?;
    module.add_function(wrap_pyfunction!(unpack_bitstream_numpy, module)?)?;
    module.add_function(wrap_pyfunction!(encoding::batch_encode, module)?)?;
    module.add_function(wrap_pyfunction!(encoding::batch_encode_numpy, module)?)?;
    Ok(())
}

#[pyclass(
    name = "BitstreamAverager",
    module = "sc_neurocore_engine.sc_neurocore_engine"
)]
pub struct PyBitstreamAverager {
    inner: neuron::BitstreamAverager,
}

#[pymethods]
impl PyBitstreamAverager {
    #[new]
    #[pyo3(signature = (window=1024))]
    fn new(window: usize) -> PyResult<Self> {
        if window == 0 {
            return Err(PyValueError::new_err("window must be > 0"));
        }
        let inner = neuron::BitstreamAverager::try_new(window).map_err(PyMemoryError::new_err)?;
        Ok(Self { inner })
    }

    fn push(&mut self, bit: u8) -> PyResult<()> {
        self.inner.try_push(bit).map_err(PyValueError::new_err)
    }

    fn estimate(&self) -> f64 {
        self.inner.estimate()
    }

    fn reset(&mut self) {
        self.inner.reset();
    }

    #[getter]
    fn window(&self) -> usize {
        self.inner.window()
    }
}

#[pyclass(module = "sc_neurocore_engine.sc_neurocore_engine")]
pub struct Lfsr16 {
    inner: encoder::Lfsr16,
    seed_init: u16,
}

#[pymethods]
impl Lfsr16 {
    #[new]
    #[pyo3(signature = (seed=0xACE1))]
    fn new(seed: u16) -> PyResult<Self> {
        if seed == 0 {
            return Err(PyValueError::new_err("LFSR seed must be non-zero."));
        }
        Ok(Self {
            inner: encoder::Lfsr16::new(seed),
            seed_init: seed,
        })
    }

    fn step(&mut self) -> u16 {
        self.inner.step()
    }

    #[getter]
    fn reg(&self) -> u16 {
        self.inner.reg
    }

    #[getter]
    fn width(&self) -> u32 {
        self.inner.width
    }

    #[pyo3(signature = (seed=None))]
    fn reset(&mut self, seed: Option<u16>) -> PyResult<()> {
        let next = seed.unwrap_or(self.seed_init);
        if next == 0 {
            return Err(PyValueError::new_err("LFSR seed must be non-zero."));
        }
        self.inner = encoder::Lfsr16::new(next);
        self.seed_init = next;
        Ok(())
    }
}

#[pyclass(module = "sc_neurocore_engine.sc_neurocore_engine")]
pub struct BitstreamEncoder {
    inner: encoder::BitstreamEncoder,
    seed_init: u16,
}

#[pymethods]
impl BitstreamEncoder {
    #[new]
    #[pyo3(signature = (data_width=16, seed=0xACE1))]
    fn new(data_width: u32, seed: u16) -> PyResult<Self> {
        if seed == 0 {
            return Err(PyValueError::new_err("LFSR seed must be non-zero."));
        }
        Ok(Self {
            inner: encoder::BitstreamEncoder::new(data_width, seed),
            seed_init: seed,
        })
    }

    fn step(&mut self, x_value: u16) -> u8 {
        self.inner.step(x_value)
    }

    #[getter]
    fn data_width(&self) -> u32 {
        self.inner.data_width
    }

    #[getter]
    fn reg(&self) -> u16 {
        self.inner.lfsr.reg
    }

    #[pyo3(signature = (seed=None))]
    fn reset(&mut self, seed: Option<u16>) -> PyResult<()> {
        let next = seed.unwrap_or(self.seed_init);
        if next == 0 {
            return Err(PyValueError::new_err("LFSR seed must be non-zero."));
        }
        self.inner.reset(Some(next));
        self.seed_init = next;
        Ok(())
    }
}

/// Unpack 64-bit words into their original one- or two-dimensional bit shape.
#[pyfunction]
#[pyo3(signature = (packed, original_length, original_shape=None))]
fn unpack_bitstream(
    py: Python<'_>,
    packed: &Bound<'_, PyAny>,
    original_length: usize,
    original_shape: Option<(usize, usize)>,
) -> PyResult<Py<PyAny>> {
    let rows = packed.extract::<PackedSequence<PackedSequence<u64>>>();
    if let Err(error) = &rows {
        if error.is_instance_of::<PyMemoryError>(py) {
            return Err(error.clone_ref(py));
        }
    }
    if let Ok(PackedSequence(rows)) = rows {
        let batch = rows.len();
        let per_batch_len = if let Some((expected_batch, length)) = original_shape {
            if expected_batch != batch {
                return Err(PyValueError::new_err(format!(
                    "original_shape batch {} does not match packed batch {}.",
                    expected_batch, batch
                )));
            }
            length
        } else {
            original_length.checked_div(batch).unwrap_or(0)
        };

        validate_unpack_length(per_batch_len)?;
        for row in &rows {
            validate_unpack(row.0.len(), per_batch_len)?;
        }
        let output = PyList::empty(py);
        for row in rows {
            output.append(unpack_bytes(py, &row.0, per_batch_len)?)?;
        }
        return Ok(output.into_any().unbind());
    }

    let words = packed.extract::<PackedSequence<u64>>().map_err(|error| {
        if error.is_instance_of::<PyMemoryError>(py) {
            error
        } else {
            PyValueError::new_err("Expected packed uint64 words as 1-D or 2-D sequence.")
        }
    })?;
    Ok(unpack_bytes(py, &words.0, original_length)?
        .into_any()
        .unbind())
}

/// Count set bits in a one- or two-dimensional collection of packed words.
#[pyfunction]
fn popcount(packed: &Bound<'_, PyAny>) -> PyResult<u64> {
    // Zero-copy fast path: a 1-D numpy uint64 array borrows its buffer straight into the
    // SIMD dispatch instead of deep-copying every word into a Vec, as the `extract::<Vec…>`
    // paths below do. External review (KR-4) flagged that path as a large-array footgun;
    // this mirrors `popcount_numpy` so `popcount(np.ndarray)` no longer copies.
    if let Ok(array) = packed.extract::<PyReadonlyArray1<'_, u64>>() {
        return Ok(simd::popcount_dispatch(array.as_slice()?));
    }

    if let Ok(rows) = packed.extract::<Vec<Vec<u64>>>() {
        return Ok(rows
            .iter()
            .map(|row| simd::popcount_dispatch(row))
            .sum::<u64>());
    }

    let words = packed.extract::<Vec<u64>>().map_err(|_| {
        PyValueError::new_err("Expected packed uint64 words as 1-D or 2-D sequence.")
    })?;
    Ok(simd::popcount_dispatch(&words))
}

/// Popcount on a numpy uint64 array — zero-copy input.
#[pyfunction]
fn popcount_numpy(packed: PyReadonlyArray1<'_, u64>) -> PyResult<u64> {
    let words = packed
        .as_slice()
        .map_err(|e| PyValueError::new_err(format!("Cannot read numpy array: {e}")))?;
    Ok(simd::popcount_dispatch(words))
}

/// Unpack a numpy uint64 array back to a numpy uint8 array.
#[pyfunction]
fn unpack_bitstream_numpy<'py>(
    py: Python<'py>,
    packed: PyReadonlyArray1<'py, u64>,
    original_length: usize,
) -> PyResult<Bound<'py, PyArray1<u8>>> {
    let words = packed
        .as_slice()
        .map_err(|e| PyValueError::new_err(format!("Cannot read numpy array: {e}")))?;
    validate_unpack(words.len(), original_length)?;
    let output =
        crate::fixed_point_lif_binding::array_allocation::zeros_1d::<u8>(py, original_length)?;
    let mut writable = output.readwrite();
    for (index, bit) in writable.as_slice_mut()?.iter_mut().enumerate() {
        *bit = ((words[index / 64] >> (index % 64)) & 1) as u8;
    }
    drop(writable);
    Ok(output)
}
