// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Fallible public bitstream packing

//! Pack Python sequences and typed NumPy input without infallible output conversions.

use std::collections::TryReserveError;

use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1};
use pyo3::exceptions::{PyMemoryError, PyValueError};
use pyo3::prelude::*;
use pyo3::sync::critical_section::with_critical_section;
use pyo3::types::{PyByteArray, PyByteArrayMethods, PyBytes, PyBytesMethods, PyList};
use pyo3::Borrowed;

use super::PackedSequence;
use crate::{bitstream, simd};

/// Preserve PyO3's byte-buffer fast paths and generic unsigned-byte extraction.
struct ByteSequence(Vec<u8>);

impl<'py> FromPyObject<'_, 'py> for ByteSequence {
    type Error = PyErr;

    fn extract(obj: Borrowed<'_, 'py, PyAny>) -> PyResult<Self> {
        if let Ok(bytes) = obj.cast::<PyBytes>() {
            return copy_bytes(bytes.to_owned().as_bytes())
                .map(Self)
                .map_err(|_| PyMemoryError::new_err("cannot allocate packed input sequence"));
        }
        if let Ok(array) = obj.cast::<PyByteArray>() {
            let array = array.to_owned();
            let copied = with_critical_section(array.as_any(), || {
                // SAFETY: The bytearray remains strongly owned and locked for
                // this copy, which performs no Python calls or detach. The
                // borrowed slice never escapes the critical section. Python
                // exception construction happens only after the lock ends.
                copy_bytes(unsafe { array.as_bytes() })
            });
            return copied
                .map(Self)
                .map_err(|_| PyMemoryError::new_err("cannot allocate packed input sequence"));
        }
        crate::hdc_binding::extract_sequence(obj, "cannot allocate packed input sequence").map(Self)
    }
}

/// Copy bytes only after reserving their complete storage, without Python callbacks.
fn copy_bytes(bytes: &[u8]) -> Result<Vec<u8>, TryReserveError> {
    let mut copied = Vec::new();
    copied.try_reserve_exact(bytes.len())?;
    copied.extend_from_slice(bytes);
    Ok(copied)
}

/// Allocate an unpublished Python list while preserving capacity and allocator errors.
pub(super) fn allocate_list(py: Python<'_>, length: usize) -> PyResult<Bound<'_, PyList>> {
    let length = isize::try_from(length)
        .map_err(|_| PyMemoryError::new_err("packed output cannot be allocated"))?;
    // SAFETY: The checked length fits Py_ssize_t. PyList_New returns a new
    // owned reference or null with a Python exception; the fallible owner
    // checks null before casting. Slots remain private until all are filled,
    // and dropping a partially filled list releases its non-null references.
    unsafe {
        Ok(
            Bound::from_owned_ptr_or_err(py, pyo3::ffi::PyList_New(length))?
                .cast_into::<PyList>()?,
        )
    }
}

/// Pack a row directly into fallibly allocated Python integers and list slots.
fn pack_row<'py>(py: Python<'py>, bits: &[u8]) -> PyResult<Bound<'py, PyList>> {
    let output = allocate_list(py, bits.len().div_ceil(64))?;
    for (index, chunk) in bits.chunks(64).enumerate() {
        let mut word = [0_u64];
        bitstream::pack_fast_into(chunk, &mut word).map_err(PyValueError::new_err)?;
        set_word(&output, index, word[0])?;
    }
    Ok(output)
}

/// Convert already packed words with checked list and unsigned-integer allocation.
pub(super) fn word_list<'py>(py: Python<'py>, words: &[u64]) -> PyResult<Bound<'py, PyList>> {
    let output = allocate_list(py, words.len())?;
    for (index, &word) in words.iter().enumerate() {
        set_word(&output, index, word)?;
    }
    Ok(output)
}

/// Own a fallibly allocated unsigned integer and transfer it into a list slot.
fn set_word(output: &Bound<'_, PyList>, index: usize, word: u64) -> PyResult<()> {
    // SAFETY: The constructor receives a valid u64 and returns an owned object
    // or null with a pending exception, checked by the fallible owner. Callers
    // own unpublished lists and pass an in-range index; set_item handles the
    // reference transfer and propagates any C API error.
    let value = unsafe {
        Bound::from_owned_ptr_or_err(output.py(), pyo3::ffi::PyLong_FromUnsignedLongLong(word))?
    };
    output.set_item(index, value)
}

/// Pack a one- or two-dimensional Python bit sequence into lists of 64-bit words.
///
/// Parameters
/// ----------
/// bits : sequence of uint8 or sequence of such sequences
///     Each nonzero byte sets a bit. Byte buffers, ragged rows and empty
///     sequences retain their existing admission and row shapes.
///
/// Returns
/// -------
/// list[int] or list[list[int]]
///     Packed words in least-significant-bit-first order, with zero high tail bits.
///
/// Raises
/// ------
/// ValueError
///     The input cannot be converted to a one- or two-dimensional byte sequence.
/// MemoryError
///     An input copy, Python list or Python integer cannot be allocated.
#[pyfunction]
pub(super) fn pack_bitstream(py: Python<'_>, bits: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let rows = bits.extract::<PackedSequence<ByteSequence>>();
    if let Err(error) = &rows {
        if error.is_instance_of::<PyMemoryError>(py) {
            return Err(error.clone_ref(py));
        }
    }
    if let Ok(PackedSequence(rows)) = rows {
        let output = allocate_list(py, rows.len())?;
        for (index, ByteSequence(row)) in rows.iter().enumerate() {
            output.set_item(index, pack_row(py, row)?)?;
        }
        return Ok(output.into_any().unbind());
    }

    let ByteSequence(flat) = bits.extract::<ByteSequence>().map_err(|error| {
        if error.is_instance_of::<PyMemoryError>(py) {
            error
        } else {
            PyValueError::new_err("Expected a 1-D or 2-D array of uint8 bits.")
        }
    })?;
    Ok(pack_row(py, &flat)?.into_any().unbind())
}

/// Borrow a contiguous uint8 array and fallibly allocate its owning packed NumPy result.
///
/// Parameters
/// ----------
/// bits : numpy.ndarray, uint8, shape (N,)
///     Contiguous input, optionally read-only. Each nonzero byte sets a bit.
///
/// Returns
/// -------
/// numpy.ndarray, uint64, shape (ceil(N / 64),)
///     Independent owning, writable output with least-significant-bit-first
///     ordering and zero unused high bits in a partial final word.
///
/// Raises
/// ------
/// TypeError
///     The input has a different rank or dtype.
/// ValueError
///     The input is not contiguous.
/// MemoryError
///     NumPy cannot allocate the output.
#[pyfunction]
pub(super) fn pack_bitstream_numpy<'py>(
    py: Python<'py>,
    bits: PyReadonlyArray1<'py, u8>,
) -> PyResult<Bound<'py, PyArray1<u64>>> {
    let slice = bits
        .as_slice()
        .map_err(|e| PyValueError::new_err(format!("Cannot read numpy array: {e}")))?;
    let output = crate::fixed_point_lif_binding::array_allocation::zeros_1d::<u64>(
        py,
        slice.len().div_ceil(64),
    )?;
    let mut writable = output.readwrite();
    simd::pack_dispatch_into(slice, writable.as_slice_mut()?).map_err(PyValueError::new_err)?;
    drop(writable);
    Ok(output)
}
