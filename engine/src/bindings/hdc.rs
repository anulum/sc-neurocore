// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — HDC BitStreamTensor PyO3 binding

//! Python binding for the packed binary vector used by HDC/VSA operations.

use pyo3::conversion::FromPyObjectOwned;
use pyo3::exceptions::{PyMemoryError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyList, PySequence, PyString};
use pyo3::{Borrowed, CastError, PyTypeInfo};
use rand::SeedableRng;

/// Extract a Python sequence without infallible Rust capacity growth.
struct FallibleVec<T>(Vec<T>);

impl<'py, T> FromPyObject<'_, 'py> for FallibleVec<T>
where
    T: FromPyObjectOwned<'py>,
{
    type Error = PyErr;

    /// Preserve PyO3 sequence and element conversion while reserving fallibly.
    fn extract(obj: Borrowed<'_, 'py, PyAny>) -> PyResult<Self> {
        if obj.is_instance_of::<PyString>() {
            return Err(PyTypeError::new_err("Can't extract `str` to `Vec`"));
        }
        // SAFETY: the attached Python borrow keeps the inspected object alive.
        if unsafe { pyo3::ffi::PySequence_Check(obj.as_ptr()) } == 0 {
            return Err(CastError::new(obj, PySequence::type_object(obj.py()).into_any()).into());
        }
        let mut values = Vec::new();
        values
            .try_reserve_exact(obj.len().unwrap_or(0))
            .map_err(|_| PyMemoryError::new_err("cannot allocate HDC input sequence"))?;
        for item in obj.try_iter()? {
            let value = item?.extract::<T>().map_err(Into::into)?;
            if values.len() == values.capacity() {
                values
                    .try_reserve(1)
                    .map_err(|_| PyMemoryError::new_err("cannot allocate HDC input sequence"))?;
            }
            values.push(value);
        }
        Ok(Self(values))
    }
}

/// Register the HDC/VSA binding with the extension module.
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyBitStreamTensor>()?;
    Ok(())
}

/// Python wrapper for a packed binary hypervector.
#[pyclass(
    name = "BitStreamTensor",
    module = "sc_neurocore_engine.sc_neurocore_engine"
)]
pub struct PyBitStreamTensor {
    inner: crate::bitstream::BitStreamTensor,
}

#[pymethods]
impl PyBitStreamTensor {
    /// Create a random binary vector of `dimension` bits.
    #[new]
    #[pyo3(signature = (dimension=10000, seed=0xACE1))]
    fn new(dimension: usize, seed: u64) -> PyResult<Self> {
        if dimension == 0 {
            return Err(PyValueError::new_err("bitstream length must be > 0"));
        }
        let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(seed);
        let data = crate::bitstream::try_bernoulli_packed(0.5, dimension, &mut rng)
            .map_err(|_| PyMemoryError::new_err("cannot allocate packed bitstream"))?;
        Ok(Self {
            inner: crate::bitstream::BitStreamTensor::from_words(data, dimension),
        })
    }

    /// Create from exactly `ceil(length / 64)` words with zero unused high bits.
    #[staticmethod]
    fn from_packed(data: FallibleVec<u64>, length: usize) -> PyResult<Self> {
        let data = data.0;
        if length == 0 {
            return Err(PyValueError::new_err("bitstream length must be > 0"));
        }
        if data.len() != length.div_ceil(64) {
            return Err(PyValueError::new_err(
                "packed word count must equal ceil(length / 64)",
            ));
        }
        let trailing = length % 64;
        if trailing != 0 && data[data.len() - 1] >> trailing != 0 {
            return Err(PyValueError::new_err("unused packed bits must be zero"));
        }
        Ok(Self {
            inner: crate::bitstream::BitStreamTensor::from_words(data, length),
        })
    }

    /// In-place XOR (HDC bind), refusing unequal lengths before mutation.
    fn xor_inplace(&mut self, other: &PyBitStreamTensor) -> PyResult<()> {
        if self.inner.length != other.inner.length {
            return Err(PyValueError::new_err(
                "bitstream lengths must match for XOR",
            ));
        }
        self.inner.xor_inplace(&other.inner);
        Ok(())
    }

    /// XOR returning a new tensor (HDC bind).
    fn xor(&self, other: &PyBitStreamTensor) -> PyResult<PyBitStreamTensor> {
        if self.inner.length != other.inner.length {
            return Err(PyValueError::new_err(
                "bitstream lengths must match for XOR",
            ));
        }
        Ok(PyBitStreamTensor {
            inner: self
                .inner
                .try_xor(&other.inner)
                .map_err(|_| PyMemoryError::new_err("cannot allocate packed XOR output"))?,
        })
    }

    /// Cyclic right rotation by `shift` bits (HDC permute).
    fn rotate_right(&mut self, shift: usize) -> PyResult<()> {
        self.inner
            .try_rotate_right(shift)
            .map_err(|_| PyMemoryError::new_err("cannot allocate packed rotation output"))
    }

    /// Normalized Hamming distance (0.0 = identical, 1.0 = opposite).
    fn hamming_distance(&self, other: &PyBitStreamTensor) -> PyResult<f32> {
        if self.inner.length != other.inner.length {
            return Err(PyValueError::new_err(
                "bitstream lengths must match for Hamming distance",
            ));
        }
        Ok(self.inner.hamming_distance(&other.inner))
    }

    /// Majority-vote bundle of multiple tensors.
    #[staticmethod]
    fn bundle(vectors: FallibleVec<PyRef<'_, PyBitStreamTensor>>) -> PyResult<PyBitStreamTensor> {
        let vectors = vectors.0;
        let Some(first) = vectors.first() else {
            return Err(PyValueError::new_err("Cannot bundle zero vectors."));
        };
        if vectors
            .iter()
            .any(|vector| vector.inner.length != first.inner.length)
        {
            return Err(PyValueError::new_err(
                "bitstream lengths must match for bundle",
            ));
        }
        let mut refs = Vec::new();
        refs.try_reserve_exact(vectors.len())
            .map_err(|_| PyMemoryError::new_err("cannot allocate HDC input sequence"))?;
        refs.extend(vectors.iter().map(|vector| &vector.inner));
        Ok(PyBitStreamTensor {
            inner: crate::bitstream::BitStreamTensor::try_bundle(&refs)
                .map_err(|_| PyMemoryError::new_err("cannot allocate packed bundle output"))?,
        })
    }

    /// Count of set bits.
    fn popcount(&self) -> u64 {
        crate::bitstream::popcount(&self.inner)
    }

    /// Packed u64 words (read-only copy).
    #[getter]
    fn data<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let words = self
            .inner
            .try_clone()
            .map_err(|_| PyMemoryError::new_err("cannot allocate packed data copy"))?
            .data;
        // SAFETY: Vec<u64> length fits Py_ssize_t; PyList_New returns an owned
        // Python reference or null with a Python error. The attached token
        // owns that reference before any element is exposed to Python.
        let list = unsafe {
            Bound::from_owned_ptr_or_err(py, pyo3::ffi::PyList_New(words.len() as isize))
        }?
        .cast_into::<PyList>()?;
        for (index, word) in words.into_iter().enumerate() {
            // SAFETY: conversion preserves the unsigned 64-bit value and
            // returns an owned reference or null with a Python error.
            let item = unsafe {
                Bound::from_owned_ptr_or_err(py, pyo3::ffi::PyLong_FromUnsignedLongLong(word))
            }?;
            list.set_item(index, item)?;
        }
        Ok(list)
    }

    /// Logical bit length.
    #[getter]
    fn length(&self) -> usize {
        self.inner.length
    }

    fn __len__(&self) -> usize {
        self.inner.length
    }

    fn __repr__(&self) -> String {
        format!(
            "BitStreamTensor(length={}, popcount={})",
            self.inner.length,
            crate::bitstream::popcount(&self.inner)
        )
    }
}
