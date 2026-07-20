// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — HDC BitStreamTensor PyO3 binding

//! Python binding for the packed binary vector used by HDC/VSA operations.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rand::SeedableRng;

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
    fn new(dimension: usize, seed: u64) -> Self {
        let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(seed);
        let data = crate::bitstream::bernoulli_packed(0.5, dimension, &mut rng);
        Self {
            inner: crate::bitstream::BitStreamTensor::from_words(data, dimension),
        }
    }

    /// Create from pre-packed u64 words.
    #[staticmethod]
    fn from_packed(data: Vec<u64>, length: usize) -> PyResult<Self> {
        if length == 0 {
            return Err(PyValueError::new_err("bitstream length must be > 0"));
        }
        Ok(Self {
            inner: crate::bitstream::BitStreamTensor::from_words(data, length),
        })
    }

    /// In-place XOR (HDC bind).
    fn xor_inplace(&mut self, other: &PyBitStreamTensor) {
        self.inner.xor_inplace(&other.inner);
    }

    /// XOR returning a new tensor (HDC bind).
    fn xor(&self, other: &PyBitStreamTensor) -> PyBitStreamTensor {
        PyBitStreamTensor {
            inner: self.inner.xor(&other.inner),
        }
    }

    /// Cyclic right rotation by `shift` bits (HDC permute).
    fn rotate_right(&mut self, shift: usize) {
        self.inner.rotate_right(shift);
    }

    /// Normalized Hamming distance (0.0 = identical, 1.0 = opposite).
    fn hamming_distance(&self, other: &PyBitStreamTensor) -> f32 {
        self.inner.hamming_distance(&other.inner)
    }

    /// Majority-vote bundle of multiple tensors.
    #[staticmethod]
    fn bundle(vectors: Vec<PyRef<'_, PyBitStreamTensor>>) -> PyBitStreamTensor {
        let refs: Vec<&crate::bitstream::BitStreamTensor> =
            vectors.iter().map(|vector| &vector.inner).collect();
        PyBitStreamTensor {
            inner: crate::bitstream::BitStreamTensor::bundle(&refs),
        }
    }

    /// Count of set bits.
    fn popcount(&self) -> u64 {
        crate::bitstream::popcount(&self.inner)
    }

    /// Packed u64 words (read-only copy).
    #[getter]
    fn data(&self) -> Vec<u64> {
        self.inner.data.clone()
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
