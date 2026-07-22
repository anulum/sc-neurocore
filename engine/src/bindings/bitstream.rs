// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Bitstream PyO3 binding

//! Python bindings for packing, unpacking, counting, and encoding bitstreams.

use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::IntoPyObject;

use crate::{bitstream, encoder, simd};

/// Register bitstream bindings without adding implementation code to the crate root.
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<Lfsr16>()?;
    module.add_class::<BitstreamEncoder>()?;
    module.add_function(wrap_pyfunction!(pack_bitstream, module)?)?;
    module.add_function(wrap_pyfunction!(unpack_bitstream, module)?)?;
    module.add_function(wrap_pyfunction!(popcount, module)?)?;
    module.add_function(wrap_pyfunction!(pack_bitstream_numpy, module)?)?;
    module.add_function(wrap_pyfunction!(popcount_numpy, module)?)?;
    module.add_function(wrap_pyfunction!(unpack_bitstream_numpy, module)?)?;
    module.add_function(wrap_pyfunction!(batch_encode, module)?)?;
    module.add_function(wrap_pyfunction!(batch_encode_numpy, module)?)?;
    Ok(())
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

/// Pack a one- or two-dimensional Python bit sequence into 64-bit words.
#[pyfunction]
fn pack_bitstream(py: Python<'_>, bits: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    if let Ok(rows) = bits.extract::<Vec<Vec<u8>>>() {
        let packed_rows: Vec<Vec<u64>> = rows.iter().map(|row| bitstream::pack(row).data).collect();
        return Ok(packed_rows
            .into_pyobject(py)
            .map_err(|e| PyValueError::new_err(e.to_string()))?
            .into_any()
            .unbind());
    }

    let flat = bits
        .extract::<Vec<u8>>()
        .map_err(|_| PyValueError::new_err("Expected a 1-D or 2-D array of uint8 bits."))?;
    Ok(bitstream::pack(&flat)
        .data
        .into_pyobject(py)
        .map_err(|e| PyValueError::new_err(e.to_string()))?
        .into_any()
        .unbind())
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
    if let Ok(rows) = packed.extract::<Vec<Vec<u64>>>() {
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

        let unpacked_rows: Vec<Vec<u8>> = rows
            .into_iter()
            .map(|row| {
                bitstream::unpack(&bitstream::BitStreamTensor::from_words(row, per_batch_len))
            })
            .collect();
        return Ok(unpacked_rows
            .into_pyobject(py)
            .map_err(|e| PyValueError::new_err(e.to_string()))?
            .into_any()
            .unbind());
    }

    let words = packed.extract::<Vec<u64>>().map_err(|_| {
        PyValueError::new_err("Expected packed uint64 words as 1-D or 2-D sequence.")
    })?;
    let tensor = bitstream::BitStreamTensor::from_words(words, original_length);
    Ok(bitstream::unpack(&tensor)
        .into_pyobject(py)
        .map_err(|e| PyValueError::new_err(e.to_string()))?
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

/// Pack a 1-D numpy uint8 array into packed u64 words, returning a numpy array.
/// Zero-copy input, single-allocation output.
#[pyfunction]
fn pack_bitstream_numpy<'py>(
    py: Python<'py>,
    bits: PyReadonlyArray1<'py, u8>,
) -> PyResult<Bound<'py, PyArray1<u64>>> {
    let slice = bits
        .as_slice()
        .map_err(|e| PyValueError::new_err(format!("Cannot read numpy array: {e}")))?;
    let tensor = simd::pack_dispatch(slice);
    Ok(tensor.data.into_pyarray(py))
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
    let tensor = bitstream::BitStreamTensor::from_words(words.to_vec(), original_length);
    let bits = bitstream::unpack(&tensor);
    Ok(bits.into_pyarray(py))
}

/// Bernoulli-encode a numpy float64 array into packed bitstream words.
///
/// Returns nested packed words with shape (n_probs, ceil(length / 64)).
#[pyfunction]
#[pyo3(signature = (probs, length=1024, seed=0xACE1))]
fn batch_encode<'py>(
    _py: Python<'py>,
    probs: PyReadonlyArray1<'py, f64>,
    length: usize,
    seed: u64,
) -> PyResult<Vec<Vec<u64>>> {
    let prob_slice = probs
        .as_slice()
        .map_err(|e| PyValueError::new_err(format!("Cannot read probs: {e}")))?;
    let words = length.div_ceil(64);

    use rand::SeedableRng;
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);

    let packed: Vec<Vec<u64>> = prob_slice
        .iter()
        .map(|&p| {
            let mut data = bitstream::bernoulli_packed(p, length, &mut rng);
            data.resize(words, 0);
            data
        })
        .collect();

    Ok(packed)
}

/// Bernoulli-encode a numpy float64 array into a 2-D numpy uint64 array.
///
/// Returns shape `(n_probs, ceil(length / 64))`.
#[pyfunction]
#[pyo3(signature = (probs, length=1024, seed=0xACE1))]
fn batch_encode_numpy<'py>(
    py: Python<'py>,
    probs: PyReadonlyArray1<'py, f64>,
    length: usize,
    seed: u64,
) -> PyResult<Bound<'py, PyArray2<u64>>> {
    use rayon::prelude::*;

    let prob_slice = probs
        .as_slice()
        .map_err(|e| PyValueError::new_err(format!("Cannot read probs: {e}")))?;
    let words = length.div_ceil(64);
    let n_probs = prob_slice.len();

    let rows: Vec<Vec<u64>> = prob_slice
        .par_iter()
        .enumerate()
        .map(|(idx, &p)| {
            use rand::SeedableRng;

            let prob_seed = seed.wrapping_add(idx as u64);
            let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(prob_seed);
            let mut row = bitstream::bernoulli_packed_simd(p, length, &mut rng);
            row.resize(words, 0);
            row
        })
        .collect();

    let mut flat = Vec::with_capacity(n_probs * words);
    for row in &rows {
        flat.extend_from_slice(row);
    }

    let arr = ndarray::Array2::from_shape_vec((n_probs, words), flat)
        .map_err(|e| PyValueError::new_err(format!("Shape construction failed: {e}")))?;
    Ok(arr.into_pyarray(py))
}
