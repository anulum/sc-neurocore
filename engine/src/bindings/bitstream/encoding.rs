// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Fallible packed Bernoulli batch bindings

//! Preserve distinct scalar and parallel RNG traces with checked output allocation.

use numpy::{PyArray2, PyArrayMethods, PyReadonlyArray1};
use pyo3::exceptions::{PyMemoryError, PyValueError};
use pyo3::prelude::*;
use rand::SeedableRng;

use super::packing::{allocate_list, word_list};
use crate::bitstream;

/// Refuse an output whose total word storage cannot fit a native allocation.
fn validate_output(rows: usize, words: usize) -> PyResult<()> {
    let bytes = rows
        .checked_mul(words)
        .and_then(|count| count.checked_mul(std::mem::size_of::<u64>()));
    if bytes.is_none_or(|bytes| bytes > isize::MAX as usize) {
        return Err(PyMemoryError::new_err(
            "encoded bitstreams cannot be allocated",
        ));
    }
    Ok(())
}

/// Bernoulli-encode probabilities with a single sequential ChaCha8 stream.
///
/// Parameters
/// ----------
/// probs : numpy.ndarray, float64, shape (N,)
///     Contiguous probabilities, including read-only input. Sampling retains
///     clamping, NaN behavior and one f64 draw per bit for every row.
/// length : int, optional
///     Logical bits per row; zero is valid. Defaults to 1024.
/// seed : int, optional
///     Unsigned 64-bit ChaCha8 seed. Defaults to 44257.
///
/// Returns
/// -------
/// list[list[int]]
///     Packed words with shape `(N, ceil(length / 64))`.
///
/// Raises
/// ------
/// ValueError
///     The probability buffer is not contiguous.
/// MemoryError
///     Packed storage or Python output objects cannot be allocated.
#[pyfunction]
#[pyo3(signature = (probs, length=1024, seed=0xACE1))]
pub(super) fn batch_encode(
    py: Python<'_>,
    probs: PyReadonlyArray1<'_, f64>,
    length: usize,
    seed: u64,
) -> PyResult<Py<PyAny>> {
    let prob_slice = probs
        .as_slice()
        .map_err(|e| PyValueError::new_err(format!("Cannot read probs: {e}")))?;
    validate_output(prob_slice.len(), length.div_ceil(64))?;
    let output = allocate_list(py, prob_slice.len())?;
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
    for (index, &probability) in prob_slice.iter().enumerate() {
        let words = bitstream::try_bernoulli_packed(probability, length, &mut rng)
            .map_err(|_| PyMemoryError::new_err("encoded bitstreams cannot be allocated"))?;
        output.set_item(index, word_list(py, &words)?)?;
    }
    Ok(output.into_any().unbind())
}

/// Bernoulli-encode probabilities with independent Xoshiro/SIMD rows.
///
/// Parameters
/// ----------
/// probs : numpy.ndarray, float64, shape (N,)
///     Contiguous input, optionally read-only. Retain byte-threshold sampling,
///     clamped endpoint and NaN behavior, with wrapping `seed + row_index`.
/// length : int, optional
///     Logical bits per row; zero is valid. Defaults to 1024.
/// seed : int, optional
///     Unsigned 64-bit base Xoshiro seed. Defaults to 44257.
///
/// Returns
/// -------
/// numpy.ndarray, uint64, shape (N, ceil(length / 64))
///     Independent owning, writable, contiguous output with zero high tail bits.
///
/// Raises
/// ------
/// ValueError
///     The probability buffer is not contiguous.
/// MemoryError
///     NumPy output storage or the shared parallel workers cannot be allocated.
#[pyfunction]
#[pyo3(signature = (probs, length=1024, seed=0xACE1))]
pub(super) fn batch_encode_numpy<'py>(
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
    validate_output(prob_slice.len(), words)?;
    let output = crate::fixed_point_lif_binding::array_allocation::zeros_2d::<u64>(
        py,
        prob_slice.len(),
        words,
    )?;
    if words == 0 || prob_slice.is_empty() {
        return Ok(output);
    }
    crate::runtime_control_binding::ensure_parallel_runtime()?;
    let mut writable = output.readwrite();
    writable
        .as_slice_mut()?
        .par_chunks_mut(words)
        .zip(prob_slice.par_iter())
        .enumerate()
        .try_for_each(|(index, (row, &probability))| {
            let mut rng =
                rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(seed.wrapping_add(index as u64));
            bitstream::bernoulli_packed_simd_into(probability, length, &mut rng, row)
        })
        .map_err(PyValueError::new_err)?;
    drop(writable);
    Ok(output)
}
