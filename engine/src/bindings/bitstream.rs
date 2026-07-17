// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Bitstream PyO3 binding

//! Python binding for bitstream operations that accept generic Python inputs.

use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Register bitstream bindings without adding implementation code to the crate root.
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(popcount, module)?)?;
    Ok(())
}

/// Count set bits in a one- or two-dimensional collection of packed words.
#[pyfunction]
fn popcount(packed: &Bound<'_, PyAny>) -> PyResult<u64> {
    // Zero-copy fast path: a 1-D numpy uint64 array borrows its buffer straight into the
    // SIMD dispatch instead of deep-copying every word into a Vec, as the `extract::<Vec…>`
    // paths below do. External review (KR-4) flagged that path as a large-array footgun;
    // this mirrors `popcount_numpy` so `popcount(np.ndarray)` no longer copies.
    if let Ok(array) = packed.extract::<PyReadonlyArray1<'_, u64>>() {
        return Ok(crate::simd::popcount_dispatch(array.as_slice()?));
    }

    if let Ok(rows) = packed.extract::<Vec<Vec<u64>>>() {
        return Ok(rows
            .iter()
            .map(|row| crate::simd::popcount_dispatch(row))
            .sum::<u64>());
    }

    let words = packed.extract::<Vec<u64>>().map_err(|_| {
        PyValueError::new_err("Expected packed uint64 words as 1-D or 2-D sequence.")
    })?;
    Ok(crate::simd::popcount_dispatch(&words))
}
