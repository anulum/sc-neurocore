// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — stochastic-inference PyO3 binding

//! Python binding for inference over caller-owned packed stochastic weights.

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::sc_inference;

/// Register stochastic-inference bindings with the extension module.
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(py_sc_forward_packed, module)?)?;
    Ok(())
}

/// Stochastic forward pass over caller-owned packed weight bitstreams.
///
/// Parity contract with `sc_neurocore.accel.sc_forward`: this Rust path and the
/// NumPy fallback return bit-identical results for a fixed seed because the input
/// encoder is the deterministic 16-bit LFSR comparator.
///
/// `weights_packed` is row-major `n_out * n_in * n_words` (`n_words =
/// ceil(length / 64)`); `input_probs` is `n_in` float64 in `[0, 1]`. Returns an
/// `n_out` float64 array, the AND-then-popcount estimate of
/// `weights @ input_probs` divided by `length`.
#[pyfunction]
#[pyo3(signature = (weights_packed, n_out, n_in, n_words, input_probs, length, seed))]
#[allow(clippy::too_many_arguments)]
fn py_sc_forward_packed<'py>(
    py: Python<'py>,
    weights_packed: PyReadonlyArray1<'py, u64>,
    n_out: usize,
    n_in: usize,
    n_words: usize,
    input_probs: PyReadonlyArray1<'py, f64>,
    length: usize,
    seed: u64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let outputs = sc_inference::sc_forward_packed(
        weights_packed.as_slice()?,
        n_out,
        n_in,
        n_words,
        input_probs.as_slice()?,
        length,
        seed,
    )
    .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(outputs.into_pyarray(py))
}
