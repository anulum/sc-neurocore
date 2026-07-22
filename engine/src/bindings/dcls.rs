// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — DCLS-max tent-kernel PyO3 binding

//! Python binding for the bit-true batched DCLS-max tent contraction.

use numpy::{IntoPyArray, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::scpn;

/// Register the batched DCLS-max tent contraction with the extension module.
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(py_dcls_max_forward_batch_q88, module)?)?;
    Ok(())
}

/// Batched DCLS-max triangular (tent) contraction in bit-true Q8.8 arithmetic.
///
/// Parity contract with `sc_neurocore.scpn.dcls_tent_kernel`: this Rust path,
/// the Mojo, Julia and Go backends, and the Python floor all return
/// bit-identical arrays because the kernel is exact integer arithmetic.
///
/// `spikes` and `weights_q88` are row-major `n_channels * n_taps`; `centres_q88`
/// and `sigmas_q88` carry one learnable `(centre, sigma)` per output channel.
///
/// Returns a dict with keys `outputs_q88` (int16), `accumulators_q16_16`
/// (int32), `overflow` (bool), `active_tap_counts` (int64) and `max_gates_q88`
/// (int16), each a 1-D array of length `n_channels`.
#[pyfunction]
#[pyo3(signature = (spikes, weights_q88, centres_q88, sigmas_q88, n_taps))]
fn py_dcls_max_forward_batch_q88<'py>(
    py: Python<'py>,
    spikes: PyReadonlyArray1<'py, u8>,
    weights_q88: PyReadonlyArray1<'py, i16>,
    centres_q88: PyReadonlyArray1<'py, i16>,
    sigmas_q88: PyReadonlyArray1<'py, i16>,
    n_taps: usize,
) -> PyResult<Py<PyAny>> {
    let result = scpn::dcls_max_forward_batch_q88(
        spikes.as_slice()?,
        weights_q88.as_slice()?,
        centres_q88.as_slice()?,
        sigmas_q88.as_slice()?,
        n_taps,
    )
    .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let active_tap_counts: Vec<i64> = result.active_tap_counts.iter().map(|&c| c as i64).collect();
    let d = PyDict::new(py);
    d.set_item("outputs_q88", result.outputs_q88.into_pyarray(py))?;
    d.set_item(
        "accumulators_q16_16",
        result.accumulators_q16_16.into_pyarray(py),
    )?;
    d.set_item("overflow", result.overflow.into_pyarray(py))?;
    d.set_item("active_tap_counts", active_tap_counts.into_pyarray(py))?;
    d.set_item("max_gates_q88", result.max_gates_q88.into_pyarray(py))?;
    Ok(d.into_any().unbind())
}
