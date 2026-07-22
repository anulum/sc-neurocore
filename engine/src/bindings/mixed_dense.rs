// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — mixed-precision dense PyO3 binding

//! Python binding for the bit-true Q8.8 by Q16.16 dense contraction.

use numpy::{IntoPyArray, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Register the mixed-precision dense contraction with the extension module.
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(
        py_mixed_dense_forward_batch_q88_q1616,
        module
    )?)?;
    Ok(())
}

/// Batched integer mixed-precision Q8.8 × Q16.16 dense MAC.
///
/// Parity contract with `sc_neurocore.compiler.mixed_dense_kernel`: this Rust
/// path and the Julia, Go, Mojo and Python backends return bit-identical arrays
/// because the integer branch (divisor equal to the Q8.8 weight scale) is exact.
///
/// `weights_q88` is row-major `n_outputs * n_inputs`; `inputs_q1616` is row-major
/// `n_batch * n_inputs`. Returns a dict with `outputs_q1616` (int32), `overflow`
/// (bool) and `underflow` (bool), each a 1-D array of length `n_batch * n_outputs`.
#[pyfunction]
#[pyo3(signature = (weights_q88, inputs_q1616, n_outputs, n_inputs))]
fn py_mixed_dense_forward_batch_q88_q1616<'py>(
    py: Python<'py>,
    weights_q88: PyReadonlyArray1<'py, i16>,
    inputs_q1616: PyReadonlyArray1<'py, i32>,
    n_outputs: usize,
    n_inputs: usize,
) -> PyResult<Py<PyAny>> {
    let result = crate::ir::qformat::mixed_dense_forward_batch_q88_q1616(
        weights_q88.as_slice()?,
        inputs_q1616.as_slice()?,
        n_outputs,
        n_inputs,
    )
    .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let d = PyDict::new(py);
    d.set_item("outputs_q1616", result.outputs_q1616.into_pyarray(py))?;
    d.set_item("overflow", result.overflow.into_pyarray(py))?;
    d.set_item("underflow", result.underflow.into_pyarray(py))?;
    Ok(d.into_any().unbind())
}
