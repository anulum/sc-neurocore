// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — CORDIV PyO3 bindings

//! Python bindings for CORDIV stochastic division and stream-length planning.

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

/// Register CORDIV and adaptive stream-length functions with the extension module.
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(py_cordiv, module)?)?;
    module.add_function(wrap_pyfunction!(py_adaptive_length, module)?)?;
    Ok(())
}

/// Divide two byte-encoded stochastic bitstreams with the CORDIV recurrence.
#[pyfunction]
fn py_cordiv(
    py: Python<'_>,
    numerator: PyReadonlyArray1<'_, u8>,
    denominator: PyReadonlyArray1<'_, u8>,
) -> PyResult<Py<PyArray1<u8>>> {
    let numerator = numerator.as_slice()?;
    let denominator = denominator.as_slice()?;
    let quotient = crate::cordiv::cordiv(numerator, denominator);
    Ok(quotient.into_pyarray(py).into())
}

/// Compute a power-of-two stream length from the Hoeffding bound.
#[pyfunction]
fn py_adaptive_length(epsilon: f64, confidence: f64) -> usize {
    crate::cordiv::adaptive_length_hoeffding(epsilon, confidence)
}
