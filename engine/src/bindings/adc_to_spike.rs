// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — ADC-to-spike PyO3 binding

//! Python binding for decimating ADC samples into exact integer rate codes.

use numpy::{IntoPyArray, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::adc_to_spike;

/// Register the ADC-to-spike window encoder with the extension module.
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(py_adc_to_spike_windows, module)?)?;
    Ok(())
}

/// Encode raw ADC samples into per-window spike rate codes.
///
/// Parity contract with `sc_neurocore.sensors.adc_to_spike_kernel`: this Rust
/// path and the Julia, Go, Mojo and Python backends return bit-identical arrays
/// because the per-window quantise/average/rate-code arithmetic is exact integer.
///
/// `signed_input` is `0` for offset-binary or `1` for two's-complement ADC
/// samples. Returns a dict with `window_values_q` (int32), `spike_counts` (int32)
/// and `polarities` (bool), each of length `samples.len() / decimation`.
#[pyfunction]
#[pyo3(signature = (
    samples, adc_width, q_int, q_frac, decimation, signed_input, threshold_q,
))]
#[allow(clippy::too_many_arguments)]
fn py_adc_to_spike_windows<'py>(
    py: Python<'py>,
    samples: PyReadonlyArray1<'py, i64>,
    adc_width: u32,
    q_int: u32,
    q_frac: u32,
    decimation: u32,
    signed_input: i64,
    threshold_q: i64,
) -> PyResult<Py<PyAny>> {
    let result = adc_to_spike::adc_to_spike_windows(
        samples.as_slice()?,
        adc_width,
        q_int,
        q_frac,
        decimation,
        signed_input != 0,
        threshold_q,
    )
    .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let d = PyDict::new(py);
    d.set_item("window_values_q", result.window_values_q.into_pyarray(py))?;
    d.set_item("spike_counts", result.spike_counts.into_pyarray(py))?;
    d.set_item("polarities", result.polarities.into_pyarray(py))?;
    Ok(d.into_any().unbind())
}
