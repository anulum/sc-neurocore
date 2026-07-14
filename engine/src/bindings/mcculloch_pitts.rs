// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — McCulloch-Pitts PyO3 batch binding

//! Python binding for the source-faithful McCulloch-Pitts batch contract.

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Register this binding without adding implementation code to the crate root.
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(py_mcculloch_pitts_evaluate_batch, module)?)?;
    Ok(())
}

/// Evaluate one fully validated varying-input batch.
#[pyfunction]
fn py_mcculloch_pitts_evaluate_batch<'py>(
    py: Python<'py>,
    theta: i64,
    excitatory_counts: PyReadonlyArray1<'py, i64>,
    inhibitory_flags: PyReadonlyArray1<'py, u8>,
) -> PyResult<(Bound<'py, PyArray1<u8>>, i64)> {
    let theta = i32::try_from(theta)
        .map_err(|_| PyValueError::new_err("theta must be in signed 32-bit range"))?;
    let neuron = crate::neurons::McCullochPittsNeuron::new(theta).map_err(PyValueError::new_err)?;
    let counts = excitatory_counts.as_slice()?;
    let flags = inhibitory_flags.as_slice()?;
    if counts.len() != flags.len() {
        return Err(PyValueError::new_err(
            "inhibitory_flags must match excitatory_counts length",
        ));
    }

    let mut validated = Vec::with_capacity(counts.len());
    for (&count, &flag) in counts.iter().zip(flags) {
        let count = i32::try_from(count).map_err(|_| {
            PyValueError::new_err("excitatory counts must be non-negative signed 32-bit integers")
        })?;
        if count < 0 {
            return Err(PyValueError::new_err(
                "excitatory counts must be non-negative signed 32-bit integers",
            ));
        }
        if flag > 1 {
            return Err(PyValueError::new_err(
                "inhibitory flags must contain only zero or one",
            ));
        }
        validated.push((count, flag != 0));
    }

    let mut event_count = 0_i64;
    let events: Vec<u8> = validated
        .into_iter()
        .map(|(count, inhibited)| {
            let event = neuron
                .try_step(count, inhibited)
                .expect("the complete batch was validated before evaluation");
            event_count += i64::from(event);
            event as u8
        })
        .collect();
    Ok((events.into_pyarray(py), event_count))
}
