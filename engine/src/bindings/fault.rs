// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Byte-level fault-injection PyO3 bindings

//! Python bindings for deterministic byte-level hardware fault injection.

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

/// Register byte-level fault-injection functions with the extension module.
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(py_inject_bitflip_u8, module)?)?;
    module.add_function(wrap_pyfunction!(py_inject_stuck_at_0_u8, module)?)?;
    module.add_function(wrap_pyfunction!(py_inject_stuck_at_1_u8, module)?)?;
    module.add_function(wrap_pyfunction!(py_inject_dropout_u8, module)?)?;
    module.add_function(wrap_pyfunction!(py_inject_gaussian_u8, module)?)?;
    Ok(())
}

type FaultResult = (Py<PyArray1<u8>>, u64);

fn inject<'py>(
    py: Python<'py>,
    bitstream: PyReadonlyArray1<'_, u8>,
    ber: f64,
    seed: u64,
    operation: fn(&mut [u8], f64, u64) -> u64,
) -> PyResult<FaultResult> {
    let mut bytes = bitstream.as_slice()?.to_vec();
    let affected = operation(&mut bytes, ber, seed);
    Ok((bytes.into_pyarray(py).into(), affected))
}

/// Flip each byte-encoded bit independently with probability `ber`.
#[pyfunction]
fn py_inject_bitflip_u8<'py>(
    py: Python<'py>,
    bitstream: PyReadonlyArray1<'_, u8>,
    ber: f64,
    seed: u64,
) -> PyResult<FaultResult> {
    inject(py, bitstream, ber, seed, crate::fault::inject_bitflip_u8)
}

/// Force selected byte-encoded bits to zero.
#[pyfunction]
fn py_inject_stuck_at_0_u8<'py>(
    py: Python<'py>,
    bitstream: PyReadonlyArray1<'_, u8>,
    ber: f64,
    seed: u64,
) -> PyResult<FaultResult> {
    inject(py, bitstream, ber, seed, crate::fault::inject_stuck_at_0_u8)
}

/// Force selected byte-encoded bits to one.
#[pyfunction]
fn py_inject_stuck_at_1_u8<'py>(
    py: Python<'py>,
    bitstream: PyReadonlyArray1<'_, u8>,
    ber: f64,
    seed: u64,
) -> PyResult<FaultResult> {
    inject(py, bitstream, ber, seed, crate::fault::inject_stuck_at_1_u8)
}

/// Drop selected byte-encoded spikes to zero.
#[pyfunction]
fn py_inject_dropout_u8<'py>(
    py: Python<'py>,
    bitstream: PyReadonlyArray1<'_, u8>,
    ber: f64,
    seed: u64,
) -> PyResult<FaultResult> {
    inject(py, bitstream, ber, seed, crate::fault::inject_dropout_u8)
}

/// Apply thresholded Gaussian noise to byte-encoded bits.
#[pyfunction]
fn py_inject_gaussian_u8<'py>(
    py: Python<'py>,
    bitstream: PyReadonlyArray1<'_, u8>,
    ber: f64,
    seed: u64,
) -> PyResult<FaultResult> {
    inject(py, bitstream, ber, seed, crate::fault::inject_gaussian_u8)
}
