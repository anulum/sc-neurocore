// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Checked NumPy analysis array borrowing and dimensions

//! Shared analysis input borrowing, layout, used-prefix and capacity contracts.

use numpy::{Element, PyArray1, PyArrayMethods, PyReadonlyArray1};
use pyo3::exceptions::{PyMemoryError, PyValueError};
use pyo3::prelude::*;

pub(super) fn readonly<'py, T: Element>(
    array: &Bound<'py, PyArray1<T>>,
    name: &str,
) -> PyResult<PyReadonlyArray1<'py, T>> {
    array
        .try_readonly()
        .map_err(|_| PyValueError::new_err(format!("{name} cannot be borrowed for reading")))
}

pub(super) fn contiguous<'a, T: Element>(
    array: &'a PyReadonlyArray1<'_, T>,
    name: &str,
) -> PyResult<&'a [T]> {
    array
        .as_slice()
        .map_err(|_| PyValueError::new_err(format!("{name} must be contiguous")))
}

pub(super) fn matrix_size(rows: usize, columns: usize) -> PyResult<usize> {
    rows.checked_mul(columns)
        .filter(|&size| size <= isize::MAX as usize / size_of::<f64>())
        .ok_or_else(|| PyMemoryError::new_err("requested matrix is too large"))
}

pub(super) fn require_prefix<T>(array: &[T], size: usize, name: &str) -> PyResult<()> {
    if array.len() < size {
        return Err(PyValueError::new_err(format!(
            "{name} must contain at least {size} elements"
        )));
    }
    Ok(())
}
