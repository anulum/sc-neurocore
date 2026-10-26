// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Dense Python sequence ownership

//! Fallible sequence extraction and unpublished Python list construction.

use pyo3::conversion::FromPyObjectOwned;
use pyo3::exceptions::PyMemoryError;
use pyo3::prelude::*;
use pyo3::types::PyList;
use pyo3::Borrowed;

/// Own a flat sequence after fallible growth and ordinary element conversion.
pub(crate) struct Values<T>(pub(crate) Vec<T>);

impl<'py, T: FromPyObjectOwned<'py>> FromPyObject<'_, 'py> for Values<T> {
    type Error = PyErr;

    fn extract(object: Borrowed<'_, 'py, PyAny>) -> PyResult<Self> {
        crate::hdc_binding::extract_sequence(object, "dense input cannot be allocated").map(Self)
    }
}

/// Own every row of a nested Python sequence with fallible outer storage.
pub(super) struct Rows<T>(pub(super) Vec<Vec<T>>);

impl<'py, T: FromPyObjectOwned<'py>> FromPyObject<'_, 'py> for Rows<T> {
    type Error = PyErr;

    fn extract(object: Borrowed<'_, 'py, PyAny>) -> PyResult<Self> {
        let rows = crate::hdc_binding::extract_sequence::<Values<T>>(
            object,
            "dense input cannot be allocated",
        )?;
        let mut owned = Vec::new();
        owned
            .try_reserve_exact(rows.len())
            .map_err(|_| PyMemoryError::new_err("dense input cannot be allocated"))?;
        for row in rows {
            owned.push(row.0);
        }
        Ok(Self(owned))
    }
}

/// Allocate private list slots while propagating Python allocator errors.
fn allocate_list(py: Python<'_>, length: usize) -> PyResult<Bound<'_, PyList>> {
    let count = isize::try_from(length)
        .map_err(|_| PyMemoryError::new_err("dense output cannot be allocated"))?;
    // SAFETY: The checked length fits Py_ssize_t. PyList_New returns a new
    // reference or null with an exception, owned by the fallible constructor.
    // Unfilled slots remain private; dropping the list releases filled slots.
    unsafe {
        Ok(
            Bound::from_owned_ptr_or_err(py, pyo3::ffi::PyList_New(count))?
                .cast_into::<PyList>()?,
        )
    }
}

/// Fill an unpublished list with fallibly owned Python float objects.
pub(crate) fn float_list<'py>(py: Python<'py>, values: &[f64]) -> PyResult<Bound<'py, PyList>> {
    let output = allocate_list(py, values.len())?;
    for (index, &value) in values.iter().enumerate() {
        // SAFETY: PyFloat_FromDouble returns a new reference or null with a
        // Python exception; the fallible owner checks it before slot transfer.
        let object =
            unsafe { Bound::from_owned_ptr_or_err(py, pyo3::ffi::PyFloat_FromDouble(value))? };
        output.set_item(index, object)?;
    }
    Ok(output)
}

/// Copy all probability rows without publishing a partial Python matrix.
pub(super) fn float_matrix<'py>(
    py: Python<'py>,
    values: &[Vec<f64>],
) -> PyResult<Bound<'py, PyList>> {
    let output = allocate_list(py, values.len())?;
    for (index, row) in values.iter().enumerate() {
        output.set_item(index, float_list(py, row)?)?;
    }
    Ok(output)
}
