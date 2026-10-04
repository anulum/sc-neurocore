// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Fallible native neuron output allocation

//! Propagate NumPy dimension and allocation errors without a Rust panic.

use ndarray::Dimension;
use numpy::{Element, PyArray, PyArray1, PyArray2, PyArrayDescrMethods, PY_ARRAY_API};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Allocate a zeroed contiguous vector while preserving NumPy errors.
pub(crate) fn zeros_1d<T: Element>(
    py: Python<'_>,
    length: usize,
) -> PyResult<Bound<'_, PyArray1<T>>> {
    zeros(py, [length])
}

/// Allocate a zeroed contiguous matrix while preserving NumPy errors.
pub(super) fn zeros_2d<T: Element>(
    py: Python<'_>,
    rows: usize,
    columns: usize,
) -> PyResult<Bound<'_, PyArray2<T>>> {
    zeros(py, [rows, columns])
}

/// Convert dimensions and own the C API's array or pending Python exception.
fn zeros<T: Element, D: Dimension, const N: usize>(
    py: Python<'_>,
    dimensions: [usize; N],
) -> PyResult<Bound<'_, PyArray<T, D>>> {
    let mut shape = [0_isize; N];
    for (target, dimension) in shape.iter_mut().zip(dimensions) {
        *target = isize::try_from(dimension)
            .map_err(|_| PyValueError::new_err("array dimensions must fit numpy.intp"))?;
    }
    // SAFETY: The private callers use rank one or two with that many checked
    // npy_intp dimensions alive for the call. Element supplies the matching
    // owned dtype descriptor, which PyArray_Zeros steals. NumPy returns a new
    // array reference or null with an exception; the fallible owner propagates
    // that exception. The checked cast verifies the returned rank and dtype.
    unsafe {
        let pointer = PY_ARRAY_API.PyArray_Zeros(
            py,
            N as i32,
            shape.as_mut_ptr(),
            T::get_dtype(py).into_dtype_ptr(),
            0,
        );
        Ok(Bound::from_owned_ptr_or_err(py, pointer)?.cast_into::<PyArray<T, D>>()?)
    }
}
