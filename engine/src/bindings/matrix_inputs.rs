// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — PyO3 matrix-input conversion

//! Shared matrix parsing and row reshaping for Python learning and solver bindings.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

pub(crate) fn extract_matrix_f64(
    data: &Bound<'_, PyAny>,
    name: &str,
) -> PyResult<(Vec<f64>, usize, usize)> {
    if let Ok(rows) = data.extract::<Vec<Vec<f64>>>() {
        if rows.is_empty() {
            return Err(PyValueError::new_err(format!(
                "{} must not be an empty matrix.",
                name
            )));
        }
        let row_count = rows.len();
        let cols = rows[0].len();
        if cols == 0 {
            return Err(PyValueError::new_err(format!(
                "{} must not have zero columns.",
                name
            )));
        }
        if rows.iter().any(|r| r.len() != cols) {
            return Err(PyValueError::new_err(format!(
                "{} must be a rectangular matrix.",
                name
            )));
        }
        let out = rows.into_iter().flatten().collect::<Vec<f64>>();
        return Ok((out, row_count, cols));
    }

    if let Ok(flat) = data.extract::<Vec<f64>>() {
        if flat.is_empty() {
            return Err(PyValueError::new_err(format!(
                "{} must not be an empty vector.",
                name
            )));
        }
        let cols = flat.len();
        return Ok((flat, 1, cols));
    }

    Err(PyValueError::new_err(format!(
        "{} must be a 1-D or 2-D float array.",
        name
    )))
}

pub(crate) fn reshape_flat_to_rows(flat: Vec<f64>, rows: usize, cols: usize) -> Vec<Vec<f64>> {
    let mut out = Vec::with_capacity(rows);
    for i in 0..rows {
        out.push(flat[i * cols..(i + 1) * cols].to_vec());
    }
    out
}
