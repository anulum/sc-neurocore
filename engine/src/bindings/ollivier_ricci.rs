// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Ollivier-Ricci PyO3 binding

//! Python binding for discrete Ollivier-Ricci graph curvature.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::topology::{self, CurvatureError};

/// Register the Ollivier-Ricci curvature function with the extension module.
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(py_ollivier_ricci_curvature, module)?)?;
    Ok(())
}

fn curvature_error_message(error: CurvatureError) -> &'static str {
    match error {
        CurvatureError::BadShape => "knm must be a square coupling matrix with at least one node",
        CurvatureError::BadValue => "knm must contain only finite, non-negative values",
        CurvatureError::BadIndex => "node index out of range for coupling graph",
        CurvatureError::Infeasible => "transport problem is infeasible",
    }
}

fn map_curvature_error(error: CurvatureError) -> PyErr {
    PyValueError::new_err(curvature_error_message(error))
}

/// Discrete Ollivier-Ricci curvature between two nodes of a coupling graph.
///
/// Parity contract with `sc_neurocore.math.topology.ollivier_ricci_curvature`:
/// for the same `knm` and `(i, j)`, the Rust value agrees with the Python
/// value to within float64 round-off.
///
/// `knm_flat` is the row-major `n x n` coupling matrix. Raises `ValueError`
/// on a malformed shape, a non-finite or negative entry, or an out-of-range
/// index, mirroring the Python validation.
#[pyfunction]
#[pyo3(signature = (knm_flat, n, i, j))]
fn py_ollivier_ricci_curvature(knm_flat: Vec<f64>, n: usize, i: usize, j: usize) -> PyResult<f64> {
    topology::ollivier_ricci_curvature(&knm_flat, n, i, j).map_err(map_curvature_error)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_messages_preserve_the_public_contract() {
        assert_eq!(
            curvature_error_message(CurvatureError::BadShape),
            "knm must be a square coupling matrix with at least one node"
        );
        assert_eq!(
            curvature_error_message(CurvatureError::BadValue),
            "knm must contain only finite, non-negative values"
        );
        assert_eq!(
            curvature_error_message(CurvatureError::BadIndex),
            "node index out of range for coupling graph"
        );
        assert_eq!(
            curvature_error_message(CurvatureError::Infeasible),
            "transport problem is infeasible"
        );
    }
}
