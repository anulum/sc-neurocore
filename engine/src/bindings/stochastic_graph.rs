// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Stochastic graph-layer PyO3 binding

//! Python binding for stochastic graph-layer kernels.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::{
    graph,
    matrix_inputs_binding::{extract_matrix_f64, reshape_flat_to_rows},
};

/// Register the stochastic graph layer with the extension module.
pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyStochasticGraphLayer>()?;
    Ok(())
}

#[pyclass(
    name = "StochasticGraphLayer",
    module = "sc_neurocore_engine.sc_neurocore_engine"
)]
pub struct PyStochasticGraphLayer {
    inner: graph::StochasticGraphLayer,
}

#[pymethods]
impl PyStochasticGraphLayer {
    #[new]
    #[pyo3(signature = (adj_matrix, n_features, seed=42))]
    fn new(adj_matrix: &Bound<'_, PyAny>, n_features: usize, seed: u64) -> PyResult<Self> {
        let (adj_flat, n_rows, n_cols) = extract_matrix_f64(adj_matrix, "adj_matrix")?;
        if n_rows != n_cols {
            return Err(PyValueError::new_err(format!(
                "adj_matrix must be square, got {}x{}.",
                n_rows, n_cols
            )));
        }
        Ok(Self {
            inner: graph::StochasticGraphLayer::new(adj_flat, n_rows, n_features, seed),
        })
    }

    /// Construct from CSR arrays (row_offsets, col_indices, values).
    #[staticmethod]
    #[pyo3(signature = (row_offsets, col_indices, values, n_nodes, n_features, seed=42))]
    fn from_sparse(
        row_offsets: Vec<usize>,
        col_indices: Vec<usize>,
        values: Vec<f64>,
        n_nodes: usize,
        n_features: usize,
        seed: u64,
    ) -> PyResult<Self> {
        let csr = graph::CsrMatrix::new(row_offsets, col_indices, values, n_nodes, n_nodes)
            .map_err(PyValueError::new_err)?;
        let inner = graph::StochasticGraphLayer::new_sparse(csr, n_features, seed)
            .map_err(PyValueError::new_err)?;
        Ok(Self { inner })
    }

    /// Dense adjacency with automatic CSR conversion if density < threshold.
    #[staticmethod]
    #[pyo3(signature = (adj_matrix, n_features, seed=42, density_threshold=0.3))]
    fn from_dense_auto(
        adj_matrix: &Bound<'_, PyAny>,
        n_features: usize,
        seed: u64,
        density_threshold: f64,
    ) -> PyResult<Self> {
        let (adj_flat, n_rows, n_cols) = extract_matrix_f64(adj_matrix, "adj_matrix")?;
        if n_rows != n_cols {
            return Err(PyValueError::new_err(format!(
                "adj_matrix must be square, got {}x{}.",
                n_rows, n_cols
            )));
        }
        Ok(Self {
            inner: graph::StochasticGraphLayer::from_dense_auto(
                adj_flat,
                n_rows,
                n_features,
                seed,
                density_threshold,
            ),
        })
    }

    fn is_sparse(&self) -> bool {
        self.inner.is_sparse()
    }

    fn forward(&self, node_features: &Bound<'_, PyAny>) -> PyResult<Vec<Vec<f64>>> {
        let (x_flat, x_rows, x_cols) = extract_matrix_f64(node_features, "node_features")?;
        if x_rows != self.inner.n_nodes || x_cols != self.inner.n_features {
            return Err(PyValueError::new_err(format!(
                "Expected node_features shape ({}, {}), got ({}, {}).",
                self.inner.n_nodes, self.inner.n_features, x_rows, x_cols
            )));
        }
        let out = self.inner.forward(&x_flat).map_err(PyValueError::new_err)?;
        Ok(reshape_flat_to_rows(
            out,
            self.inner.n_nodes,
            self.inner.n_features,
        ))
    }

    #[pyo3(signature = (node_features, length=1024, seed=44257))]
    fn forward_sc(
        &self,
        node_features: &Bound<'_, PyAny>,
        length: usize,
        seed: u64,
    ) -> PyResult<Vec<Vec<f64>>> {
        let (x_flat, x_rows, x_cols) = extract_matrix_f64(node_features, "node_features")?;
        if x_rows != self.inner.n_nodes || x_cols != self.inner.n_features {
            return Err(PyValueError::new_err(format!(
                "Expected node_features shape ({}, {}), got ({}, {}).",
                self.inner.n_nodes, self.inner.n_features, x_rows, x_cols
            )));
        }
        let out = self
            .inner
            .forward_sc(&x_flat, length, seed)
            .map_err(PyValueError::new_err)?;
        Ok(reshape_flat_to_rows(
            out,
            self.inner.n_nodes,
            self.inner.n_features,
        ))
    }

    fn get_weights(&self) -> Vec<f64> {
        self.inner.get_weights()
    }

    fn set_weights(&mut self, weights: Vec<f64>) -> PyResult<()> {
        self.inner
            .set_weights(weights)
            .map_err(PyValueError::new_err)
    }
}
