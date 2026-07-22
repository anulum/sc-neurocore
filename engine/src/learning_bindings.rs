// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Differentiable and stochastic learning PyO3 bindings

//! Python bindings for surrogate training, differentiable dense layers, attention, and GNNs.

use crate::{
    attention, grad, graph,
    matrix_inputs_binding::{extract_matrix_f64, reshape_flat_to_rows},
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Register the differentiable and stochastic learning classes.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySurrogateLif>()?;
    m.add_class::<PyDifferentiableDenseLayer>()?;
    m.add_class::<PyStochasticAttention>()?;
    m.add_class::<PyStochasticGraphLayer>()?;
    Ok(())
}

fn parse_surrogate(name: &str, k: Option<f32>) -> PyResult<grad::SurrogateType> {
    let normalized = name.to_ascii_lowercase().replace('-', "_");
    match normalized.as_str() {
        "fast_sigmoid" => Ok(grad::SurrogateType::FastSigmoid {
            k: k.unwrap_or(25.0),
        }),
        "superspike" | "super_spike" => Ok(grad::SurrogateType::SuperSpike {
            k: k.unwrap_or(100.0),
        }),
        "arctan" | "arc_tan" => Ok(grad::SurrogateType::ArcTan { k: k.unwrap_or(10.0) }),
        "straightthrough" | "straight_through" | "ste" => Ok(grad::SurrogateType::StraightThrough),
        _ => Err(PyValueError::new_err(format!(
            "Unknown surrogate '{}'. Use one of: fast_sigmoid, superspike, arctan, straight_through.",
            name
        ))),
    }
}

#[pyclass(
    name = "SurrogateLif",
    module = "sc_neurocore_engine.sc_neurocore_engine"
)]
pub struct PySurrogateLif {
    inner: grad::SurrogateLif,
}

#[pymethods]
impl PySurrogateLif {
    #[new]
    #[pyo3(signature = (
        data_width=16,
        fraction=8,
        v_rest=0,
        v_reset=0,
        v_threshold=256,
        refractory_period=2,
        surrogate="fast_sigmoid",
        k=None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        data_width: u32,
        fraction: u32,
        v_rest: i16,
        v_reset: i16,
        v_threshold: i16,
        refractory_period: i32,
        surrogate: &str,
        k: Option<f32>,
    ) -> PyResult<Self> {
        let surrogate = parse_surrogate(surrogate, k)?;
        Ok(Self {
            inner: grad::SurrogateLif::new(
                data_width,
                fraction,
                v_rest,
                v_reset,
                v_threshold,
                refractory_period,
                surrogate,
            ),
        })
    }

    #[pyo3(signature = (leak_k, gain_k, i_t, noise_in=0))]
    fn forward(&mut self, leak_k: i16, gain_k: i16, i_t: i16, noise_in: i16) -> (i32, i16) {
        self.inner.forward(leak_k, gain_k, i_t, noise_in)
    }

    fn backward(&mut self, grad_output: f32) -> PyResult<f32> {
        self.inner
            .backward(grad_output)
            .map_err(PyValueError::new_err)
    }

    fn clear_trace(&mut self) {
        self.inner.clear_trace();
    }

    fn reset(&mut self) {
        self.inner.reset();
    }

    fn trace_len(&self) -> usize {
        self.inner.trace_len()
    }
}

#[pyclass(
    name = "DifferentiableDenseLayer",
    module = "sc_neurocore_engine.sc_neurocore_engine"
)]
pub struct PyDifferentiableDenseLayer {
    inner: grad::DifferentiableDenseLayer,
}

#[pymethods]
impl PyDifferentiableDenseLayer {
    #[new]
    #[pyo3(signature = (
        n_inputs,
        n_neurons,
        length=1024,
        seed=24301,
        surrogate="fast_sigmoid",
        k=None
    ))]
    fn new(
        n_inputs: usize,
        n_neurons: usize,
        length: usize,
        seed: u64,
        surrogate: &str,
        k: Option<f32>,
    ) -> PyResult<Self> {
        let surrogate = parse_surrogate(surrogate, k)?;
        Ok(Self {
            inner: grad::DifferentiableDenseLayer::new(
                n_inputs, n_neurons, length, seed, surrogate,
            ),
        })
    }

    fn get_weights(&self) -> Vec<Vec<f64>> {
        self.inner.layer.get_weights()
    }

    #[pyo3(signature = (input_values, seed=44257))]
    fn forward(&mut self, input_values: Vec<f64>, seed: u64) -> PyResult<Vec<f64>> {
        self.inner
            .forward(&input_values, seed)
            .map_err(PyValueError::new_err)
    }

    fn backward(&self, grad_output: Vec<f64>) -> PyResult<(Vec<f64>, Vec<Vec<f64>>)> {
        self.inner
            .backward(&grad_output)
            .map_err(PyValueError::new_err)
    }

    fn update_weights(&mut self, weight_grads: Vec<Vec<f64>>, lr: f64) -> PyResult<()> {
        if weight_grads.len() != self.inner.layer.n_neurons {
            return Err(PyValueError::new_err(format!(
                "Expected {} grad rows, got {}.",
                self.inner.layer.n_neurons,
                weight_grads.len()
            )));
        }
        if weight_grads
            .iter()
            .any(|row| row.len() != self.inner.layer.n_inputs)
        {
            return Err(PyValueError::new_err(format!(
                "Expected each grad row to have length {}.",
                self.inner.layer.n_inputs
            )));
        }
        self.inner.update_weights(&weight_grads, lr);
        Ok(())
    }

    fn clear_cache(&mut self) {
        self.inner.clear_cache();
    }
}

#[pyclass(
    name = "StochasticAttention",
    module = "sc_neurocore_engine.sc_neurocore_engine"
)]
pub struct PyStochasticAttention {
    inner: attention::StochasticAttention,
}

#[pymethods]
impl PyStochasticAttention {
    #[new]
    #[pyo3(signature = (dim_k, temperature=None))]
    fn new(dim_k: usize, temperature: Option<f64>) -> Self {
        Self {
            inner: match temperature {
                Some(t) => attention::StochasticAttention::with_temperature(dim_k, t),
                None => attention::StochasticAttention::new(dim_k),
            },
        }
    }

    fn forward_softmax(
        &self,
        q: &Bound<'_, PyAny>,
        k: &Bound<'_, PyAny>,
        v: &Bound<'_, PyAny>,
    ) -> PyResult<Vec<Vec<f64>>> {
        let (q_data, q_rows, q_cols) = extract_matrix_f64(q, "Q")?;
        let (k_data, k_rows, k_cols) = extract_matrix_f64(k, "K")?;
        let (v_data, v_rows, v_cols) = extract_matrix_f64(v, "V")?;

        let out = self
            .inner
            .forward_softmax(
                &q_data, q_rows, q_cols, &k_data, k_rows, k_cols, &v_data, v_rows, v_cols,
            )
            .map_err(PyValueError::new_err)?;

        Ok(reshape_flat_to_rows(out, q_rows, v_cols))
    }

    #[pyo3(signature = (q, k, v, n_heads))]
    fn forward_multihead_softmax(
        &self,
        q: &Bound<'_, PyAny>,
        k: &Bound<'_, PyAny>,
        v: &Bound<'_, PyAny>,
        n_heads: usize,
    ) -> PyResult<Vec<Vec<f64>>> {
        let (q_data, q_rows, q_cols) = extract_matrix_f64(q, "Q")?;
        let (k_data, k_rows, k_cols) = extract_matrix_f64(k, "K")?;
        let (v_data, v_rows, v_cols) = extract_matrix_f64(v, "V")?;

        let out = self
            .inner
            .forward_multihead_softmax(
                &q_data, q_rows, q_cols, &k_data, k_rows, k_cols, &v_data, v_rows, v_cols, n_heads,
            )
            .map_err(PyValueError::new_err)?;

        let out_cols = v_cols;
        Ok(reshape_flat_to_rows(out, q_rows, out_cols))
    }

    fn forward(
        &self,
        q: &Bound<'_, PyAny>,
        k: &Bound<'_, PyAny>,
        v: &Bound<'_, PyAny>,
    ) -> PyResult<Vec<Vec<f64>>> {
        let (q_data, q_rows, q_cols) = extract_matrix_f64(q, "Q")?;
        let (k_data, k_rows, k_cols) = extract_matrix_f64(k, "K")?;
        let (v_data, v_rows, v_cols) = extract_matrix_f64(v, "V")?;

        let out = self
            .inner
            .forward(
                &q_data, q_rows, q_cols, &k_data, k_rows, k_cols, &v_data, v_rows, v_cols,
            )
            .map_err(PyValueError::new_err)?;

        Ok(reshape_flat_to_rows(out, q_rows, v_cols))
    }

    #[pyo3(signature = (q, k, v, length=1024, seed=44257))]
    fn forward_sc(
        &self,
        q: &Bound<'_, PyAny>,
        k: &Bound<'_, PyAny>,
        v: &Bound<'_, PyAny>,
        length: usize,
        seed: u64,
    ) -> PyResult<Vec<Vec<f64>>> {
        let (q_data, q_rows, q_cols) = extract_matrix_f64(q, "Q")?;
        let (k_data, k_rows, k_cols) = extract_matrix_f64(k, "K")?;
        let (v_data, v_rows, v_cols) = extract_matrix_f64(v, "V")?;

        let out = self
            .inner
            .forward_sc(
                &q_data, q_rows, q_cols, &k_data, k_rows, k_cols, &v_data, v_rows, v_cols, length,
                seed,
            )
            .map_err(PyValueError::new_err)?;

        Ok(reshape_flat_to_rows(out, q_rows, v_cols))
    }

    #[pyo3(signature = (q, k, v, n_heads))]
    fn forward_multihead(
        &self,
        q: &Bound<'_, PyAny>,
        k: &Bound<'_, PyAny>,
        v: &Bound<'_, PyAny>,
        n_heads: usize,
    ) -> PyResult<Vec<Vec<f64>>> {
        let (q_data, q_rows, q_cols) = extract_matrix_f64(q, "Q")?;
        let (k_data, k_rows, k_cols) = extract_matrix_f64(k, "K")?;
        let (v_data, v_rows, v_cols) = extract_matrix_f64(v, "V")?;

        let out = self
            .inner
            .forward_multihead(
                &q_data, q_rows, q_cols, &k_data, k_rows, k_cols, &v_data, v_rows, v_cols, n_heads,
            )
            .map_err(PyValueError::new_err)?;

        let out_cols = v_cols;
        Ok(reshape_flat_to_rows(out, q_rows, out_cols))
    }
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
