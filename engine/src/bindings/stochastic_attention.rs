// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Stochastic attention PyO3 binding

//! Python binding for stochastic attention kernels.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::{
    attention,
    matrix_inputs_binding::{extract_matrix_f64, reshape_flat_to_rows},
};

/// Register stochastic attention with the extension module.
pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyStochasticAttention>()?;
    Ok(())
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
