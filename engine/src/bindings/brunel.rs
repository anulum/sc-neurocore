// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Brunel-network PyO3 binding

//! Python binding for the fixed-point Brunel network simulator.

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::brunel;

/// Register the fixed-point Brunel network with the extension module.
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyBrunelNetwork>()?;
    Ok(())
}

#[pyclass(
    name = "BrunelNetwork",
    module = "sc_neurocore_engine.sc_neurocore_engine"
)]
pub struct PyBrunelNetwork {
    inner: brunel::BrunelNetwork,
}

#[pymethods]
impl PyBrunelNetwork {
    #[new]
    #[pyo3(signature = (
        n_neurons,
        w_indptr,
        w_indices,
        w_data,
        leak_k,
        gain_k,
        ext_lambda,
        ext_weight_fp,
        data_width=16,
        fraction=8,
        v_rest=0,
        v_reset=0,
        v_threshold=256,
        refractory_period=2,
        seed=42
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        n_neurons: usize,
        w_indptr: PyReadonlyArray1<'_, i64>,
        w_indices: PyReadonlyArray1<'_, i64>,
        w_data: PyReadonlyArray1<'_, i16>,
        leak_k: i16,
        gain_k: i16,
        ext_lambda: f64,
        ext_weight_fp: i16,
        data_width: u32,
        fraction: u32,
        v_rest: i16,
        v_reset: i16,
        v_threshold: i16,
        refractory_period: i32,
        seed: u64,
    ) -> PyResult<Self> {
        let indptr = w_indptr
            .as_slice()
            .map_err(|e| PyValueError::new_err(format!("Cannot read w_indptr: {e}")))?;
        let indices = w_indices
            .as_slice()
            .map_err(|e| PyValueError::new_err(format!("Cannot read w_indices: {e}")))?;
        let data = w_data
            .as_slice()
            .map_err(|e| PyValueError::new_err(format!("Cannot read w_data: {e}")))?;

        let row_offsets: Vec<usize> = indptr.iter().map(|&v| v as usize).collect();
        let col_indices: Vec<usize> = indices.iter().map(|&v| v as usize).collect();
        let values: Vec<i16> = data.to_vec();

        let inner = brunel::BrunelNetwork::new(
            n_neurons,
            row_offsets,
            col_indices,
            values,
            data_width,
            fraction,
            v_rest,
            v_reset,
            v_threshold,
            refractory_period,
            leak_k,
            gain_k,
            ext_lambda,
            ext_weight_fp,
            seed,
        )
        .map_err(PyValueError::new_err)?;

        Ok(Self { inner })
    }

    fn run<'py>(&mut self, py: Python<'py>, n_steps: usize) -> Bound<'py, PyArray1<u32>> {
        let counts = self.inner.run(n_steps);
        counts.into_pyarray(py)
    }
}
