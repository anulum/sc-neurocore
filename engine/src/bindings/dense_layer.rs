// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Dense-layer PyO3 binding

//! Python binding for the stochastic-computing dense layer.

use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2,
    PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::layer;

/// Register the stochastic-computing dense layer with the extension module.
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<DenseLayer>()?;
    Ok(())
}

#[pyclass(module = "sc_neurocore_engine.sc_neurocore_engine")]
pub struct DenseLayer {
    inner: layer::DenseLayer,
}

#[pymethods]
impl DenseLayer {
    #[new]
    #[pyo3(signature = (n_inputs, n_neurons, length=1024, seed=24301))]
    fn new(n_inputs: usize, n_neurons: usize, length: usize, seed: u64) -> Self {
        Self {
            inner: layer::DenseLayer::new(n_inputs, n_neurons, length, seed),
        }
    }

    fn get_weights(&self) -> Vec<Vec<f64>> {
        self.inner.get_weights()
    }

    fn set_weights(&mut self, weights: Vec<Vec<f64>>) -> PyResult<()> {
        self.inner
            .set_weights(weights)
            .map_err(PyValueError::new_err)
    }

    fn refresh_packed_weights(&mut self) {
        self.inner.refresh_packed_weights();
    }

    #[pyo3(signature = (input_values, seed=44257))]
    fn forward(&self, input_values: Vec<f64>, seed: u64) -> PyResult<Vec<f64>> {
        self.inner
            .forward(&input_values, seed)
            .map_err(PyValueError::new_err)
    }

    #[pyo3(signature = (input_values, seed=44257))]
    fn forward_fast(&self, input_values: Vec<f64>, seed: u64) -> PyResult<Vec<f64>> {
        self.inner
            .forward_fast(&input_values, seed)
            .map_err(PyValueError::new_err)
    }

    /// Dense forward accepting numpy input and returning numpy output.
    ///
    /// This performs parallel encoding + parallel compute in one FFI call.
    #[pyo3(signature = (input_values, seed=44257))]
    fn forward_numpy<'py>(
        &self,
        py: Python<'py>,
        input_values: PyReadonlyArray1<'py, f64>,
        seed: u64,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let slice = input_values
            .as_slice()
            .map_err(|e| PyValueError::new_err(format!("Cannot read input array: {e}")))?;
        let out = self
            .inner
            .forward_numpy_inner(slice, seed)
            .map_err(PyValueError::new_err)?;
        Ok(out.into_pyarray(py))
    }

    /// Dense forward for a batch of input samples in one FFI call.
    ///
    /// `inputs` must be a contiguous float64 array of shape (n_samples, n_inputs).
    /// Returns float64 array of shape (n_samples, n_neurons).
    #[pyo3(signature = (inputs, seed=44257))]
    fn forward_batch_numpy<'py>(
        &self,
        py: Python<'py>,
        inputs: PyReadonlyArray2<'py, f64>,
        seed: u64,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let shape = inputs.shape();
        let n_samples = shape[0];
        let n_inputs = shape[1];
        if n_inputs != self.inner.n_inputs {
            return Err(PyValueError::new_err(format!(
                "Expected {} input features, got {}.",
                self.inner.n_inputs, n_inputs
            )));
        }

        let flat_inputs = inputs
            .as_slice()
            .map_err(|e| PyValueError::new_err(format!("Array not contiguous: {e}")))?;
        let out = PyArray2::<f64>::zeros(py, [n_samples, self.inner.n_neurons], false);
        // SAFETY: Newly allocated numpy arrays are contiguous.
        let out_slice = unsafe {
            out.as_slice_mut()
                .expect("newly allocated output array must be contiguous")
        };

        self.inner
            .forward_batch_into(flat_inputs, n_samples, seed, out_slice)
            .map_err(PyValueError::new_err)?;
        Ok(out)
    }

    /// Forward pass with pre-packed input bitstreams.
    ///
    /// Accepts either:
    /// - 2-D numpy array of dtype uint64 with shape (n_inputs, words)
    /// - list[list[int]]
    fn forward_prepacked(&self, packed_inputs: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
        if let Ok(arr) = packed_inputs.extract::<PyReadonlyArray2<u64>>() {
            let view = arr.as_array();
            let rows: Vec<Vec<u64>> = (0..view.nrows()).map(|i| view.row(i).to_vec()).collect();
            return self
                .inner
                .forward_prepacked(&rows)
                .map_err(PyValueError::new_err);
        }

        let rows = packed_inputs.extract::<Vec<Vec<u64>>>().map_err(|_| {
            PyValueError::new_err(
                "packed_inputs must be a 2-D numpy uint64 array or list[list[int]].",
            )
        })?;
        self.inner
            .forward_prepacked(&rows)
            .map_err(PyValueError::new_err)
    }

    /// Dense forward with pre-packed numpy 2-D input (true zero-copy).
    ///
    /// Accepts a contiguous numpy uint64 array of shape (n_inputs, words).
    /// This avoids all row-copying that the `forward_prepacked` method does.
    #[pyo3(signature = (packed_inputs,))]
    fn forward_prepacked_numpy<'py>(
        &self,
        py: Python<'py>,
        packed_inputs: PyReadonlyArray2<'py, u64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let shape = packed_inputs.shape();
        let n_inputs = shape[0];
        let words = shape[1];
        let flat = packed_inputs
            .as_slice()
            .map_err(|e| PyValueError::new_err(format!("Array not contiguous: {e}")))?;
        let out = self
            .inner
            .forward_prepacked_2d(flat, n_inputs, words)
            .map_err(PyValueError::new_err)?;
        Ok(out.into_pyarray(py))
    }
}
