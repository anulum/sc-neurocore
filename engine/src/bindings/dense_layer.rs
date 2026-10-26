// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Dense-layer PyO3 binding

//! Python binding for the stochastic-computing dense layer.

use numpy::{
    PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods,
};
use pyo3::exceptions::{PyMemoryError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyList;

use crate::fixed_point_lif_binding::array_allocation::{zeros_1d, zeros_2d};
use crate::{layer, runtime_control_binding::ensure_parallel_runtime};

#[path = "dense_layer/sequences.rs"]
pub(crate) mod sequences;
use sequences::{float_list, float_matrix, Rows, Values};

/// Map typed storage exhaustion separately from authored input refusals.
fn python_error(error: layer::DenseLayerError) -> PyErr {
    match error {
        layer::DenseLayerError::Invalid(message) => PyValueError::new_err(message),
        layer::DenseLayerError::Allocation => {
            PyMemoryError::new_err("dense storage cannot be allocated")
        }
    }
}

/// Register the stochastic-computing dense layer with the extension module.
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<DenseLayer>()?;
    Ok(())
}

/// Python owner of a stochastic dense layer and its packed weights.
#[pyclass(module = "sc_neurocore_engine.sc_neurocore_engine")]
pub struct DenseLayer {
    inner: layer::DenseLayer,
}

#[pymethods]
impl DenseLayer {
    #[new]
    #[pyo3(signature = (n_inputs, n_neurons, length=1024, seed=24301))]
    /// Validate dimensions and admit workers before fallible weight allocation.
    fn new(n_inputs: usize, n_neurons: usize, length: usize, seed: u64) -> PyResult<Self> {
        if length == 0 {
            return Err(PyValueError::new_err("bitstream length must be > 0"));
        }
        if n_inputs == 0 {
            return Err(PyValueError::new_err("n_inputs must be > 0"));
        }
        if n_neurons == 0 {
            return Err(PyValueError::new_err("n_neurons must be > 0"));
        }
        ensure_parallel_runtime()?;
        Ok(Self {
            inner: layer::DenseLayer::try_new(n_inputs, n_neurons, length, seed)
                .map_err(python_error)?,
        })
    }

    /// Return independently owned rows, propagating Python allocation refusal.
    fn get_weights<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        float_matrix(py, &self.inner.weights)
    }

    /// Publish replacement probabilities and packed words atomically.
    fn set_weights(&mut self, weights: Rows<f64>) -> PyResult<()> {
        self.inner.try_set_weights(weights.0).map_err(python_error)
    }

    /// Rebuild packed words while preserving their old buffer on refusal.
    fn refresh_packed_weights(&mut self) -> PyResult<()> {
        self.inner
            .try_refresh_packed_weights()
            .map_err(python_error)
    }

    /// Encode scalar Bernoulli inputs with one shared sequential RNG.
    #[pyo3(signature = (input_values, seed=44257))]
    fn forward<'py>(
        &self,
        py: Python<'py>,
        input_values: Values<f64>,
        seed: u64,
    ) -> PyResult<Bound<'py, PyList>> {
        let output = self
            .inner
            .try_forward(&input_values.0, seed)
            .map_err(python_error)?;
        float_list(py, &output)
    }

    /// Encode SIMD Bernoulli inputs with independent input-index seeds.
    #[pyo3(signature = (input_values, seed=44257))]
    fn forward_fast<'py>(
        &self,
        py: Python<'py>,
        input_values: Values<f64>,
        seed: u64,
    ) -> PyResult<Bound<'py, PyList>> {
        let output = self
            .inner
            .try_forward_fast(&input_values.0, seed)
            .map_err(python_error)?;
        float_list(py, &output)
    }

    /// Dense forward accepting contiguous float64 input and returning float64 output.
    #[pyo3(signature = (input_values, seed=44257))]
    fn forward_numpy<'py>(
        &self,
        py: Python<'py>,
        input_values: PyReadonlyArray1<'py, f64>,
        seed: u64,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let slice = input_values
            .as_slice()
            .map_err(|error| PyValueError::new_err(format!("Cannot read input array: {error}")))?;
        let admitted = self.inner.validate_inputs(slice).map_err(python_error)?;
        let output = zeros_1d::<f64>(py, self.inner.n_neurons)?;
        // SAFETY: The new contiguous output is exclusively owned and
        // unpublished. Its storage cannot alias the borrowed input slice.
        let destination = unsafe { output.as_slice_mut() }
            .map_err(|_| PyValueError::new_err("dense output must be contiguous"))?;
        admitted
            .try_forward_fast_into(seed, destination)
            .map_err(python_error)?;
        Ok(output)
    }

    /// Dense forward for a contiguous float64 batch of shape `(samples, inputs)`.
    #[pyo3(signature = (inputs, seed=44257))]
    fn forward_batch_numpy<'py>(
        &self,
        py: Python<'py>,
        inputs: PyReadonlyArray2<'py, f64>,
        seed: u64,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let shape = inputs.shape();
        let samples = shape[0];
        let n_inputs = shape[1];
        if n_inputs != self.inner.n_inputs {
            return Err(PyValueError::new_err(format!(
                "Expected {} input features, got {}.",
                self.inner.n_inputs, n_inputs
            )));
        }
        let flat_inputs = inputs
            .as_slice()
            .map_err(|error| PyValueError::new_err(format!("Array not contiguous: {error}")))?;
        let output = zeros_2d::<f64>(py, samples, self.inner.n_neurons)?;
        // SAFETY: The array is newly allocated, contiguous and unpublished.
        // No alias can mutate it until it is returned after complete inference.
        let destination = unsafe { output.as_slice_mut() }
            .map_err(|_| PyValueError::new_err("dense output must be contiguous"))?;
        self.inner
            .try_forward_batch_unpublished_into(flat_inputs, samples, seed, destination)
            .map_err(python_error)?;
        Ok(output)
    }

    /// Forward from a 2-D uint64 NumPy array or nested Python integer sequence.
    fn forward_prepacked<'py>(
        &self,
        py: Python<'py>,
        packed_inputs: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyList>> {
        let rows = if let Ok(array) = packed_inputs.extract::<PyReadonlyArray2<u64>>() {
            let view = array.as_array();
            let mut rows = Vec::new();
            rows.try_reserve_exact(view.nrows())
                .map_err(|_| PyMemoryError::new_err("dense input cannot be allocated"))?;
            for row in view.rows() {
                let mut values = Vec::new();
                values
                    .try_reserve_exact(row.len())
                    .map_err(|_| PyMemoryError::new_err("dense input cannot be allocated"))?;
                values.extend(row.iter().copied());
                rows.push(values);
            }
            rows
        } else {
            packed_inputs
                .extract::<Rows<u64>>()
                .map_err(|error| {
                    if error.is_instance_of::<PyMemoryError>(py) {
                        error
                    } else {
                        PyValueError::new_err(
                            "packed_inputs must be a 2-D numpy uint64 array or list[list[int]].",
                        )
                    }
                })?
                .0
        };
        let values = self
            .inner
            .try_forward_prepacked(&rows)
            .map_err(python_error)?;
        float_list(py, &values)
    }

    /// Forward from borrowed contiguous NumPy uint64 `[input][word]` storage.
    #[pyo3(signature = (packed_inputs,))]
    fn forward_prepacked_numpy<'py>(
        &self,
        py: Python<'py>,
        packed_inputs: PyReadonlyArray2<'py, u64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let shape = packed_inputs.shape();
        let flat = packed_inputs
            .as_slice()
            .map_err(|error| PyValueError::new_err(format!("Array not contiguous: {error}")))?;
        let admitted = self
            .inner
            .validate_prepacked_2d(flat, shape[0], shape[1])
            .map_err(python_error)?;
        let output = zeros_1d::<f64>(py, self.inner.n_neurons)?;
        // SAFETY: This new contiguous array is exclusively owned and
        // unpublished; its output storage cannot alias borrowed packed input.
        let destination = unsafe { output.as_slice_mut() }
            .map_err(|_| PyValueError::new_err("dense output must be contiguous"))?;
        admitted
            .try_forward_into(destination)
            .map_err(python_error)?;
        Ok(output)
    }
}
