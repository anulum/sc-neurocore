// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — GPU dense Python binding

//! Public GPU dense identity with fallible sequence and output ownership.

use super::dense::{GpuDenseError, GpuDenseLayer};
use super::is_available;
use crate::dense_layer_binding::sequences::{float_list, Values};
use crate::layer::DenseLayerError;
use pyo3::exceptions::{PyMemoryError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyList;

/// Preserve authored input errors separately from storage and backend failures.
fn python_error(error: GpuDenseError) -> PyErr {
    match error {
        GpuDenseError::Dense(DenseLayerError::Invalid(message)) => PyValueError::new_err(message),
        GpuDenseError::Dense(DenseLayerError::Allocation) => {
            PyMemoryError::new_err("dense storage cannot be allocated")
        }
        GpuDenseError::Backend(message) => PyRuntimeError::new_err(message),
    }
}

/// Python-facing GPU-accelerated DenseLayer.
///
/// Usage from Python:
/// ```python
/// from sc_neurocore_engine import GpuDenseLayer
/// layer = GpuDenseLayer(256, 128, length=1024, seed=42)
/// out = layer.forward_fast([0.5] * 256, seed=42)
/// ```
#[pyclass(
    name = "GpuDenseLayer",
    module = "sc_neurocore_engine.sc_neurocore_engine"
)]
pub struct PyGpuDenseLayer {
    inner: GpuDenseLayer,
}

#[pymethods]
impl PyGpuDenseLayer {
    /// Admit dimensions and fallible host/device storage before publishing the layer.
    #[new]
    #[pyo3(signature = (n_inputs, n_neurons, length=1024, seed=24301, max_batch=256))]
    fn new(
        n_inputs: usize,
        n_neurons: usize,
        length: usize,
        seed: u64,
        max_batch: usize,
    ) -> PyResult<Self> {
        for (value, message) in [
            (length, "bitstream length must be > 0"),
            (n_inputs, "n_inputs must be > 0"),
            (n_neurons, "n_neurons must be > 0"),
            (max_batch, "max_batch must be > 0"),
        ] {
            if value == 0 {
                return Err(PyValueError::new_err(message));
            }
        }
        crate::runtime_control_binding::ensure_parallel_runtime()?;
        let inner = GpuDenseLayer::try_new_checked(n_inputs, n_neurons, length, seed, max_batch)
            .map_err(python_error)?
            .ok_or_else(|| {
                pyo3::exceptions::PyRuntimeError::new_err(
                    "No GPU available. Check Vulkan/Metal drivers.",
                )
            })?;
        Ok(PyGpuDenseLayer { inner })
    }

    /// Forward pass for a single sample (returns list of f64).
    #[pyo3(signature = (input_values, seed=42))]
    fn forward_fast<'py>(
        &self,
        py: Python<'py>,
        input_values: Values<f64>,
        seed: u64,
    ) -> PyResult<Bound<'py, PyList>> {
        let output = self
            .inner
            .try_forward_gpu(&input_values.0, seed)
            .map_err(python_error)?;
        float_list(py, &output)
    }

    /// Batch forward pass. `inputs_flat` is row-major [n_samples × n_inputs].
    #[pyo3(signature = (inputs_flat, n_samples, seed=42))]
    fn forward_batch<'py>(
        &self,
        py: Python<'py>,
        inputs_flat: Values<f64>,
        n_samples: usize,
        seed: u64,
    ) -> PyResult<Bound<'py, PyList>> {
        let output = self
            .inner
            .try_forward_batch_gpu(&inputs_flat.0, n_samples, seed)
            .map_err(python_error)?;
        float_list(py, &output)
    }

    /// Name of the GPU adapter.
    fn gpu_name(&self) -> String {
        self.inner.gpu_name().to_string()
    }

    /// Check if a GPU is available (class method).
    #[staticmethod]
    fn is_gpu_available() -> bool {
        is_available()
    }

    /// Number of input features.
    #[getter]
    fn n_inputs(&self) -> usize {
        self.inner.cpu.n_inputs
    }

    /// Number of output neurons.
    #[getter]
    fn n_neurons(&self) -> usize {
        self.inner.cpu.n_neurons
    }

    /// Bitstream length.
    #[getter]
    fn length(&self) -> usize {
        self.inner.cpu.length
    }
}
