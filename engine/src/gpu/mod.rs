// SPDX-License-Identifier: AGPL-3.0-or-later
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li

//! GPU compute backend for SC-NeuroCore stochastic computing layers.
//!
//! Provides [`GpuDenseLayer`] — a GPU-accelerated version of
//! [`crate::layer::DenseLayer`] using wgpu compute shaders.
//!
//! # Feature gate
//!
//! This module is only available when the `gpu` feature is enabled:
//! ```sh
//! cargo build --features gpu
//! ```
//!
//! # Architecture
//!
//! Two compute shader kernels mirror the CPU hot path:
//! 1. **Encode** — Bernoulli sampling via Philox 4×32-10 GPU-native PRNG
//! 2. **Accumulate** — AND + popcount + workgroup reduction
//!
//! Weights are uploaded once and persist on the GPU. Input probabilities
//! and output values are transferred per forward call.

pub mod buffers;
pub mod context;
pub mod dense;

pub use context::is_available;
pub use dense::GpuDenseLayer;

// ---- PyO3 wrapper ----

use pyo3::prelude::*;

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
    #[new]
    #[pyo3(signature = (n_inputs, n_neurons, length=1024, seed=24301, max_batch=256))]
    fn new(
        n_inputs: usize,
        n_neurons: usize,
        length: usize,
        seed: u64,
        max_batch: usize,
    ) -> PyResult<Self> {
        let inner = GpuDenseLayer::try_new(n_inputs, n_neurons, length, seed, max_batch)
            .ok_or_else(|| {
                pyo3::exceptions::PyRuntimeError::new_err(
                    "No GPU available. Check Vulkan/Metal drivers.",
                )
            })?;
        Ok(PyGpuDenseLayer { inner })
    }

    /// Forward pass for a single sample (returns list of f64).
    #[pyo3(signature = (input_values, seed=42))]
    fn forward_fast(&self, input_values: Vec<f64>, seed: u64) -> PyResult<Vec<f64>> {
        Ok(self.inner.forward_gpu(&input_values, seed))
    }

    /// Batch forward pass. `inputs_flat` is row-major [n_samples × n_inputs].
    #[pyo3(signature = (inputs_flat, n_samples, seed=42))]
    fn forward_batch(
        &self,
        inputs_flat: Vec<f64>,
        n_samples: usize,
        seed: u64,
    ) -> PyResult<Vec<f64>> {
        Ok(self.inner.forward_batch_gpu(&inputs_flat, n_samples, seed))
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
