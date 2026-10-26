// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — GPU dense-layer ownership

//! Owned GPU buffers, original shader parameters and typed dense refusals.

use crate::layer::DenseLayerError;
use std::fmt;
use std::sync::{Arc, Mutex};

mod forward;
mod storage;

use bytemuck::{Pod, Zeroable};
use wgpu;

use super::context::GpuContext;
use crate::layer::DenseLayer;

// ---- Uniform structs (must match WGSL layout exactly) ----

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct EncodeParams {
    n_inputs: u32,
    words_per_input: u32,
    seed_lo: u32,
    seed_hi: u32,
    n_samples: u32,
    length: u32,
    _pad0: u32,
    _pad1: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct AccumParams {
    n_inputs: u32,
    n_neurons: u32,
    words_per_input: u32,
    inv_length: f32,
    n_samples: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

/// GPU-accelerated stochastic computing dense layer.
pub struct GpuDenseLayer {
    /// CPU fallback — always available.
    pub cpu: DenseLayer,
    ctx: Arc<GpuContext>,

    // Persistent GPU buffers.
    weight_buf: wgpu::Buffer,

    // Per-call buffers (pre-allocated for max_batch_size).
    input_prob_buf: wgpu::Buffer,
    packed_input_buf: wgpu::Buffer,
    output_buf: wgpu::Buffer,
    output_staging: wgpu::Buffer,
    encode_uniform_buf: wgpu::Buffer,
    accum_uniform_buf: wgpu::Buffer,

    max_batch_size: usize,
    call_lock: Mutex<()>,
    dimensions: (usize, usize, usize, usize),
}

/// Separate dense host allocation failures from GPU backend failures.
#[derive(Debug)]
pub enum GpuDenseError {
    /// Invalid dimensions, shapes, or exhausted host/device storage.
    Dense(DenseLayerError),
    /// Device submission, mapping, or synchronization failed.
    Backend(String),
}

impl fmt::Display for GpuDenseError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Dense(error) => fmt::Display::fmt(error, formatter),
            Self::Backend(message) => formatter.write_str(message),
        }
    }
}

impl std::error::Error for GpuDenseError {}

impl From<DenseLayerError> for GpuDenseError {
    fn from(error: DenseLayerError) -> Self {
        Self::Dense(error)
    }
}

impl From<std::collections::TryReserveError> for GpuDenseError {
    fn from(_: std::collections::TryReserveError) -> Self {
        Self::Dense(DenseLayerError::Allocation)
    }
}

impl GpuDenseLayer {
    /// Create a GPU layer, retaining the original optional-backend Rust API.
    ///
    /// # Panics
    /// Panics on invalid dimensions or storage/backend refusal. Foreign-language
    /// callers use [`Self::try_new_checked`] to propagate errors.
    pub fn try_new(
        n_inputs: usize,
        n_neurons: usize,
        length: usize,
        seed: u64,
        max_batch: usize,
    ) -> Option<Self> {
        Self::try_new_checked(n_inputs, n_neurons, length, seed, max_batch)
            .expect("GPU dense layer cannot be created")
    }

    /// Compute one sample using the original Philox GPU kernel.
    ///
    /// # Panics
    /// Panics on invalid shape or storage/backend refusal; use
    /// [`Self::try_forward_gpu`] for typed errors.
    pub fn forward_gpu(&self, inputs: &[f64], seed: u64) -> Vec<f64> {
        self.try_forward_gpu(inputs, seed)
            .expect("GPU dense inference failed")
    }

    /// Compute row-major samples using the original Philox GPU kernel.
    ///
    /// # Panics
    /// Panics on invalid shape or storage/backend refusal; use
    /// [`Self::try_forward_batch_gpu`] for typed errors.
    pub fn forward_batch_gpu(&self, inputs: &[f64], samples: usize, seed: u64) -> Vec<f64> {
        self.try_forward_batch_gpu(inputs, samples, seed)
            .expect("GPU dense inference failed")
    }

    /// Return the GPU adapter name.
    pub fn gpu_name(&self) -> &str {
        &self.ctx.adapter_name
    }
}
