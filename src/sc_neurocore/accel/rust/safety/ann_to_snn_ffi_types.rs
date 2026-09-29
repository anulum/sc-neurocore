// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Dense IF C ownership and buffer descriptors

//! Native ABI version one; all sizes use host usize and all values float64.

/// One borrowed dense layer and its optional initial state.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct LayerSpec {
    /// Positive first dimension of the output-by-input matrix.
    pub outputs: usize,
    /// Positive second dimension of the output-by-input matrix.
    pub inputs: usize,
    /// Live aligned row-major weight storage, outputs*inputs doubles.
    pub weights: *const f64,
    /// Optional live aligned bias; null with zero length means absent.
    pub bias: *const f64,
    /// Bias length, either zero or outputs.
    pub bias_len: usize,
    /// Positive finite inclusive IF threshold.
    pub threshold: f64,
    /// Finite membrane preload in threshold units.
    pub initial_fraction: f64,
    /// Optional batch/output initial state when request bit eight is set.
    pub initial: *const f64,
    /// Initial state length; ignored when request bit eight is unset.
    pub initial_len: usize,
}

/// Complete borrowed replay request. Buffers remain live and unchanged during the call.
#[repr(C)]
pub struct ReplayRequest {
    /// Exact ABI version, currently one.
    pub version: u32,
    /// Bits: one trace, two binary input, four linear final, eight supplied state.
    pub flags: u32,
    /// Live aligned array of nonempty layer descriptors.
    pub layers: *const LayerSpec,
    /// Number of connected dense layers.
    pub layer_count: usize,
    /// Explicit time/batch/input frame storage; null allowed for zero length.
    pub frames: *const f64,
    /// Frame storage length.
    pub frames_len: usize,
    /// Nonnegative timestep extent.
    pub steps: usize,
    /// Nonnegative batch extent.
    pub batch: usize,
    /// Positive addressable numeric buffer reservation limit.
    pub max_working_bytes: usize,
}

/// Borrowed mutable view into an owned opaque replay result.
#[repr(C)]
pub struct BufferView {
    /// Storage remains valid until its owning result handle is released.
    pub data: *mut f64,
    /// Number of doubles in the row-major buffer.
    pub len: usize,
}
