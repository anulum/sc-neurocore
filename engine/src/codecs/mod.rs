// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Spike and waveform codec modules

//! Spike and waveform codecs: predictive XOR coding, the WaveformCodec kernels and the
//! streaming lossless coder.
//!
//! The source files keep their places in `src/` and `src/bindings/`; this
//! module groups them so the crate root does not grow.

#[path = "../predictive_coding.rs"]
pub mod predictive_coding;
#[path = "../bindings/predictive_coding.rs"]
mod predictive_coding_binding;
#[path = "../waveform_codec.rs"]
pub mod waveform_codec;
#[path = "../bindings/waveform_codec.rs"]
mod waveform_codec_binding;
#[path = "../waveform_stream.rs"]
pub mod waveform_stream;
#[path = "../bindings/waveform_stream.rs"]
mod waveform_stream_binding;
#[cfg(feature = "gpu")]
#[path = "../waveform_stream_gpu.rs"]
pub mod waveform_stream_gpu;

use pyo3::prelude::*;

/// Register the predictive-coding, WaveformCodec and streaming-coder bindings on the extension module.
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    predictive_coding_binding::register(module)?;
    waveform_codec_binding::register(module)?;
    waveform_stream_binding::register(module)
}
