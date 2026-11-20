// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Spike and waveform codec modules

//! Spike and waveform codecs: ISI and delta packets, predictive XOR coding, waveform kernels, the
//! streaming lossless coder and the spike-event layer.
//!
//! The source files keep their places in `src/` and `src/bindings/`; this
//! module groups them so the crate root does not grow.

pub mod delta;
#[path = "../bindings/delta.rs"]
mod delta_binding;

pub mod isi;
#[path = "../bindings/isi.rs"]
mod isi_binding;

#[path = "../predictive_coding.rs"]
pub mod predictive_coding;
#[path = "../bindings/predictive_coding.rs"]
mod predictive_coding_binding;
#[path = "../spike_events.rs"]
pub mod spike_events;
#[path = "../bindings/spike_events.rs"]
mod spike_events_binding;
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

/// Register the raster, waveform and spike-event bindings on the extension module.
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    isi_binding::register(module)?;
    delta_binding::register(module)?;
    predictive_coding_binding::register(module)?;
    waveform_codec_binding::register(module)?;
    waveform_stream_binding::register(module)?;
    spike_events_binding::register(module)
}
