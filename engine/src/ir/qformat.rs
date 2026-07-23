// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Q-format and dense-quantisation compatibility facade

//! Stable public surface for fixed-point formats and dense quantisation.

mod block_floating;
mod dense_result;
mod fixed_format;
mod mixed_dense;

pub use block_floating::{
    block_floating_dense_q16, BlockExponentLayout, BlockFloatingDenseError, BlockFloatingError,
    BlockFloatingMode,
};
pub use dense_result::{MixedDenseResult, PrecisionEnvelopeReport, PrecisionTrapReport};
pub use fixed_format::{QFormat, QFormatError, QFormatMixed};
pub use mixed_dense::{
    mixed_dense_forward_batch_q88_q1616, mixed_dense_q88_q1616, MixedDenseBatchResult,
    MixedDenseError,
};
