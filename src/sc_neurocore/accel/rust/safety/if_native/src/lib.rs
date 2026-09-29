// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust owned dense IF replay library

//! C library exposing the maintained canonical dense IF kernel with owned results.
#![deny(missing_docs)]

#[path = "../../ann_to_snn_ffi.rs"]
pub mod ffi;
#[path = "../../ann_to_snn.rs"]
pub mod kernel;
#[path = "../../ann_to_snn_ffi_types.rs"]
pub mod types;
