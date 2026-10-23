// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — SC Compute Graph IR

//! # SC Compute Graph IR
//!
//! A Rust-native intermediate representation for stochastic computing
//! pipelines. The IR captures the semantics of the planned MLIR "sc"
//! dialect (Blueprint §5) and can be lowered directly to synthesizable
//! SystemVerilog or exported as a text format for future MLIR/CIRCT
//! integration.
//!
//! # Design Principles
//!
//! - **SSA**: Every operation has a unique ID; terminators do not produce data values.
//! - **Typed**: All data values carry an `ScType` for static verification.
//! - **Acyclic**: The operation list forms a DAG (verified by `verify()`).
//! - **Portable**: Typed Rust enums and structs; source checksums use the crate's SHA-256 dependency.

pub(crate) mod bindings;
pub mod builder;
pub(crate) mod constant_words;
pub(crate) mod dense_contract;
pub mod emit_mlir;
pub mod emit_sv;
pub mod graph;
pub(crate) mod identifiers;
pub mod parser;
pub mod printer;
pub mod qformat;
pub mod runtime_check;
pub mod source;
pub(crate) mod state_contract;
pub(crate) mod stream_contract;
pub mod sv_target;
pub(crate) mod type_contract;
pub mod verify;
pub mod word;
pub(crate) mod word_contract;
