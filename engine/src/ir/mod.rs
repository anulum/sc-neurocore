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
//! - **SSA**: Every operation produces exactly one named value.
//! - **Typed**: All values carry an `ScType` for static verification.
//! - **Acyclic**: The operation list forms a DAG (verified by `verify()`).
//! - **Portable**: No external dependencies; pure Rust enums and structs.

pub mod builder;
pub mod emit_sv;
pub mod graph;
pub mod parser;
pub mod printer;
pub mod verify;
