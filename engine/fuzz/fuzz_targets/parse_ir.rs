// SPDX-License-Identifier: AGPL-3.0-or-later
//! Fuzz target for the SC IR text parser.
//!
//! `ir::parser::parse` accepts an arbitrary `&str` (it is exposed to Python as
//! `ir_parse`), so it must never panic on malformed input — it may only return a
//! `ParseError`. This target feeds it arbitrary UTF-8 and asserts (via the absence of
//! a libFuzzer abort) that no input drives a panic, unwrap, index-out-of-bounds or
//! arithmetic overflow. Build without Z3: `cargo +nightly fuzz build parse_ir`.
#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if let Ok(text) = std::str::from_utf8(data) {
        // The parser must return Result, never panic, for any input.
        let _ = sc_neurocore_engine::ir::parser::parse(text);
    }
});
