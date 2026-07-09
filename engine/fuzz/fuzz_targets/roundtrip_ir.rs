// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Source/config provenance header

//! Differential fuzz target: the SC IR printer and parser must round-trip.
//!
//! Rather than assert only that `parse` never panics (see `parse_ir`), this target uses
//! `parse` to mint valid graphs from fuzzed input and then checks the stronger property
//! that `parse` and `printer::print` are inverse on a parsed graph:
//!
//! * printing a parsed graph must produce text the parser accepts (a printer that emits
//!   un-parseable IR is a bug), and
//! * re-parsing that text must reproduce the same graph (`parse(print(g)) == g`), so no
//!   information is lost or reordered across the round trip.
//!
//! Either failure aborts under libFuzzer. Build without Z3:
//! `cargo +nightly fuzz build roundtrip_ir`.
#![no_main]

use libfuzzer_sys::fuzz_target;
use sc_neurocore_engine::ir::{parser, printer};

fuzz_target!(|data: &[u8]| {
    let Ok(text) = std::str::from_utf8(data) else {
        return;
    };
    let Ok(graph) = parser::parse(text) else {
        return;
    };
    // The printer's own output must always re-parse — a panic here is a real print/parse
    // asymmetry, not a fuzzing artefact, because `graph` is a valid parsed graph.
    let printed = printer::print(&graph);
    let reparsed = parser::parse(&printed)
        .unwrap_or_else(|e| panic!("printer output failed to re-parse: {e:?}\n---\n{printed}"));
    assert_eq!(graph, reparsed, "parse . print changed the graph");
});
