// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Public bitstream counting and decoding through real RTL

//! Shared public graph and real RTL checks for primitive compiler integration tests.

#[path = "ir_rtl.rs"]
mod ir_rtl;
use sc_neurocore_engine::ir::{emit_mlir, emit_sv, graph::ScGraph, parser, printer, verify};
use std::{fs, path::Path};

/// Exercise public graph round-tripping and real RTL, retaining CIRCT inputs on request.
pub fn run(label: &str, graph: &ScGraph, bench: &str, expected: &[String]) {
    verify::verify(graph).unwrap();
    let text = printer::print(graph);
    let restored = parser::parse(&text).unwrap();
    assert_eq!(&restored, graph);
    let sv = emit_sv::emit(&restored).unwrap();
    assert_eq!(sv, emit_sv::emit(graph).unwrap());
    let mlir = emit_mlir::emit(&restored).unwrap();
    if let Some(root) = std::env::var_os("SCN_IR_CIRCT_RECEIPT_DIR") {
        let dir = Path::new(&root).join(label);
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("graph.scir"), text).unwrap();
        fs::write(dir.join("graph.mlir"), mlir).unwrap();
        fs::write(dir.join("tb.sv"), bench).unwrap();
        fs::write(dir.join("expected.txt"), expected.join("\n") + "\n").unwrap();
    }
    let trace = ir_rtl::simulate(
        label,
        &sv,
        bench,
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("../hdl"),
    );
    let actual: Vec<_> = trace.lines().filter(|s| s.starts_with("OBS ")).collect();
    assert_eq!(actual, expected, "{label}");
}
