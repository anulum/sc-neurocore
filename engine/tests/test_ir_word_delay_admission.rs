// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Pipeline word register admission

//! Public graph/parser diagnostics reject ambiguous register geometry and producers.

use sc_neurocore_engine::ir::{
    builder::ScGraphBuilder, emit_mlir, emit_sv, graph::*, parser, verify,
};

fn refuse(graph: ScGraph, diagnostic: &str) {
    let errors = verify::verify(&graph)
        .unwrap_err()
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join("\n");
    assert!(errors.contains(diagnostic), "{errors}");
    assert!(emit_sv::emit(&graph).unwrap_err().contains(diagnostic));
    assert!(emit_mlir::emit(&graph).unwrap_err().contains(diagnostic));
}

#[test]
fn clocked_words_require_identical_explicit_integer_types() {
    for (input, output, diagnostic) in [
        (
            ScType::SInt { width: 128 },
            ScType::UInt { width: 128 },
            "requires u128",
        ),
        (
            ScType::UInt { width: 128 },
            ScType::UInt { width: 64 },
            "requires u64",
        ),
        (ScType::Bool, ScType::Bool, "raw arithmetic"),
        (ScType::Rate, ScType::Rate, "raw arithmetic"),
        (
            ScType::FixedPoint { width: 16, frac: 8 },
            ScType::FixedPoint { width: 16, frac: 8 },
            "raw arithmetic",
        ),
        (
            ScType::Bitstream { length: 128 },
            ScType::Bitstream { length: 128 },
            "positive scalar",
        ),
        (
            ScType::UInt { width: 0 },
            ScType::UInt { width: 0 },
            "physical word width",
        ),
    ] {
        let mut b = ScGraphBuilder::new("invalid_delay");
        let input = b.input("sample", input);
        let result = b.word_delay(input, output);
        b.output("result", result);
        refuse(b.build(), diagnostic);
    }
}

#[test]
fn register_text_has_one_producer_and_no_ignored_timing_attributes() {
    for text in [
        "sc.graph @invalid_delay {\n%0 = sc.input \"sample\" : i128\n%1 = sc.word delay %0, enable=%0 : i128\n}",
        "sc.graph @invalid_delay {\n%0 = sc.input \"sample\" : i128\n%1 = sc.word delay %0, 2 : i128\n}",
        "sc.graph @invalid_delay {\n%0 = sc.input \"sample\" : i128\n%1 = sc.word delay %0 : i128, latency=2\n}",
    ] {
        assert!(parser::parse(text).is_err(), "{text}");
    }
    let mut b = ScGraphBuilder::new("undefined_delay");
    b.word_delay(ValueId(42), ScType::UInt { width: 131 });
    refuse(b.build(), "not defined before use");

    let mut b = ScGraphBuilder::new("terminator_delay");
    let input = b.input("sample", ScType::UInt { width: 128 });
    let output = b.output("sample_out", input);
    b.word_delay(output, ScType::UInt { width: 128 });
    refuse(b.build(), "not defined before use");
}
