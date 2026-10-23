// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Scalar word persistence and parser diagnostics

//! Public text parsing preserves operation attributes and rejects malformed syntax.

use sc_neurocore_engine::ir::{parser, printer, verify};

/// Persist literal bounds, signed types and rounding intent without guessing operand types.
#[test]
fn scalar_word_text_round_trip_preserves_every_attribute() {
    let text="sc.graph @words {\n  %0 = sc.input \"x\" : i16\n  %1 = sc.word resize %0 : i32\n  %2 = sc.word mul %1, %1 : i32\n  %3 = sc.word round %2, 8, half_even : i32\n  %4 = sc.word clamp %3, -32768, 32767 : i32\n  %5 = sc.word resize %4 : i16\n  %6 = sc.word bitcast %5 : fixed<16,8>\n  %7 = sc.word cmp_lt %0, %5 : bool\n  %8 = sc.word select %7, %0, %5 : i16\n  %9 = sc.word neg %8 : i16\n  sc.output \"value\" %9\n}\n";
    let graph = parser::parse(text).unwrap();
    verify::verify(&graph).unwrap();
    assert_eq!(printer::print(&graph), text);
    assert_eq!(parser::parse(&printer::print(&graph)).unwrap(), graph);
}

/// Extra operands, unknown spellings and malformed result/rounding attributes fail with locations.
#[test]
fn scalar_word_parser_refuses_malformed_instructions() {
    for body in [
        "round %0, 1, floor, extra : i16",
        "round bad, 1, floor : i16",
        "clamp %0, 0, 9223372036854775808 : i16",
        "clamp bad, 0, 1 : i16",
        "select %0, %0 : i16",
        "select bad, %0, %0 : i16",
        "select %0, bad, %0 : i16",
        "select %0, %0, bad : i16",
        "neg %0, %0 : i16",
        "not bad : i16",
        "resize bad : i16",
        "cmp_lt %0 : bool",
        "cmp_bad %0, %0 : bool",
        "cmp_lt bad, %0 : bool",
        "cmp_lt %0, bad : bool",
        "add bad, %0 : i16",
        "add %0, bad : i16",
        "add %0, %0",
        "neg : i16",
        "add %0, %0, %0 : i16",
        "round %0, -1, floor : i16",
        "round %0, 1, nearest : i16",
        "cmp_lt %0, %0 : i16",
        "clamp %0, nope, 1 : i16",
        "mystery %0, %0 : i16",
        "resize %0 : invalid",
        "bitcast %0, %0 : i16",
    ] {
        let text = format!(
            "sc.graph @invalid {{\n  %0 = sc.input \"x\" : i16\n  %1 = sc.word {body}\n}}\n"
        );
        let error = parser::parse(&text).unwrap_err();
        assert_eq!(error.line, 3);
        assert!(!error.message.is_empty());
    }
}

/// Persisted values cannot wrap the next SSA identifier or smuggle a malformed assignment.
#[test]
fn scalar_word_parser_rejects_overflowing_ids_and_assignment_syntax() {
    for assignment in [
        "%4294967295 = sc.word neg %0 : i16",
        "bad = sc.word neg %0 : i16",
        "%1=sc.word neg %0 : i16",
    ] {
        let text =
            format!("sc.graph @invalid {{\n  %0 = sc.input \"x\" : i16\n  {assignment}\n}}\n");
        let error = parser::parse(&text).unwrap_err();
        assert_eq!(error.line, 3);
    }
}

/// Every simple declaration family rejects an exhausted parser ID without wrapping or panic.
#[test]
fn legacy_scalar_declarations_check_the_shared_id_allocator() {
    use sc_neurocore_engine::ir::{builder::ScGraphBuilder, graph::*};
    let mut b = ScGraphBuilder::new("parser_ids");
    let bits = b.input("bits", ScType::UInt { width: 8 });
    let one = b.constant(ScConst::U64(1), ScType::UInt { width: 8 });
    let prob = b.input("prob", ScType::Rate);
    b.encode(prob, 3, 1);
    b.bitwise_and(bits, one);
    b.bitwise_xor(bits, one);
    let count = b.popcount(bits);
    b.decode(bits);
    b.scale(count, 0.5);
    b.offset(count, 1.0);
    b.div_const(bits, 3);
    b.reduce(bits, ReduceMode::Sum);
    b.graph_forward(bits, bits, 1, 1);
    b.softmax_attention(bits, bits, bits, 1);
    b.kuramoto_step(bits, bits, bits, 0.01);
    b.output("value", count);
    let graph = b.build();
    let text = printer::print(&graph);
    assert_eq!(parser::parse(&text).unwrap(), graph);
    let lines: Vec<_> = text.lines().collect();
    for (index, line) in lines
        .iter()
        .enumerate()
        .filter(|(_, line)| line.contains(" = sc."))
    {
        let (_, instruction) = line.split_once(" = ").unwrap();
        let mut changed = lines.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        changed[index] = format!("  %4294967295 = {instruction}");
        let error = parser::parse(&changed.join("\n")).unwrap_err();
        assert!(
            error.message.contains("value ID overflow"),
            "{line}: {error}"
        );
    }
    let text="sc.graph @exhausted_output {\n  %4294967294 = sc.input \"x\" : i8\n  sc.output \"value\" %4294967294\n}\n";
    let error = parser::parse(text).unwrap_err();
    assert_eq!(error.line, 3);
    assert!(error.message.contains("value ID overflow"));
}
