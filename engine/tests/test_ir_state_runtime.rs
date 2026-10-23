// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Clocked register traces

//! Independent clock-by-clock state traces through public native graph and both RTL paths.
#[path = "common/ir_stream.rs"]
mod ir_stream;
use sc_neurocore_engine::ir::{builder::ScGraphBuilder, graph::*, word::*};

#[test]
fn simultaneous_swap_enable_hold_and_asynchronous_reset() {
    let ty = ScType::FixedPoint { width: 16, frac: 8 };
    let mut b = ScGraphBuilder::new("atomic_swap");
    let enabled = b.input("enabled", ScType::Bool);
    let ai = b.constant(ScConst::F64(-1.25), ty.clone());
    let bi = b.constant(ScConst::F64(0.375), ty.clone());
    let a = b.state(ai, ty.clone());
    let c = b.state(bi, ty);
    b.store(a, c, Some(enabled));
    b.store(c, a, Some(enabled));
    b.output("a", a);
    b.output("b", c);
    let mut bench = String::from("module tb; reg clk=0; reg rst_n=1; reg enabled=0; wire [15:0] a,b; atomic_swap dut(.*); initial begin #1; rst_n=0; #1; $display(\"OBS %h %h\",a,b); #1; rst_n=1;\n");
    let (mut a, mut c) = (-320i32, 96i32);
    let mut expected = vec![format!("OBS {:04x} {:04x}", a as u16, c as u16)];
    for (i, enabled) in [true, true, false, false, true, false, true, true]
        .into_iter()
        .enumerate()
    {
        bench.push_str(&format!(
            "enabled={}; #2; clk=1; #1; $display(\"OBS %h %h\",a,b); #2; clk=0;\n",
            u8::from(enabled)
        ));
        if enabled {
            (a, c) = (c, a);
        }
        expected.push(format!("OBS {:04x} {:04x}", a as u16, c as u16));
        if i == 4 {
            // Assert reset with the clock low and enable false, then pulse it again.
            bench.push_str("enabled=0; #1; rst_n=0; #1; $display(\"OBS %h %h\",a,b); #1; rst_n=1; #1; rst_n=0; #1; $display(\"OBS %h %h\",a,b); #1; rst_n=1;\n");
            (a, c) = (-320, 96);
            expected.extend(vec![format!("OBS {:04x} {:04x}", a as u16, c as u16); 2]);
        }
    }
    bench.push_str("$finish; end endmodule");
    ir_stream::run("state_atomic_swap", &b.build(), &bench, &expected);
}

#[test]
fn nonlinear_signed_recurrence_and_counter_match_integer_reference() {
    let ty = ScType::SInt { width: 8 };
    let mut b = ScGraphBuilder::new("recurrence");
    let enabled = b.input("enabled", ScType::Bool);
    let initial = b.constant(ScConst::I64(-7), ty.clone());
    let three = b.constant(ScConst::I64(3), ty.clone());
    let ci = b.constant(ScConst::U64(254), ScType::UInt { width: 8 });
    let one = b.constant(ScConst::U64(1), ScType::UInt { width: 8 });
    let x = b.state(initial, ty.clone());
    let count = b.state(ci, ScType::UInt { width: 8 });
    let square = b.word(WordOp::Binary {
        kind: WordBinaryOp::Mul,
        lhs: x,
        rhs: x,
        ty: ty.clone(),
    });
    let next = b.word(WordOp::Binary {
        kind: WordBinaryOp::Add,
        lhs: square,
        rhs: three,
        ty,
    });
    let increment = b.word(WordOp::Binary {
        kind: WordBinaryOp::Add,
        lhs: count,
        rhs: one,
        ty: ScType::UInt { width: 8 },
    });
    b.store(x, next, Some(enabled));
    b.store(count, increment, None);
    b.output("x", x);
    b.output("count", count);
    let mut bench = String::from("module tb; reg clk=0; reg rst_n=1; reg enabled=0; wire [7:0] x,count; recurrence dut(.*); initial begin #1; rst_n=0; #1; $display(\"OBS %h %h\",x,count); #1; rst_n=1;\n");
    let (mut x, mut count) = (-7i32, 254u8);
    let mut expected = vec![format!("OBS {:02x} {count:02x}", x as u8)];
    for i in 0..40 {
        let enabled = i % 5 != 2 && i % 5 != 3;
        if enabled {
            x = (x * x + 3) as i8 as i32;
        }
        count = count.wrapping_add(1);
        expected.push(format!("OBS {:02x} {count:02x}", x as u8));
        bench.push_str(&format!(
            "enabled={}; #2; clk=1; #1; $display(\"OBS %h %h\",x,count); #2; clk=0;\n",
            u8::from(enabled)
        ));
    }
    bench.push_str("$finish; end endmodule");
    ir_stream::run("state_nonlinear_counter", &b.build(), &bench, &expected);
}

#[test]
fn signed_64_bit_wrap_boolean_state_and_reset_priority() {
    let mut b = ScGraphBuilder::new("wide_state");
    let enabled = b.input("enabled", ScType::Bool);
    let ty = ScType::SInt { width: 64 };
    let initial = b.constant(ScConst::I64(i64::MAX), ty.clone());
    let one = b.constant(ScConst::I64(1), ty.clone());
    let zero = b.constant(ScConst::I64(0), ty.clone());
    let false_word = b.constant(ScConst::U64(0), ScType::Bool);
    let state = b.state(initial, ty.clone());
    let flag = b.state(false_word, ScType::Bool);
    let next = b.word(WordOp::Binary {
        kind: WordBinaryOp::Add,
        lhs: state,
        rhs: one,
        ty,
    });
    let negative = b.word(WordOp::Compare {
        predicate: WordCompareOp::Lt,
        lhs: state,
        rhs: zero,
    });
    b.store(state, next, Some(enabled));
    b.store(flag, negative, None);
    b.output("word", state);
    b.output("flag", flag);
    let mut bench = String::from("module tb; reg clk=0,rst_n=1,enabled=0; wire [63:0] word; wire flag; wide_state dut(.*); initial begin #1; rst_n=0; #1; $display(\"OBS %h %b\",word,flag); #1; rst_n=1;\n");
    let mut word = i64::MAX;
    let mut expected = vec![format!("OBS {:016x} 0", word as u64)];
    for advance in [true, false, true, true, false, true] {
        let flag = word < 0;
        if advance {
            word = word.wrapping_add(1);
        }
        expected.push(format!("OBS {:016x} {}", word as u64, u8::from(flag)));
        bench.push_str(&format!(
            "enabled={}; #2; clk=1; #1; $display(\"OBS %h %b\",word,flag); #2; clk=0;\n",
            u8::from(advance)
        ));
    }
    bench.push_str("enabled=1; #1; clk=1; #1; rst_n=0; #1; $display(\"OBS %h %b\",word,flag); #1; clk=0; #1; clk=1; #1; $display(\"OBS %h %b\",word,flag); $finish; end endmodule");
    expected.extend(vec!["OBS 7fffffffffffffff 0".to_string(); 2]);
    ir_stream::run("state_signed64_boolean", &b.build(), &bench, &expected);
}

#[test]
fn raw_rate_reset_words_preserve_scale_after_persistence() {
    let mut b = ScGraphBuilder::new("raw_rate_state");
    let ai = b.constant(ScConst::I64(1), ScType::Rate);
    let bi = b.constant(ScConst::U64(257), ScType::Rate);
    let a = b.state(ai, ScType::Rate);
    let other = b.state(bi, ScType::Rate);
    b.store(a, other, None);
    b.store(other, a, None);
    b.output("a", a);
    b.output("b", other);
    let bench="module tb; reg clk=0,rst_n=1; wire [15:0] a,b; raw_rate_state dut(.*); initial begin #1; rst_n=0; #1; $display(\"OBS %h %h\",a,b); #1; rst_n=1; #2; clk=1; #1; $display(\"OBS %h %h\",a,b); #2; clk=0; #2; clk=1; #1; $display(\"OBS %h %h\",a,b); $finish; end endmodule";
    ir_stream::run(
        "state_raw_rate_reset",
        &b.build(),
        bench,
        &[
            "OBS 0001 0101".into(),
            "OBS 0101 0001".into(),
            "OBS 0001 0101".into(),
        ],
    );
}
