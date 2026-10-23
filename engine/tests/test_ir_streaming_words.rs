// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Word geometry and complete encoder-period contracts

//! Exercise full-width division, vector constants and the physical LFSR distribution.

#[path = "common/ir_stream.rs"]
mod ir_stream;
use ir_stream::run;
use sc_neurocore_engine::ir::{builder::ScGraphBuilder, graph::*};

/// Compute a wide quotient using u128 limb division, independently of restoring RTL.
fn reference_divide(word: &str, divisor: u64) -> u64 {
    let mut remainder = 0_u128;
    let mut quotient = 0_u64;
    for chunk in word.as_bytes().chunks(32) {
        let limb = chunk
            .iter()
            .fold(0_u128, |n, bit| (n << 1) | u128::from(*bit == b'1'));
        let numerator = (remainder << chunk.len()) | limb;
        quotient =
            quotient.wrapping_shl(chunk.len() as u32) | (numerator / u128::from(divisor)) as u64;
        remainder = numerator % u128::from(divisor);
    }
    quotient
}

/// Preserve unsigned integer division before narrowing, including operands wider than64bits.
#[test]
fn direct_integer_division_preserves_the_full_input_word() {
    for width in [1, 8, 63, 64, 65, 129] {
        for divisor in [1, 3, 65535, u64::MAX] {
            let mut b = ScGraphBuilder::new("word_division");
            let word = b.input("word", ScType::UInt { width });
            let quotient = b.div_const(word, divisor);
            b.output("quotient", quotient);
            let mut bench = format!("module tb; reg clk=0,rst_n=1; reg [{}:0] word=0; wire [63:0] quotient; word_division dut(.*); initial begin\n", width-1);
            let mut expected = Vec::new();
            for pattern in 0..4 {
                let word: String = (0..width)
                    .map(|i| match pattern {
                        0 => '0',
                        1 => '1',
                        _ if (i + pattern) % 3 == 0 => '1',
                        _ => '0',
                    })
                    .collect();
                expected.push(format!("OBS {}", reference_divide(&word, divisor)));
                bench.push_str(&format!(
                    "word={width}'b{word}; #1; $display(\"OBS %0d\",quotient);\n"
                ));
            }
            bench.push_str("$finish; end endmodule\n");
            run(
                &format!("division-{width}-{divisor}"),
                &b.build(),
                &bench,
                &expected,
            );
        }
    }
}

/// Count a non-byte-aligned vector constant and expose all its original packed bits.
#[test]
fn vector_constants_and_signed_bits_reach_both_backends_exactly() {
    for values in [vec![0, 0, 0], vec![127, 0, 64], vec![127, 127, 127]] {
        let mut b = ScGraphBuilder::new("constant_counts");
        let source = b.constant(
            ScConst::I64Vec(values.clone()),
            ScType::Vec {
                element: Box::new(ScType::UInt { width: 7 }),
                count: 3,
            },
        );
        let count = b.popcount(source);
        let rate = b.decode(source);
        b.output("packed_word", source);
        b.output("count", count);
        b.output("rate", rate);
        let packed = values
            .iter()
            .enumerate()
            .map(|(i, v)| (*v as u64) << (i * 7))
            .sum::<u64>();
        let n = packed.count_ones();
        let bench = "module tb; wire [20:0] packed_word; wire [63:0] count; wire [15:0] rate; constant_counts dut(.clk(1'b0),.rst_n(1'b1),.packed_word(packed_word),.count(count),.rate(rate)); initial begin #1; $display(\"OBS %0d %0d %0d\",packed_word,count,rate); $finish; end endmodule";
        run(
            &format!("constant-vector-{packed}"),
            &b.build(),
            bench,
            &[format!("OBS {packed} {n} {}", u64::from(n) * 256 / 21)],
        );
    }
    let mut b = ScGraphBuilder::new("signed_bits");
    let word = b.input("word", ScType::SInt { width: 8 });
    let count = b.popcount(word);
    let rate = b.decode(word);
    let quotient = b.div_const(word, 3);
    b.output("count", count);
    b.output("rate", rate);
    b.output("quotient", quotient);
    let bench = "module tb; reg [7:0] word=255; wire [63:0] count,quotient; wire [15:0] rate; signed_bits dut(.clk(1'b0),.rst_n(1'b1),.word(word),.count(count),.rate(rate),.quotient(quotient)); initial begin #1; $display(\"OBS %0d %0d %0d\",count,rate,quotient); word=128; #1; $display(\"OBS %0d %0d %0d\",count,rate,quotient); $finish; end endmodule";
    run(
        "signed-packed-bits",
        &b.build(),
        bench,
        &["OBS 8 256 85".into(), "OBS 1 32 42".into()],
    );
}

/// Establish the strict comparator's exact period law, including positive probabilities with T=0.
#[test]
fn encoder_full_period_has_max_threshold_minus_one_hits() {
    let raw = [-65536_i64, 0, 1, 2, 32768, 65536, 131072];
    let mut b = ScGraphBuilder::new("period_encoder");
    let mut expected = String::from("OBS");
    let mut declarations = String::new();
    let mut increments = String::new();
    let mut formats = String::new();
    let mut args = String::new();
    for (i, raw) in raw.into_iter().enumerate() {
        let value = b.constant(
            ScConst::I64(raw),
            ScType::FixedPoint {
                width: 32,
                frac: 16,
            },
        );
        let bit = b.encode(value, 65535, 0xACE1);
        b.output(format!("bit{i}"), bit);
        let threshold = (i128::from(raw).clamp(0, 65536) * 65535 / 65536) as u64;
        expected.push_str(&format!(" {}", threshold.saturating_sub(1)));
        declarations.push_str(&format!("wire bit{i}; integer hits{i}=0; "));
        increments.push_str(&format!("hits{i}=hits{i}+bit{i}; "));
        formats.push_str(" %0d");
        args.push_str(&format!(",hits{i}"));
    }
    let bench = format!("module tb; reg clk=0,rst_n=1; {declarations} period_encoder dut(.*); initial begin #1; rst_n=0; #1; rst_n=1; repeat(65535) begin #2; clk=1; #1; {increments} #2; clk=0; end $display(\"OBS{formats}\"{args}); $finish; end endmodule");
    run("encoder-full-period", &b.build(), &bench, &[expected]);
}

/// Boolean observations are combinational; an empty graph retains a valid empty output tuple.
#[test]
fn boolean_and_empty_output_graphs_have_real_circuit_semantics() {
    let mut b = ScGraphBuilder::new("boolean_count");
    let value = b.input("value", ScType::Bool);
    let count = b.popcount(value);
    let rate = b.decode(value);
    b.output("count", count);
    b.output("rate", rate);
    let bench = "module tb; reg value=0; wire [63:0] count; wire [15:0] rate; boolean_count dut(.clk(1'b0),.rst_n(1'b1),.value(value),.count(count),.rate(rate)); initial begin #1; $display(\"OBS %0d %0d\",count,rate); value=1; #1; $display(\"OBS %0d %0d\",count,rate); $finish; end endmodule";
    run(
        "boolean-count",
        &b.build(),
        bench,
        &["OBS 0 0".into(), "OBS 1 256".into()],
    );
    let g = ScGraphBuilder::new("empty_outputs").build();
    run("empty-outputs",&g,"module tb; empty_outputs dut(.clk(1'b0),.rst_n(1'b1)); initial begin #1; $finish; end endmodule",&[]);
}

/// A maximum-length window remains a valid circuit; this exercises its prefix, not its completion.
#[test]
fn maximum_window_prefix_preserves_full_width_state_and_scaling() {
    let mut b = ScGraphBuilder::new("maximum_window");
    let bits = b.input("bits", ScType::Bitstream { length: usize::MAX });
    let count = b.popcount(bits);
    let rate = b.decode(bits);
    b.output("count", count);
    b.output("rate", rate);
    let bench="module tb; reg clk=0,rst_n=1,bits=1; wire [63:0] count; wire [15:0] rate; maximum_window dut(.*); initial begin #1; rst_n=0; #1; rst_n=1; repeat(8) begin #2; clk=1; #1; $display(\"OBS %0d %0d\",count,rate); #2; clk=0; end $finish; end endmodule";
    run(
        "maximum-window-prefix",
        &b.build(),
        bench,
        &vec!["OBS 0 0".into(); 8],
    );
}

/// Derived count and quotient words retain their integer types when consumed by other bit operations.
#[test]
fn computed_words_keep_their_u64_bit_observation_type() {
    let mut b = ScGraphBuilder::new("computed_words");
    let word = b.input("word", ScType::UInt { width: 8 });
    b.output("input_word", word);
    let count = b.popcount(word);
    let half = b.div_const(count, 2);
    let rate = b.decode(half);
    b.output("rate", rate);
    let bench="module tb; reg [7:0] word=0; wire [7:0] input_word; wire [15:0] rate; computed_words dut(.clk(1'b0),.rst_n(1'b1),.word(word),.input_word(input_word),.rate(rate)); initial begin #1; $display(\"OBS %0d %0d\",input_word,rate); word=255; #1; $display(\"OBS %0d %0d\",input_word,rate); word=63; #1; $display(\"OBS %0d %0d\",input_word,rate); $finish; end endmodule";
    run(
        "computed-word-type",
        &b.build(),
        bench,
        &["OBS 0 0".into(), "OBS 255 4".into(), "OBS 63 8".into()],
    );
}

/// Constant division observes physical signed words and zero-extends instead of sign-extending.
#[test]
fn division_by_one_zero_extends_signed_constants() {
    for width in [1, 8, 16, 63, 64] {
        let minimum = i64::MIN >> (64 - width);
        let words = if width == 1 {
            vec![0, -1]
        } else {
            vec![-1, -2, minimum]
        };
        for raw in words {
            let mut b = ScGraphBuilder::new("signed_identity");
            let value = b.constant(ScConst::I64(raw), ScType::SInt { width });
            let quotient = b.div_const(value, 1);
            b.output("quotient", quotient);
            let mask = u64::MAX >> (64 - width);
            let expected = (raw as u64) & mask;
            let bench = "module tb; wire [63:0] quotient; signed_identity dut(.clk(1'b0),.rst_n(1'b1),.quotient(quotient)); initial begin #1; $display(\"OBS %0d\",quotient); $finish; end endmodule";
            run(
                &format!("signed-identity-{width}-{raw}"),
                &b.build(),
                bench,
                &[format!("OBS {expected}")],
            );
        }
    }
}

/// A one-tick decode uses the same identity division while retaining reset and completion timing.
#[test]
fn one_tick_decode_commits_scaled_bits_after_each_clock() {
    let mut b = ScGraphBuilder::new("one_tick_decode");
    let bits = b.input("bits", ScType::Bitstream { length: 1 });
    let rate = b.decode(bits);
    b.output("rate", rate);
    let bench = "module tb; reg clk=0,rst_n=1,bits=0; wire [15:0] rate; one_tick_decode dut(.*); initial begin #1; rst_n=0; #1; $display(\"OBS %0d\",rate); rst_n=1; bits=1; #1; $display(\"OBS %0d\",rate); #1; clk=1; #1; $display(\"OBS %0d\",rate); clk=0; bits=0; #1; $display(\"OBS %0d\",rate); clk=1; #1; $display(\"OBS %0d\",rate); clk=0; bits=1; #1; clk=1; #1; $display(\"OBS %0d\",rate); #1; rst_n=0; #1; $display(\"OBS %0d\",rate); $finish; end endmodule";
    run(
        "one-tick-decode",
        &b.build(),
        bench,
        &[0, 0, 256, 256, 0, 256, 0].map(|value| format!("OBS {value}")),
    );
}
