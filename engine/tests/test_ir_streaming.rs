// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Public bitstream counting and decoding through real RTL

//! Compare complete public output traces with independent integer observations.

#[path = "common/ir_stream.rs"]
mod ir_stream;
use ir_stream::run;
use sc_neurocore_engine::ir::{builder::ScGraphBuilder, emit_mlir, emit_sv, graph::*, verify};

/// Count every accepted clock sample, hold results, and discard partial windows on async reset.
#[test]
fn temporal_windows_match_integer_reference_including_midclock_resets() {
    for window in [1, 3, 17, 64, 1024] {
        for lanes in [1, 3] {
            let mut b = ScGraphBuilder::new("stream_counts");
            let scalar = ScType::Bitstream { length: window };
            let ty = if lanes == 1 {
                scalar
            } else {
                ScType::Vec {
                    element: Box::new(scalar),
                    count: lanes,
                }
            };
            let input = b.input("bits", ty);
            let count = b.popcount(input);
            let rate = b.decode(input);
            let integer = b.div_const(count, window as u64 * lanes as u64);
            b.output("count", count);
            b.output("rate", rate);
            b.output("quotient", integer);
            let mut bench = format!("module tb; reg clk=0; reg rst_n=1; reg [{}:0] bits=0; wire [63:0] count,quotient; wire [15:0] rate; stream_counts dut(.*); initial begin #1; rst_n=0; #1; $display(\"OBS %0d %0d %0d\",count,rate,quotient); rst_n=1;\n", lanes-1);
            let mut expected = vec!["OBS 0 0 0".into()];
            let (mut phase, mut acc, mut held) = (0, 0_u64, 0_u64);
            for i in 0..window * 3 + 7 {
                if i == window + 1 {
                    bench.push_str("#1; rst_n=0; #1; $display(\"OBS %0d %0d %0d\",count,rate,quotient); rst_n=1;\n");
                    expected.push("OBS 0 0 0".into());
                    (phase, acc, held) = (0, 0, 0);
                }
                let word = if i < window {
                    0
                } else if i < window * 2 {
                    (1_u64 << lanes) - 1
                } else {
                    ((i * 13 + i / 3) & ((1 << lanes) - 1)) as u64
                };
                acc += u64::from(word.count_ones());
                phase += 1;
                if phase == window {
                    held = acc;
                    phase = 0;
                    acc = 0;
                }
                let opportunities = window as u64 * lanes as u64;
                expected.push(format!(
                    "OBS {held} {} {}",
                    held * 256 / opportunities,
                    held / opportunities
                ));
                bench.push_str(&format!("bits={lanes}'d{word}; #2; clk=1; #1; $display(\"OBS %0d %0d %0d\",count,rate,quotient); #2; clk=0;\n"));
            }
            bench.push_str("$finish; end endmodule\n");
            run(
                &format!("temporal-{window}-{lanes}"),
                &b.build(),
                &bench,
                &expected,
            );
        }
    }
}

/// Packed words use their actual width, including buses wider than a machine word.
#[test]
fn packed_bitwise_popcount_decode_and_integer_division_preserve_widths() {
    for width in [1, 8, 17, 64, 65, 129] {
        let mut b = ScGraphBuilder::new("packed_counts");
        let a = b.input("a", ScType::UInt { width });
        let z = b.input("b", ScType::UInt { width });
        let intersection = b.bitwise_and(a, z);
        let difference = b.bitwise_xor(intersection, a);
        let count = b.popcount(difference);
        let rate = b.decode(difference);
        let integer = b.div_const(count, 3);
        b.output("count", count);
        b.output("rate", rate);
        b.output("quotient", integer);
        let mut bench = format!("module tb; reg clk=0,rst_n=1; reg [{}:0] a,b; wire [63:0] count,quotient; wire [15:0] rate; packed_counts dut(.*); initial begin\n", width-1);
        let mut expected = Vec::new();
        for pattern in 0..10 {
            let aa: String = (0..width)
                .map(|i| {
                    if pattern == 0 || (pattern > 2 && (i + pattern) % 3 == 0) {
                        '0'
                    } else {
                        '1'
                    }
                })
                .collect();
            let bb: String = (0..width)
                .map(|i| {
                    if pattern == 2 || (pattern > 2 && (i * 7 + pattern) % 5 == 0) {
                        '1'
                    } else {
                        '0'
                    }
                })
                .collect();
            let n = aa
                .bytes()
                .zip(bb.bytes())
                .filter(|&(a, b)| a == b'1' && b == b'0')
                .count();
            expected.push(format!("OBS {n} {} {}", n * 256 / width as usize, n / 3));
            bench.push_str(&format!("a={width}'b{aa}; b={width}'b{bb}; #1; $display(\"OBS %0d %0d %0d\",count,rate,quotient);\n"));
        }
        bench.push_str("$finish; end endmodule\n");
        run(&format!("packed-{width}"), &b.build(), &bench, &expected);
    }
}

/// Validate shape/epoch compatibility and prevent division by zero before either backend.
#[test]
fn malformed_stream_geometry_and_windows_are_rejected() {
    for ty in [
        ScType::Bitstream { length: 0 },
        ScType::UInt { width: 0 },
        ScType::Vec {
            element: Box::new(ScType::Bool),
            count: 0,
        },
        ScType::Rate,
    ] {
        let mut b = ScGraphBuilder::new("bad_shape");
        let input = b.input("bits", ty);
        let rate = b.decode(input);
        b.output("rate", rate);
        let g = b.build();
        assert!(verify::verify(&g).is_err());
        assert!(emit_sv::emit(&g).is_err());
        assert!(emit_mlir::emit(&g).is_err());
    }
    let mut b = ScGraphBuilder::new("bad_windows");
    let a = b.input("a", ScType::Bitstream { length: 3 });
    let c = b.input("b", ScType::Bitstream { length: 4 });
    b.bitwise_and(a, c);
    assert!(verify::verify(&b.build()).is_err());
    let mut b = ScGraphBuilder::new("bad_divisor");
    let a = b.input("a", ScType::UInt { width: 64 });
    b.div_const(a, 0);
    assert!(verify::verify(&b.build()).is_err());
}

/// Check reset-prefix latency and dynamic probabilities through two real encoders.
#[test]
fn encoder_decode_and_reencode_match_registered_bit_reference() {
    let mut b = ScGraphBuilder::new("encoder_counts");
    let x = b.input("x", ScType::Rate);
    let w = b.input("w", ScType::Rate);
    let xb = b.encode(x, 17, 0xACE1);
    let wb = b.encode(w, 17, 0xBEEF);
    let product = b.bitwise_and(xb, wb);
    let count = b.popcount(product);
    let rate = b.decode(product);
    let reencoded = b.encode(rate, 17, 0x1234);
    b.output("count", count);
    b.output("rate", rate);
    b.output("reencoded", reencoded);
    let mut bench = String::from("module tb; reg clk=0,rst_n=1; reg [15:0] x=0,w=0; wire [63:0] count; wire [15:0] rate; wire reencoded; encoder_counts dut(.*); initial begin #1; rst_n=0; #1; rst_n=1;\n");
    let (mut xs, mut ws, mut rs) = (0xACE1_u16, 0xBEEF_u16, 0x1234_u16);
    let (mut oldx, mut oldw, mut phase, mut acc, mut held) = (false, false, 0, 0_u64, 0_u64);
    let mut expected = Vec::new();
    for i in 0..103 {
        if i == 47 {
            bench.push_str(
                "#1; rst_n=0; #1; $display(\"OBS %0d %0d %0d\",count,rate,reencoded); rst_n=1;\n",
            );
            expected.push("OBS 0 0 0".into());
            (xs, ws, rs) = (0xACE1, 0xBEEF, 0x1234);
            (oldx, oldw, phase, acc, held) = (false, false, 0, 0, 0);
        }
        let x: i32 = [-256, 0, 1, 128, 256, 512][i % 6];
        let w: i32 = [256, 192, 128, 64, 0, -1][(i / 7) % 6];
        let threshold = |raw: i32| (raw.clamp(0, 256) as u64 * 65535 / 256) as u16;
        let reencoded = rs < threshold((held * 256 / 17) as i32);
        acc += u64::from(oldx && oldw);
        phase += 1;
        if phase == 17 {
            held = acc;
            phase = 0;
            acc = 0;
        }
        (oldx, oldw) = (xs < threshold(x), ws < threshold(w));
        let next = |s: u16| s.wrapping_shl(1) | ((s >> 15 ^ s >> 13 ^ s >> 12 ^ s >> 10) & 1);
        (xs, ws, rs) = (next(xs), next(ws), next(rs));
        expected.push(format!(
            "OBS {held} {} {}",
            held * 256 / 17,
            u8::from(reencoded)
        ));
        bench.push_str(&format!("x=16'd{}; w=16'd{}; #2; clk=1; #1; $display(\"OBS %0d %0d %0d\",count,rate,reencoded); #2; clk=0;\n",x as u16,w as u16));
    }
    bench.push_str("$finish; end endmodule\n");
    run("encoder-counts", &b.build(), &bench, &expected);
}

/// Preserve each encoder input's binary point and signed full-width clipping in CIRCT and SV.
#[test]
fn encoder_formats_match_reference_at_signed_boundaries() {
    for (width, frac) in [
        (1, 0),
        (2, 1),
        (8, 3),
        (16, 8),
        (24, 12),
        (32, 24),
        (64, 63),
        (64, 0),
    ] {
        let mut b = ScGraphBuilder::new("format_encoder");
        let x = b.input("x", ScType::FixedPoint { width, frac });
        let bits = b.encode(x, 3, 0xACE1);
        b.output("bits", bits);
        let mut bench = format!("module tb; reg clk=0,rst_n=1; reg [{}:0] x=0; wire bits; format_encoder dut(.*); initial begin #1; rst_n=0; #1; rst_n=1;\n",width-1);
        let min = -(1_i128 << (width - 1));
        let max = (1_i128 << (width - 1)) - 1;
        let boundary = (1_i128 << frac).min(max);
        let cases = [min, -1, 0, 1_i128.min(max), boundary, max];
        let mut state = 0xACE1_u16;
        let mut expected = Vec::new();
        for i in 0..48 {
            let raw = cases[i % cases.len()];
            let threshold = (((raw.max(0) as u128) * 65535) >> frac).min(65535) as u16;
            expected.push(format!("OBS {}", u8::from(state < threshold)));
            state = state.wrapping_shl(1)
                | ((state >> 15 ^ state >> 13 ^ state >> 12 ^ state >> 10) & 1);
            let word = (raw & ((1_i128 << width) - 1)) as u64;
            bench.push_str(&format!(
                "x={width}'d{word}; #2; clk=1; #1; $display(\"OBS %0d\",bits); #2; clk=0;\n"
            ));
        }
        bench.push_str("$finish; end endmodule\n");
        run(
            &format!("encoder-format-{width}-{frac}"),
            &b.build(),
            &bench,
            &expected,
        );
    }
}
