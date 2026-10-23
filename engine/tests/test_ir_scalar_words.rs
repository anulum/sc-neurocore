// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Scalar integer circuits against independent arithmetic

//! Exercise public graph construction, text round-trip and emitted numerical RTL.

#[path = "common/ir_stream.rs"]
mod ir_stream;
use sc_neurocore_engine::ir::{builder::ScGraphBuilder, graph::*, word::*};

fn mask(w: u32) -> u128 {
    u128::MAX >> (128 - w)
}
fn signed(x: u128, w: u32) -> i128 {
    ((x << (128 - w)) as i128) >> (128 - w)
}
fn binary(op: WordBinaryOp, a: u128, b: u128, w: u32, is_signed: bool) -> u128 {
    use WordBinaryOp::*;
    let result = match op {
        Add => a.wrapping_add(b),
        Sub => a.wrapping_sub(b),
        Mul => a.wrapping_mul(b),
        Div | Rem if b == 0 => 0,
        Div if is_signed => signed(a, w).checked_div(signed(b, w)).unwrap_or(i128::MIN) as u128,
        Rem if is_signed => signed(a, w).checked_rem(signed(b, w)).unwrap_or(0) as u128,
        Div => a / b,
        Rem => a % b,
        And => a & b,
        Or => a | b,
        Xor => a ^ b,
        Shl if b >= u128::from(w) => 0,
        Shl => a << b,
        LShr if b >= u128::from(w) => 0,
        LShr => a >> b,
        AShr if b >= u128::from(w) => {
            if signed(a, w) < 0 {
                u128::MAX
            } else {
                0
            }
        }
        AShr => (signed(a, w) >> b) as u128,
    };
    result & mask(w)
}
fn compare(op: WordCompareOp, a: u128, b: u128, w: u32, is_signed: bool) -> u128 {
    let order = if is_signed {
        signed(a, w).cmp(&signed(b, w))
    } else {
        a.cmp(&b)
    };
    u128::from(match op {
        WordCompareOp::Eq => order.is_eq(),
        WordCompareOp::Ne => !order.is_eq(),
        WordCompareOp::Lt => order.is_lt(),
        WordCompareOp::Le => !order.is_gt(),
        WordCompareOp::Gt => order.is_gt(),
        WordCompareOp::Ge => !order.is_lt(),
    })
}
fn samples(w: u32) -> Vec<(u128, u128)> {
    let m = mask(w);
    let sign = 1u128 << (w - 1);
    let mut values = vec![
        (0, 0),
        (1, 0),
        (m, 0),
        (0, 1),
        (m, 1),
        (sign, 1),
        (sign, m),
        (m, m),
        (sign, sign),
        (m, 3 & m),
        (sign, 3 & m),
        (m, u128::from(w) & m),
        (m, u128::from(w + 1) & m),
        (m, sign),
        (1, m),
    ];
    let mut random = 0x9653_0f81_c442_6a79_13db_3c09_d3e6_5ba1u128;
    for _ in 0..17 {
        random ^= random << 23;
        random ^= random >> 17;
        random ^= random << 26;
        let a = random & m;
        random = random.rotate_left(37).wrapping_add(0x9e3779b97f4a7c15);
        values.push((a, random & m));
    }
    values
}

/// Exact signed and unsigned arithmetic, comparison, wide division and oversized shifts.
#[test]
fn scalar_arithmetic_matches_independent_integer_references() {
    for w in [1, 2, 8, 16, 32, 63, 64, 65, 127, 128] {
        for is_signed in [false, true] {
            let ty = if is_signed {
                ScType::SInt { width: w }
            } else {
                ScType::UInt { width: w }
            };
            let mut builder = ScGraphBuilder::new("scalar_arithmetic");
            let a = builder.input("a", ty.clone());
            let b = builder.input("b", ty.clone());
            let mut binary_ops = vec![
                WordBinaryOp::Add,
                WordBinaryOp::Sub,
                WordBinaryOp::Mul,
                WordBinaryOp::Div,
                WordBinaryOp::Rem,
                WordBinaryOp::And,
                WordBinaryOp::Or,
                WordBinaryOp::Xor,
                WordBinaryOp::Shl,
                WordBinaryOp::LShr,
            ];
            if is_signed {
                binary_ops.push(WordBinaryOp::AShr);
            }
            let compare_ops = [
                WordCompareOp::Eq,
                WordCompareOp::Ne,
                WordCompareOp::Lt,
                WordCompareOp::Le,
                WordCompareOp::Gt,
                WordCompareOp::Ge,
            ];
            let mut widths = Vec::new();
            for kind in &binary_ops {
                let id = builder.word(WordOp::Binary {
                    kind: *kind,
                    lhs: a,
                    rhs: b,
                    ty: ty.clone(),
                });
                builder.output(format!("o{}", widths.len()), id);
                widths.push(w);
            }
            for predicate in compare_ops {
                let id = builder.word(WordOp::Compare {
                    predicate,
                    lhs: a,
                    rhs: b,
                });
                builder.output(format!("o{}", widths.len()), id);
                widths.push(1);
            }
            for kind in [WordUnaryOp::Neg, WordUnaryOp::Not] {
                let id = builder.word(WordOp::Unary {
                    kind,
                    input: a,
                    ty: ty.clone(),
                });
                builder.output(format!("o{}", widths.len()), id);
                widths.push(w);
            }
            let declarations = widths
                .iter()
                .enumerate()
                .map(|(i, w)| format!("wire [{}:0] o{i}; ", w - 1))
                .collect::<String>();
            let formats = vec!["%h"; widths.len()].join(" ");
            let args = (0..widths.len())
                .map(|i| format!(",o{i}"))
                .collect::<String>();
            let mut bench=format!("module tb; reg clk=0,rst_n=1; reg [{}:0] a=0,b=0; {declarations} scalar_arithmetic dut(.*); initial begin\n",w-1);
            let mut expected = Vec::new();
            for (index, (x, y)) in samples(w).into_iter().enumerate() {
                let mut words = binary_ops
                    .iter()
                    .map(|op| binary(*op, x, y, w, is_signed))
                    .collect::<Vec<_>>();
                words.extend(compare_ops.map(|op| compare(op, x, y, w, is_signed)));
                words.extend([x.wrapping_neg() & mask(w), (!x) & mask(w)]);
                let values = words
                    .iter()
                    .zip(&widths)
                    .map(|(v, w)| format!("{v:0digits$x}", digits = w.div_ceil(4) as usize))
                    .collect::<Vec<_>>()
                    .join(" ");
                expected.push(format!("OBS {index} {values}"));
                bench.push_str(&format!(
                    "a={w}'h{x:x}; b={w}'h{y:x}; #1; $display(\"OBS {index} {formats}\"{args});\n"
                ));
            }
            bench.push_str("$finish; end endmodule");
            ir_stream::run(
                &format!("scalar-arithmetic-{w}-{is_signed}"),
                &builder.build(),
                &bench,
                &expected,
            );
        }
    }
}

fn rounded(x: u128, w: u32, shift: u32, mode: WordRounding, is_signed: bool) -> u128 {
    if shift == 0 {
        return x;
    }
    // Quotient/remainder of the absolute value provide an oracle independent of floor+residue RTL.
    let negative = is_signed && signed(x, w) < 0;
    let magnitude = if negative {
        signed(x, w).unsigned_abs()
    } else {
        x
    };
    let d = 1u128 << shift;
    let q = magnitude / d;
    let r = magnitude % d;
    let increment = match mode {
        WordRounding::Floor => negative && r != 0,
        WordRounding::TowardZero => false,
        WordRounding::HalfAway => r >= d / 2,
        WordRounding::HalfUp => r > d / 2 || (r == d / 2 && !negative),
        WordRounding::HalfEven => r > d / 2 || (r == d / 2 && q & 1 == 1),
    };
    let answer = q + u128::from(increment);
    (if negative {
        answer.wrapping_neg()
    } else {
        answer
    }) & mask(w)
}

/// Rounding ties and full-range inputs never overflow before rescaling; clamp is explicit.
#[test]
fn scalar_rounding_and_saturation_match_integer_references() {
    for w in [2, 8, 16, 64, 65, 128] {
        for is_signed in [false, true] {
            let ty = if is_signed {
                ScType::SInt { width: w }
            } else {
                ScType::UInt { width: w }
            };
            let mut b = ScGraphBuilder::new("scalar_rounding");
            let x = b.input("x", ty.clone());
            let mut specs = Vec::new();
            for shift in [0, 1, w / 2, w - 1] {
                for mode in [
                    WordRounding::Floor,
                    WordRounding::TowardZero,
                    WordRounding::HalfUp,
                    WordRounding::HalfAway,
                    WordRounding::HalfEven,
                ] {
                    let id = b.word(WordOp::Round {
                        input: x,
                        shift,
                        mode,
                        ty: ty.clone(),
                    });
                    b.output(format!("o{}", specs.len()), id);
                    specs.push((shift, mode));
                }
            }
            let (min, max) = if is_signed { (-2, 1) } else { (0, 3) };
            let clamp = b.word(WordOp::Clamp {
                input: x,
                min,
                max,
                ty,
            });
            b.output("clamped", clamp);
            let declarations = (0..specs.len())
                .map(|i| format!("wire [{}:0] o{i}; ", w - 1))
                .collect::<String>();
            let formats = vec!["%h"; specs.len() + 1].join(" ");
            let args = (0..specs.len())
                .map(|i| format!(",o{i}"))
                .collect::<String>()
                + ",clamped";
            let mut bench=format!("module tb; reg clk=0,rst_n=1; reg [{}:0] x=0; wire [{}:0] clamped; {declarations} scalar_rounding dut(.*); initial begin\n",w-1,w-1);
            let mut expected = Vec::new();
            let mut values: Vec<_> = (0..=31u128.min(mask(w))).collect();
            values.extend(samples(w).into_iter().map(|(x, _)| x));
            values.extend((0..=31u128.min(mask(w))).map(|x| x.wrapping_neg() & mask(w)));
            for (i, x) in values.into_iter().enumerate() {
                let mut words = specs
                    .iter()
                    .map(|(shift, mode)| rounded(x, w, *shift, *mode, is_signed))
                    .collect::<Vec<_>>();
                let clipped = if is_signed {
                    signed(x, w).clamp(i128::from(min), i128::from(max)) as u128
                } else {
                    x.min(max as u128)
                };
                words.push(clipped & mask(w));
                expected.push(format!(
                    "OBS {i} {}",
                    words
                        .iter()
                        .map(|v| format!("{v:0digits$x}", digits = w.div_ceil(4) as usize))
                        .collect::<Vec<_>>()
                        .join(" ")
                ));
                bench.push_str(&format!(
                    "x={w}'h{x:x}; #1; $display(\"OBS {i} {formats}\"{args});\n"
                ));
            }
            bench.push_str("$finish; end endmodule");
            ir_stream::run(
                &format!("scalar-rounding-{w}-{is_signed}"),
                &b.build(),
                &bench,
                &expected,
            );
        }
    }
}

/// Resize before multiplication preserves the full product; casts and selections keep bit identity.
#[test]
fn scalar_casts_and_selection_preserve_declared_bits() {
    let mut b = ScGraphBuilder::new("scalar_casts");
    let a = b.input("a", ScType::SInt { width: 16 });
    let choice = b.input("choice", ScType::Bool);
    let wide = b.word(WordOp::Resize {
        input: a,
        ty: ScType::SInt { width: 32 },
    });
    let product = b.word(WordOp::Binary {
        kind: WordBinaryOp::Mul,
        lhs: wide,
        rhs: wide,
        ty: ScType::SInt { width: 32 },
    });
    let fixed = b.word(WordOp::Bitcast {
        input: a,
        ty: ScType::FixedPoint { width: 16, frac: 8 },
    });
    let decoded = b.word(WordOp::Bitcast {
        input: fixed,
        ty: ScType::SInt { width: 16 },
    });
    let narrow = b.word(WordOp::Resize {
        input: product,
        ty: ScType::SInt { width: 16 },
    });
    let selected = b.word(WordOp::Select {
        condition: choice,
        when_true: decoded,
        when_false: narrow,
        ty: ScType::SInt { width: 16 },
    });
    let zero_extended = b.word(WordOp::Resize {
        input: choice,
        ty: ScType::UInt { width: 8 },
    });
    b.output("product", product);
    b.output("selected", selected);
    b.output("extended", zero_extended);
    let bench="module tb; reg clk=0,rst_n=1,choice=0; reg [15:0] a=16'h8000; wire [31:0] product; wire [15:0] selected; wire [7:0] extended; scalar_casts dut(.*); initial begin #1; $display(\"OBS %h %h %h\",product,selected,extended); choice=1; #1; $display(\"OBS %h %h %h\",product,selected,extended); a=16'hffff; choice=0; #1; $display(\"OBS %h %h %h\",product,selected,extended); $finish; end endmodule";
    ir_stream::run(
        "scalar-casts",
        &b.build(),
        bench,
        &[
            "OBS 40000000 0000 00".into(),
            "OBS 40000000 8000 01".into(),
            "OBS 00000001 0001 00".into(),
        ],
    );
}

/// Explicit unsigned widening, equal-width casts and 64-bit literal bounds retain every bit.
#[test]
fn scalar_extensions_and_full_range_clamps_preserve_bits() {
    let mut b = ScGraphBuilder::new("scalar_extensions");
    let a = b.input("a", ScType::UInt { width: 64 });
    let wide = b.word(WordOp::Resize {
        input: a,
        ty: ScType::UInt { width: 65 },
    });
    let equal = b.word(WordOp::Resize {
        input: wide,
        ty: ScType::SInt { width: 65 },
    });
    let limited = b.word(WordOp::Clamp {
        input: equal,
        min: i64::MIN,
        max: i64::MAX,
        ty: ScType::SInt { width: 65 },
    });
    let fixed = b.word(WordOp::Bitcast {
        input: a,
        ty: ScType::FixedPoint {
            width: 64,
            frac: 32,
        },
    });
    let recovered = b.word(WordOp::Bitcast {
        input: fixed,
        ty: ScType::UInt { width: 64 },
    });
    b.output("wide", equal);
    b.output("limited", limited);
    b.output("recovered", recovered);
    let bench="module tb; reg clk=0,rst_n=1; reg [63:0] a=64'hffffffffffffffff; wire [64:0] wide,limited; wire [63:0] recovered; scalar_extensions dut(.*); initial begin #1; $display(\"OBS %h %h %h\",wide,limited,recovered); a=0; #1; $display(\"OBS %h %h %h\",wide,limited,recovered); $finish; end endmodule";
    ir_stream::run(
        "scalar-extensions",
        &b.build(),
        bench,
        &[
            "OBS 0ffffffffffffffff 07fffffffffffffff ffffffffffffffff".into(),
            "OBS 00000000000000000 00000000000000000 0000000000000000".into(),
        ],
    );
}

/// Rate's encoded Q8.8 bits become packed observations only through an explicit scalar bitcast.
#[test]
fn explicit_rate_bitcast_feeds_packed_bit_operations() {
    let mut b = ScGraphBuilder::new("rate_bits");
    let raw = b.input("raw_rate", ScType::Rate);
    let bits = b.word(WordOp::Bitcast {
        input: raw,
        ty: ScType::UInt { width: 16 },
    });
    let pattern = b.constant(ScConst::U64(0xff), ScType::UInt { width: 16 });
    let bound = b.bitwise_xor(bits, pattern);
    let count = b.popcount(bound);
    b.output("bits", bound);
    b.output("count", count);
    let bench="module tb; reg clk=0,rst_n=1; reg [15:0] raw_rate=16'h0100; wire [15:0] bits; wire [63:0] count; rate_bits dut(.*); initial begin #1; $display(\"OBS %h %0d\",bits,count); raw_rate=16'hffff; #1; $display(\"OBS %h %0d\",bits,count); $finish; end endmodule";
    ir_stream::run(
        "explicit-rate-bits",
        &b.build(),
        bench,
        &["OBS 01ff 9".into(), "OBS ff00 8".into()],
    );
}
