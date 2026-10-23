// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Scalar word CIRCT lowering

use super::value_wire;
use crate::ir::{graph::*, stream_contract, word::*, word_contract::is_signed};

pub(super) fn emit(
    m: &mut String,
    graph: &ScGraph,
    id: ValueId,
    op: &WordOp,
) -> Result<(), String> {
    let p = format!("v{}", id.0);
    let ty = op.result_type();
    let w = ty.bit_width();
    let wire = |v| value_wire(graph, v);
    match op {
        WordOp::Delay { input, .. } => {
            m.push_str(&format!(
                "  %{p}_reset = hw.constant 0 : i{w}\n\
                 \x20 %{p} = seq.firreg {} clock %sc_clock reset async %sc_reset, %{p}_reset : i{w}\n",
                wire(*input)
            ));
        }
        WordOp::Binary {
            kind: WordBinaryOp::Div | WordBinaryOp::Rem,
            lhs,
            rhs,
            ..
        } => {
            divide(
                m,
                &p,
                &wire(*lhs),
                &wire(*rhs),
                w,
                is_signed(ty),
                matches!(
                    op,
                    WordOp::Binary {
                        kind: WordBinaryOp::Rem,
                        ..
                    }
                ),
            );
        }
        WordOp::Binary { kind, lhs, rhs, .. } => {
            let name = match kind {
                WordBinaryOp::Add => "add",
                WordBinaryOp::Sub => "sub",
                WordBinaryOp::Mul => "mul",
                WordBinaryOp::And => "and",
                WordBinaryOp::Or => "or",
                WordBinaryOp::Xor => "xor",
                WordBinaryOp::Shl => "shl",
                WordBinaryOp::LShr => "shru",
                WordBinaryOp::AShr => "shrs",
                WordBinaryOp::Div | WordBinaryOp::Rem => unreachable!(),
            };
            m.push_str(&format!(
                "  %{p} = comb.{name} {}, {} : i{w}\n",
                wire(*lhs),
                wire(*rhs)
            ));
        }
        WordOp::Unary { kind, input, .. } => {
            let (constant, name) = if *kind == WordUnaryOp::Neg {
                (0, "sub")
            } else {
                (-1, "xor")
            };
            m.push_str(&format!("  %{p}_constant = hw.constant {constant} : i{w}\n  %{p} = comb.{name} %{p}_constant, {} : i{w}\n", wire(*input)));
        }
        WordOp::Compare {
            predicate,
            lhs,
            rhs,
        } => {
            let signed = is_signed(&stream_contract::value_type(graph, *lhs)?);
            let prefix = if signed { "s" } else { "u" };
            let comparison = match predicate {
                WordCompareOp::Eq | WordCompareOp::Ne => predicate.to_string(),
                _ => format!("{prefix}{predicate}"),
            };
            let width = stream_contract::value_type(graph, *lhs)?.bit_width();
            m.push_str(&format!(
                "  %{p} = comb.icmp {comparison} {}, {} : i{width}\n",
                wire(*lhs),
                wire(*rhs)
            ));
        }
        WordOp::Resize { input, .. } => {
            let input_ty = stream_contract::value_type(graph, *input)?;
            let iw = input_ty.bit_width();
            resize(m, &p, &wire(*input), iw, w, is_signed(&input_ty));
        }
        WordOp::Bitcast { input, .. }
        | WordOp::Round {
            input, shift: 0, ..
        } => alias(m, &p, &wire(*input), w),
        WordOp::Select {
            condition,
            when_true,
            when_false,
            ..
        } => {
            m.push_str(&format!(
                "  %{p} = comb.mux {}, {}, {} : i{w}\n",
                wire(*condition),
                wire(*when_true),
                wire(*when_false)
            ));
        }
        WordOp::Round {
            input, shift, mode, ..
        } => round(m, &p, &wire(*input), w, *shift, *mode, is_signed(ty)),
        WordOp::Clamp {
            input, min, max, ..
        } => {
            let x = wire(*input);
            let prefix = if is_signed(ty) { "s" } else { "u" };
            m.push_str(&format!("  %{p}_min = hw.constant {min} : i{w}\n  %{p}_max = hw.constant {max} : i{w}\n  %{p}_below = comb.icmp {prefix}lt {x}, %{p}_min : i{w}\n  %{p}_above = comb.icmp {prefix}gt {x}, %{p}_max : i{w}\n  %{p}_upper = comb.mux %{p}_above, %{p}_max, {x} : i{w}\n  %{p} = comb.mux %{p}_below, %{p}_min, %{p}_upper : i{w}\n"));
        }
    }
    Ok(())
}
fn alias(m: &mut String, p: &str, value: &str, w: usize) {
    m.push_str(&format!("  %{p} = comb.or {value}, {value} : i{w}\n"));
}
fn resize(m: &mut String, p: &str, value: &str, iw: usize, w: usize, signed: bool) {
    match w.cmp(&iw) {
        std::cmp::Ordering::Equal => alias(m, p, value, w),
        std::cmp::Ordering::Less => m.push_str(&format!(
            "  %{p} = comb.extract {value} from 0 : (i{iw}) -> i{w}\n"
        )),
        std::cmp::Ordering::Greater => {
            let pad = w - iw;
            if signed {
                m.push_str(&format!("  %{p}_sign = comb.extract {value} from {} : (i{iw}) -> i1\n  %{p}_padding = comb.replicate %{p}_sign : (i1) -> i{pad}\n", iw-1));
            } else {
                m.push_str(&format!("  %{p}_padding = hw.constant 0 : i{pad}\n"));
            }
            m.push_str(&format!(
                "  %{p} = comb.concat %{p}_padding, {value} : i{pad}, i{iw}\n"
            ));
        }
    }
}
fn divide(m: &mut String, p: &str, lhs: &str, rhs: &str, w: usize, signed: bool, modulo: bool) {
    m.push_str(&format!("  %{p}_zero = hw.constant 0 : i{w}\n  %{p}_wide_zero = hw.constant 0 : i{}\n  %{p}_zero_bit = hw.constant 0 : i1\n", w+1));
    for (name, value) in [("a", lhs), ("b", rhs)] {
        if signed {
            m.push_str(&format!("  %{p}_{name}_sign = comb.extract {value} from {} : (i{w}) -> i1\n  %{p}_{name}_negated = comb.sub %{p}_zero, {value} : i{w}\n  %{p}_{name}_magnitude = comb.mux %{p}_{name}_sign, %{p}_{name}_negated, {value} : i{w}\n",w-1));
        } else {
            alias(m, &format!("{p}_{name}_sign"), &format!("%{p}_zero_bit"), 1);
            alias(m, &format!("{p}_{name}_magnitude"), value, w);
        }
    }
    let wide = w + 1;
    m.push_str(&format!(
        "  %{p}_divisor = comb.concat %{p}_zero_bit, %{p}_b_magnitude : i1, i{w}\n"
    ));
    let mut remainder = format!("%{p}_wide_zero");
    let mut quotient = Vec::new();
    for i in (0..w).rev() {
        m.push_str(&format!("  %{p}_bit{i} = comb.extract %{p}_a_magnitude from {i} : (i{w}) -> i1\n  %{p}_low{i} = comb.extract {remainder} from 0 : (i{wide}) -> i{w}\n  %{p}_shift{i} = comb.concat %{p}_low{i}, %{p}_bit{i} : i{w}, i1\n  %{p}_take{i} = comb.icmp uge %{p}_shift{i}, %{p}_divisor : i{wide}\n  %{p}_subtract{i} = comb.sub %{p}_shift{i}, %{p}_divisor : i{wide}\n  %{p}_rem{i} = comb.mux %{p}_take{i}, %{p}_subtract{i}, %{p}_shift{i} : i{wide}\n"));
        remainder = format!("%{p}_rem{i}");
        quotient.push(format!("%{p}_take{i}"));
    }
    if modulo {
        m.push_str(&format!(
            "  %{p}_magnitude = comb.extract {remainder} from 0 : (i{wide}) -> i{w}\n"
        ));
    } else if w == 1 {
        alias(m, &format!("{p}_magnitude"), &quotient[0], w);
    } else {
        m.push_str(&format!(
            "  %{p}_magnitude = comb.concat {} : {}\n",
            quotient.join(", "),
            vec!["i1"; w].join(", ")
        ));
    }
    let sign = if modulo {
        format!("%{p}_a_sign")
    } else {
        m.push_str(&format!(
            "  %{p}_quotient_sign = comb.xor %{p}_a_sign, %{p}_b_sign : i1\n"
        ));
        format!("%{p}_quotient_sign")
    };
    m.push_str(&format!("  %{p}_negative = comb.sub %{p}_zero, %{p}_magnitude : i{w}\n  %{p}_signed = comb.mux {sign}, %{p}_negative, %{p}_magnitude : i{w}\n  %{p}_divisor_zero = comb.icmp eq {rhs}, %{p}_zero : i{w}\n  %{p} = comb.mux %{p}_divisor_zero, %{p}_zero, %{p}_signed : i{w}\n"));
}
fn round(
    m: &mut String,
    p: &str,
    input: &str,
    w: usize,
    shift: u32,
    mode: WordRounding,
    signed: bool,
) {
    let shrop = if signed { "shrs" } else { "shru" };
    m.push_str(&format!("  %{p}_shift = hw.constant {shift} : i{w}\n  %{p}_floor = comb.{shrop} {input}, %{p}_shift : i{w}\n  %{p}_residue = comb.extract {input} from 0 : (i{w}) -> i{shift}\n  %{p}_residue_zero = hw.constant 0 : i{shift}\n  %{p}_one = hw.constant 1 : i{shift}\n  %{p}_half_shift = hw.constant {} : i{shift}\n  %{p}_half = comb.shl %{p}_one, %{p}_half_shift : i{shift}\n  %{p}_gt = comb.icmp ugt %{p}_residue, %{p}_half : i{shift}\n  %{p}_eq = comb.icmp eq %{p}_residue, %{p}_half : i{shift}\n  %{p}_nonzero = comb.icmp ne %{p}_residue, %{p}_residue_zero : i{shift}\n  %{p}_odd = comb.extract %{p}_floor from 0 : (i{w}) -> i1\n  %{p}_true = hw.constant 1 : i1\n  %{p}_false = hw.constant 0 : i1\n",shift-1));
    if signed {
        m.push_str(&format!(
            "  %{p}_sign = comb.extract {input} from {} : (i{w}) -> i1\n",
            w - 1
        ));
    } else {
        alias(m, &format!("{p}_sign"), &format!("%{p}_false"), 1);
    }
    let increment = match mode {
        WordRounding::Floor => format!("%{p}_false"),
        WordRounding::TowardZero => {
            m.push_str(&format!(
                "  %{p}_toward_zero = comb.and %{p}_sign, %{p}_nonzero : i1\n"
            ));
            format!("%{p}_toward_zero")
        }
        WordRounding::HalfUp => {
            m.push_str(&format!("  %{p}_half_up = comb.or %{p}_gt, %{p}_eq : i1\n"));
            format!("%{p}_half_up")
        }
        WordRounding::HalfEven | WordRounding::HalfAway => {
            let tie = if mode == WordRounding::HalfEven {
                format!("%{p}_odd")
            } else {
                m.push_str(&format!(
                    "  %{p}_positive = comb.xor %{p}_sign, %{p}_true : i1\n"
                ));
                format!("%{p}_positive")
            };
            m.push_str(&format!("  %{p}_tie = comb.and %{p}_eq, {tie} : i1\n  %{p}_nearest = comb.or %{p}_gt, %{p}_tie : i1\n"));
            format!("%{p}_nearest")
        }
    };
    resize(m, &format!("{p}_increment"), &increment, 1, w, false);
    m.push_str(&format!(
        "  %{p} = comb.add %{p}_floor, %{p}_increment : i{w}\n"
    ));
}
