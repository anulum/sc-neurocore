// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Typed CIRCT bitstream primitives

//! Lower primitive circuits to real HW/Comb/Seq operations with asynchronous reset.

use super::value_wire;
use crate::ir::constant_words::constant_words;
use crate::ir::dense_contract::{probability_layout, value_type};
use crate::ir::graph::{ScGraph, ScOp, ValueId};
use crate::ir::stream_contract::shape;

/// Emit binary observations and return the SSA name of their completed count.
fn count(
    m: &mut String,
    g: &ScGraph,
    id: ValueId,
    input: ValueId,
    decode: bool,
) -> Result<String, String> {
    let layout = shape(g, input)?;
    let p = format!("v{}_observation", id.0);
    let source = value_wire(g, input);
    m.push_str(&format!(
        "  %{p}_zero = hw.constant 0 : i64\n  %{p}_pad = hw.constant 0 : i63\n"
    ));
    let mut level = Vec::new();
    for i in 0..layout.width {
        let bit = if layout.width == 1 {
            source.clone()
        } else {
            m.push_str(&format!(
                "  %{p}_bit{i} = comb.extract {source} from {i} : (i{}) -> i1\n",
                layout.width
            ));
            format!("%{p}_bit{i}")
        };
        m.push_str(&format!(
            "  %{p}_word{i} = comb.concat %{p}_pad, {bit} : i63, i1\n"
        ));
        level.push(format!("%{p}_word{i}"));
    }
    let mut n = 0;
    while level.len() > 1 {
        let mut next = Vec::new();
        for pair in level.chunks(2) {
            if pair.len() == 1 {
                next.push(pair[0].clone());
            } else {
                m.push_str(&format!(
                    "  %{p}_sum{n} = comb.add {}, {} : i64\n",
                    pair[0], pair[1]
                ));
                next.push(format!("%{p}_sum{n}"));
                n += 1;
            }
        }
        level = next;
    }
    let result = if decode {
        format!("%v{}_count", id.0)
    } else {
        format!("%v{}", id.0)
    };
    if let Some(window) = layout.window {
        m.push_str(&format!(
            "  %{p}_one = hw.constant 1 : i64\n\
             \x20 %{p}_last = hw.constant {} : i64\n\
             \x20 %{p}_phase = seq.firreg %{p}_next_phase clock %sc_clock reset async %sc_reset, %{p}_zero : i64\n\
             \x20 %{p}_acc = seq.firreg %{p}_next_acc clock %sc_clock reset async %sc_reset, %{p}_zero : i64\n\
             \x20 {result} = seq.firreg %{p}_next_result clock %sc_clock reset async %sc_reset, %{p}_zero : i64\n\
             \x20 %{p}_done = comb.icmp eq %{p}_phase, %{p}_last : i64\n\
             \x20 %{p}_increment = comb.add %{p}_phase, %{p}_one : i64\n\
             \x20 %{p}_total = comb.add %{p}_acc, {} : i64\n\
             \x20 %{p}_next_phase = comb.mux %{p}_done, %{p}_zero, %{p}_increment : i64\n\
             \x20 %{p}_next_acc = comb.mux %{p}_done, %{p}_zero, %{p}_total : i64\n\
             \x20 %{p}_next_result = comb.mux %{p}_done, %{p}_total, {result} : i64\n",
            window - 1, level[0]
        ));
    } else {
        m.push_str(&format!(
            "  {result} = comb.or {}, %{p}_zero : i64\n",
            level[0]
        ));
    }
    Ok(result)
}

/// Emit the maintained encoder's full-width probability conversion and registered LFSR.
fn encode(
    m: &mut String,
    g: &ScGraph,
    id: ValueId,
    input: ValueId,
    seed: u16,
) -> Result<(), String> {
    let layout = probability_layout(&value_type(g, input)?)?;
    let (w, frac) = (layout.width, layout.fraction);
    let wide = w + 17;
    let p = format!("v{}_encoder", id.0);
    let raw = value_wire(g, input);
    m.push_str(&format!(
        "  %{p}_zero_raw = hw.constant 0 : i{w}\n\
         \x20 %{p}_nonpositive = comb.icmp sle {raw}, %{p}_zero_raw : i{w}\n\
         \x20 %{p}_pad = hw.constant 0 : i17\n\
         \x20 %{p}_raw_wide = comb.concat %{p}_pad, {raw} : i17, i{w}\n\
         \x20 %{p}_scale = hw.constant 65535 : i{wide}\n\
         \x20 %{p}_product = comb.mul %{p}_raw_wide, %{p}_scale : i{wide}\n\
         \x20 %{p}_fraction = comb.extract %{p}_product from {frac} : (i{wide}) -> i16\n\
         \x20 %{p}_zero = hw.constant 0 : i16\n"
    ));
    let positive = if frac + 1 < w {
        m.push_str(&format!(
            "  %{p}_one_raw = hw.constant {} : i{w}\n\
             \x20 %{p}_full = hw.constant 65535 : i16\n\
             \x20 %{p}_at_least_one = comb.icmp sge {raw}, %{p}_one_raw : i{w}\n\
             \x20 %{p}_clipped = comb.mux %{p}_at_least_one, %{p}_full, %{p}_fraction : i16\n",
            1_u64 << frac
        ));
        format!("%{p}_clipped")
    } else {
        format!("%{p}_fraction")
    };
    m.push_str(&format!(
        "  %{p}_threshold = comb.mux %{p}_nonpositive, %{p}_zero, {positive} : i16\n\
         \x20 %{p}_seed = hw.constant {seed} : i16\n\
         \x20 %{p}_lfsr = seq.firreg %{p}_next clock %sc_clock reset async %sc_reset, %{p}_seed : i16\n"
    ));
    for tap in [15, 13, 12, 10] {
        m.push_str(&format!(
            "  %{p}_tap{tap} = comb.extract %{p}_lfsr from {tap} : (i16) -> i1\n"
        ));
    }
    m.push_str(&format!(
        "  %{p}_feedback = comb.xor %{p}_tap15, %{p}_tap13, %{p}_tap12, %{p}_tap10 : i1\n\
         \x20 %{p}_low = comb.extract %{p}_lfsr from 0 : (i16) -> i15\n\
         \x20 %{p}_next = comb.concat %{p}_low, %{p}_feedback : i15, i1\n\
         \x20 %{p}_sample = comb.icmp ult %{p}_lfsr, %{p}_threshold : i16\n\
         \x20 %{p}_zero_bit = hw.constant 0 : i1\n\
         \x20 %v{} = seq.firreg %{p}_sample clock %sc_clock reset async %sc_reset, %{p}_zero_bit : i1\n", id.0
    ));
    Ok(())
}

/// Lower supported primitives, leaving compound neuron lowering to its own emitter.
pub(super) fn emit(m: &mut String, g: &ScGraph, op: &ScOp) -> Result<bool, String> {
    match op {
        ScOp::Constant { id, value, ty } => {
            let (layout, words) = constant_words(value, ty)?;
            if words.len() == 1 {
                m.push_str(&format!(
                    "  %c{} = hw.constant {} : i{}\n",
                    id.0, words[0], layout.width
                ));
            } else {
                let mut names = Vec::new();
                for (i, word) in words.iter().enumerate().rev() {
                    m.push_str(&format!(
                        "  %c{}_{i} = hw.constant {word} : i{}\n",
                        id.0, layout.width
                    ));
                    names.push(format!("%c{}_{i}", id.0));
                }
                m.push_str(&format!(
                    "  %c{} = comb.concat {} : {}\n",
                    id.0,
                    names.join(", "),
                    vec![format!("i{}", layout.width); names.len()].join(", ")
                ));
            }
        }
        ScOp::Encode { id, prob, seed, .. } => encode(m, g, *id, *prob, *seed)?,
        ScOp::BitwiseAnd { id, lhs, rhs } | ScOp::BitwiseXor { id, lhs, rhs } => {
            let operator = if matches!(op, ScOp::BitwiseAnd { .. }) {
                "and"
            } else {
                "xor"
            };
            let width = shape(g, *lhs)?.width;
            m.push_str(&format!(
                "  %v{} = comb.{operator} {}, {} : i{width}\n",
                id.0,
                value_wire(g, *lhs),
                value_wire(g, *rhs)
            ));
        }
        ScOp::Popcount { id, input } => {
            count(m, g, *id, *input, false)?;
        }
        ScOp::Decode { id, input } => {
            let observations = shape(g, *input)?.opportunities()?;
            let counted = count(m, g, *id, *input, true)?;
            let count_bits = 64 - observations.leading_zeros();
            let count_word = if count_bits < 64 {
                m.push_str(&format!("  %v{n}_bounded_count = comb.extract {counted} from 0 : (i64) -> i{count_bits}\n", n=id.0));
                format!("%v{}_bounded_count", id.0)
            } else {
                counted
            };
            m.push_str(&format!("  %v{n}_zero_byte = hw.constant 0 : i8\n  %v{n}_scaled = comb.concat {count_word}, %v{n}_zero_byte : i{count_bits}, i8\n",n=id.0));
            super::divide::emit(
                m,
                &format!("%v{}", id.0),
                &format!("%v{}_scaled", id.0),
                count_bits as usize + 8,
                observations,
                16,
            );
        }
        ScOp::DivConst { id, input, divisor } => {
            super::divide::emit(
                m,
                &format!("%v{}", id.0),
                &value_wire(g, *input),
                crate::ir::emit_sv::find_value_width(g, *input),
                *divisor,
                64,
            );
        }
        _ => return Ok(false),
    }
    Ok(true)
}
