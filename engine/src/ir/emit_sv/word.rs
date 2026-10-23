// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Scalar word SystemVerilog lowering

use super::value_to_wire;
use crate::ir::{graph::*, stream_contract, word::*, word_contract::is_signed};

pub(super) fn emit(
    s: &mut String,
    graph: &ScGraph,
    id: ValueId,
    op: &WordOp,
) -> Result<(), String> {
    let out = format!("v{}", id.0);
    let wire = |v| value_to_wire(graph, v);
    let ty = op.result_type();
    let w = ty.bit_width();
    let cast = |v: String| {
        if is_signed(ty) {
            format!("$signed({v})")
        } else {
            format!("$unsigned({v})")
        }
    };
    let expression = match op {
        WordOp::Delay { input, .. } => {
            s.push_str(&format!(
                "    always @(posedge clk or negedge rst_n) begin\n\
                 \x20       if (!rst_n) {out} <= {w}'d0;\n\
                 \x20       else {out} <= {};\n\
                 \x20   end\n",
                wire(*input)
            ));
            return Ok(());
        }
        WordOp::Binary {
            kind: WordBinaryOp::Div | WordBinaryOp::Rem,
            lhs,
            rhs,
            ..
        } => {
            divide(
                s,
                &out,
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
            return Ok(());
        }
        WordOp::Binary { kind, lhs, rhs, .. } => {
            let token = match kind {
                WordBinaryOp::Add => "+",
                WordBinaryOp::Sub => "-",
                WordBinaryOp::Mul => "*",
                WordBinaryOp::And => "&",
                WordBinaryOp::Or => "|",
                WordBinaryOp::Xor => "^",
                WordBinaryOp::Shl => "<<",
                WordBinaryOp::LShr => ">>",
                WordBinaryOp::AShr => ">>>",
                WordBinaryOp::Div | WordBinaryOp::Rem => unreachable!(),
            };
            // Shift counts are physical unsigned words even for signed arithmetic.
            let rhs = if matches!(
                kind,
                WordBinaryOp::Shl | WordBinaryOp::LShr | WordBinaryOp::AShr
            ) {
                format!("$unsigned({})", wire(*rhs))
            } else {
                cast(wire(*rhs))
            };
            format!("{} {token} {rhs}", cast(wire(*lhs)))
        }
        WordOp::Unary { kind, input, .. } => format!(
            "{}{}",
            if *kind == WordUnaryOp::Neg { "-" } else { "~" },
            cast(wire(*input))
        ),
        WordOp::Compare {
            predicate,
            lhs,
            rhs,
        } => {
            let signed = is_signed(&stream_contract::value_type(graph, *lhs)?);
            let cast = if signed { "$signed" } else { "$unsigned" };
            let token = match predicate {
                WordCompareOp::Eq => "==",
                WordCompareOp::Ne => "!=",
                WordCompareOp::Lt => "<",
                WordCompareOp::Le => "<=",
                WordCompareOp::Gt => ">",
                WordCompareOp::Ge => ">=",
            };
            format!("{cast}({}) {token} {cast}({})", wire(*lhs), wire(*rhs))
        }
        WordOp::Resize { input, .. } => {
            let signed = is_signed(&stream_contract::value_type(graph, *input)?);
            format!(
                "{}({})",
                if signed { "$signed" } else { "$unsigned" },
                wire(*input)
            )
        }
        WordOp::Bitcast { input, .. } => wire(*input),
        WordOp::Select {
            condition,
            when_true,
            when_false,
            ..
        } => format!(
            "{} ? {} : {}",
            wire(*condition),
            wire(*when_true),
            wire(*when_false)
        ),
        WordOp::Round {
            input, shift: 0, ..
        } => wire(*input),
        WordOp::Round {
            input, shift, mode, ..
        } => {
            round(s, &out, &wire(*input), w, *shift, *mode, is_signed(ty));
            return Ok(());
        }
        WordOp::Clamp {
            input, min, max, ..
        } => {
            let x = cast(wire(*input));
            let literal = |v: i64| {
                cast(if v < 0 {
                    format!("-{w}'d{}", v.unsigned_abs())
                } else {
                    format!("{w}'d{v}")
                })
            };
            let lo = literal(*min);
            let hi = literal(*max);
            format!("({x} < {lo}) ? {lo} : (({x} > {hi}) ? {hi} : {x})")
        }
    };
    s.push_str(&format!("    assign {out} = {expression};\n"));
    Ok(())
}

fn divide(s: &mut String, p: &str, lhs: &str, rhs: &str, w: usize, signed: bool, modulo: bool) {
    let sign_a = if signed {
        format!("a[{}]", w - 1)
    } else {
        "1'b0".into()
    };
    let sign_b = if signed {
        format!("b[{}]", w - 1)
    } else {
        "1'b0".into()
    };
    let result = if modulo { "rem_word" } else { "quotient" };
    let negative = if modulo {
        sign_a.clone()
    } else {
        format!("({sign_a} ^ {sign_b})")
    };
    s.push_str(&format!(
        r#"    function automatic [{high}:0] {p}_divide(input [{high}:0] a, b);
        reg [{high}:0] numerator, denominator, quotient, rem_word;
        reg [{w}:0] remainder;
        integer bit_index;
        begin
            numerator = {sign_a} ? -a : a;
            denominator = {sign_b} ? -b : b;
            quotient = 0;
            remainder = 0;
            for (bit_index = {high}; bit_index >= 0; bit_index = bit_index - 1) begin
                remainder = (remainder << 1) | numerator[bit_index];
                if (remainder >= {{1'b0, denominator}}) begin
                    remainder = remainder - {{1'b0, denominator}};
                    quotient[bit_index] = 1'b1;
                end
            end
            rem_word = remainder[{high}:0];
            {p}_divide = (b == 0) ? {w}'d0 : ({negative} ? -{result} : {result});
        end
    endfunction
    assign {p} = {p}_divide({lhs}, {rhs});
"#,
        high = w - 1
    ));
}

fn round(
    s: &mut String,
    p: &str,
    input: &str,
    w: usize,
    shift: u32,
    mode: WordRounding,
    signed: bool,
) {
    let cast = if signed { "$signed" } else { "$unsigned" };
    let negative = if signed {
        format!("{input}[{}]", w - 1)
    } else {
        "1'b0".into()
    };
    let residue = format!("{p}_residue");
    let half = format!("({shift}'d1 << {})", shift - 1);
    let increment = match mode {
        WordRounding::Floor => "1'b0".into(),
        WordRounding::TowardZero => format!("({negative} && ({residue} != 0))"),
        WordRounding::HalfUp => format!("({residue} >= {half})"),
        WordRounding::HalfEven => {
            format!("(({residue} > {half}) || (({residue} == {half}) && {p}_floor[0]))")
        }
        WordRounding::HalfAway => {
            format!("(({residue} > {half}) || (({residue} == {half}) && !{negative}))")
        }
    };
    s.push_str(&format!("    wire [{hi}:0] {p}_floor = {cast}({input}) >>> {shift};\n    wire [{lo}:0] {residue} = {input}[{lo}:0];\n    assign {p} = {p}_floor + {increment};\n", hi=w-1, lo=shift-1));
}
