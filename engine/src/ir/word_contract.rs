// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Scalar word admission

//! Check explicit geometry without recursively traversing graph dependencies.

use super::{graph::*, verify::VerifyError, word::*};
use std::collections::HashMap;

pub(crate) fn is_integer(ty: &ScType) -> bool {
    matches!(ty, ScType::UInt { .. } | ScType::SInt { .. })
}
pub(crate) fn is_signed(ty: &ScType) -> bool {
    matches!(
        ty,
        ScType::SInt { .. } | ScType::FixedPoint { .. } | ScType::Rate
    )
}
fn scalar(ty: &ScType) -> Result<(), String> {
    let valid = match ty {
        ScType::Bool | ScType::Rate => true,
        ScType::UInt { width } | ScType::SInt { width } => *width > 0 && *width < i32::MAX as u32,
        ScType::FixedPoint { width, frac } => {
            *width > 0 && *width < i32::MAX as u32 && *frac < *width
        }
        _ => false,
    };
    if valid {
        Ok(())
    } else {
        Err(format!(
            "word operation requires a positive scalar type, got {ty}"
        ))
    }
}
fn integer(ty: &ScType) -> Result<(), String> {
    scalar(ty)?;
    if is_integer(ty) {
        Ok(())
    } else {
        Err(format!(
            "raw arithmetic requires iN or uN, got {ty}; use an explicit bitcast"
        ))
    }
}
fn fits(value: i64, ty: &ScType) -> bool {
    let w = ty.bit_width();
    if is_signed(ty) {
        w >= 64 || ((-(1i64 << (w - 1)))..=(1i64 << (w - 1)) - 1).contains(&value)
    } else {
        value >= 0 && (w >= 63 || value < (1i64 << w))
    }
}
fn validate(op: &WordOp, types: &HashMap<ValueId, ScType>) -> Result<(), String> {
    scalar(op.result_type())?;
    let get = |id: &ValueId| {
        types
            .get(id)
            .ok_or_else(|| format!("{id} has no scalar word producer type"))
    };
    let same = |id: &ValueId, ty: &ScType| -> Result<(), String> {
        let actual = get(id)?;
        if actual == ty {
            Ok(())
        } else {
            Err(format!("word operand {id} requires {ty}, got {actual}"))
        }
    };
    match op {
        WordOp::Binary { kind, lhs, rhs, ty } => {
            integer(ty)?;
            same(lhs, ty)?;
            same(rhs, ty)?;
            if *kind == WordBinaryOp::AShr && !is_signed(ty) {
                return Err("ashr requires a signed integer".into());
            }
        }
        WordOp::Unary { input, ty, .. } | WordOp::Round { input, ty, .. } => {
            integer(ty)?;
            same(input, ty)?;
            if let WordOp::Round { shift, .. } = op {
                if *shift as usize >= ty.bit_width() {
                    return Err("round shift must be smaller than word width".into());
                }
            }
        }
        WordOp::Compare { lhs, rhs, .. } => {
            integer(get(lhs)?)?;
            same(rhs, get(lhs)?)?;
        }
        WordOp::Resize { input, ty } => {
            integer(ty)?;
            if get(input)? != &ScType::Bool {
                integer(get(input)?)?;
            }
        }
        WordOp::Bitcast { input, ty } => {
            scalar(get(input)?)?;
            if get(input)?.bit_width() != ty.bit_width() {
                return Err("bitcast must preserve physical word width".into());
            }
        }
        WordOp::Select {
            condition,
            when_true,
            when_false,
            ty,
        } => {
            same(condition, &ScType::Bool)?;
            same(when_true, ty)?;
            same(when_false, ty)?;
        }
        WordOp::Clamp {
            input,
            min,
            max,
            ty,
        } => {
            integer(ty)?;
            same(input, ty)?;
            if min > max || !fits(*min, ty) || !fits(*max, ty) {
                return Err(
                    "clamp bounds must be ordered and fit the declared integer type".into(),
                );
            }
        }
    }
    Ok(())
}

pub(crate) fn verify(graph: &ScGraph, errors: &mut Vec<VerifyError>) {
    let mut types = HashMap::new();
    for (op_index, op) in graph.ops.iter().enumerate() {
        let ty = match op {
            ScOp::Word { op, .. } => {
                if let Err(message) = validate(op, &types) {
                    errors.push(VerifyError { op_index, message });
                }
                Some(op.result_type().clone())
            }
            ScOp::Input { ty, .. } | ScOp::Constant { ty, .. } => Some(ty.clone()),
            ScOp::Encode { length, .. } => Some(ScType::Bitstream { length: *length }),
            ScOp::BitwiseAnd { lhs, .. } | ScOp::BitwiseXor { lhs, .. } => types.get(lhs).cloned(),
            ScOp::Popcount { .. } | ScOp::DivConst { .. } => Some(ScType::UInt { width: 64 }),
            ScOp::Decode { .. } => Some(ScType::Rate),
            _ => None,
        };
        if let Some(ty) = ty {
            types.insert(op.result_id(), ty);
        }
    }
}
