// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Temporal stream and packed word contracts

//! Separate clocked bitstream windows from combinational packed bit vectors.

use std::collections::HashMap;

use super::graph::{ScGraph, ScOp, ScType, ValueId};

/// Physical bus width and optional temporal observation window.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct BitShape {
    /// Number of simultaneously sampled physical bits.
    pub width: usize,
    /// Clock samples per window; None means a combinational packed word.
    pub window: Option<u64>,
}

impl BitShape {
    /// Number of binary observations represented by a completed population count.
    pub fn opportunities(self) -> Result<u64, String> {
        (self.width as u64)
            .checked_mul(self.window.unwrap_or(1))
            .ok_or_else(|| "bitstream observation count exceeds u64".into())
    }
}

/// Infer primitive result types in definition order, without following graph cycles.
pub(crate) fn value_type(graph: &ScGraph, id: ValueId) -> Result<ScType, String> {
    let mut types = HashMap::new();
    for op in &graph.ops {
        let ty = match op {
            ScOp::Input { ty, .. } | ScOp::Constant { ty, .. } => Some(ty.clone()),
            ScOp::Encode { length, .. } => Some(ScType::Bitstream { length: *length }),
            ScOp::BitwiseAnd { lhs, .. } | ScOp::BitwiseXor { lhs, .. } => types.get(lhs).cloned(),
            ScOp::Popcount { .. } | ScOp::DivConst { .. } => Some(ScType::UInt { width: 64 }),
            ScOp::Decode { .. } => Some(ScType::Rate),
            _ => None,
        };
        if let Some(ty) = ty {
            if op.result_id() == id {
                return Ok(ty);
            }
            types.insert(op.result_id(), ty);
        }
    }
    Err(format!("{id} has no declared or inferred bit operand type"))
}

/// Resolve packed integer geometry or a vector of synchronous temporal streams.
pub(crate) fn type_shape(ty: &ScType) -> Result<BitShape, String> {
    let shape = match ty {
        ScType::Bool => BitShape {
            width: 1,
            window: None,
        },
        ScType::UInt { width } | ScType::SInt { width } => BitShape {
            width: *width as usize,
            window: None,
        },
        ScType::Bitstream { length } if *length > 0 => BitShape {
            width: 1,
            window: Some(*length as u64),
        },
        ScType::Vec { element, count } if *count > 0 => {
            let element = type_shape(element)?;
            BitShape {
                width: element
                    .width
                    .checked_mul(*count)
                    .ok_or_else(|| "bit operand bus width overflows usize".to_string())?,
                window: element.window,
            }
        }
        _ => {
            return Err(format!(
                "expected a nonempty bitstream or packed integer, got {ty}"
            ))
        }
    };
    if shape.width == 0 || shape.width > i32::MAX as usize {
        return Err("bit operand width must fit positive Verilog indexing range".into());
    }
    shape.opportunities()?;
    Ok(shape)
}

/// Resolve the complete bit geometry of an operation's input.
pub(crate) fn shape(graph: &ScGraph, id: ValueId) -> Result<BitShape, String> {
    type_shape(&value_type(graph, id)?)
}

/// Admit compatible shapes before emitting scalar selects, counters or divisors.
pub(crate) fn validate(graph: &ScGraph, op: &ScOp) -> Result<(), String> {
    match op {
        ScOp::Encode { length, .. } => {
            type_shape(&ScType::Bitstream { length: *length })?;
        }
        ScOp::BitwiseAnd { lhs, rhs, .. } | ScOp::BitwiseXor { lhs, rhs, .. } => {
            let l = value_type(graph, *lhs)?;
            let r = value_type(graph, *rhs)?;
            type_shape(&l)?;
            type_shape(&r)?;
            if l != r {
                return Err(format!(
                    "bit operands require identical types and windows: {l} versus {r}"
                ));
            }
        }
        ScOp::Popcount { input, .. } | ScOp::Decode { input, .. } => {
            shape(graph, *input)?;
        }
        ScOp::DivConst { divisor: 0, .. } => return Err("DivConst divisor must be nonzero".into()),
        ScOp::DivConst { input, .. } if shape(graph, *input)?.window.is_some() => {
            return Err("DivConst requires a packed integer; decode temporal streams first".into());
        }
        _ => {}
    }
    Ok(())
}
