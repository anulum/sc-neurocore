// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Declared hardware type geometry

//! Validate all declared types before recursive type consumers or hardware dimensions.

use super::{graph::*, verify::VerifyError};

/// Maximum number of nested vector wrappers accepted by hardware IR.
pub(crate) const MAX_VECTOR_DEPTH: usize = 64;

/// Refuse excessive nesting before either public text constructor recurses.
pub(crate) fn check_text_depth(text: &str) -> Result<(), String> {
    if text
        .as_bytes()
        .windows(4)
        .filter(|part| part.eq_ignore_ascii_case(b"vec<"))
        .count()
        > MAX_VECTOR_DEPTH
    {
        Err(format!("IR vector type nesting exceeds {MAX_VECTOR_DEPTH}"))
    } else {
        Ok(())
    }
}

fn width(ty: &ScType) -> Result<usize, String> {
    let mut element = ty;
    let mut count = 1usize;
    let mut depth = 0;
    while let ScType::Vec {
        element: inner,
        count: n,
    } = element
    {
        depth += 1;
        if depth > MAX_VECTOR_DEPTH {
            return Err(format!("IR vector type nesting exceeds {MAX_VECTOR_DEPTH}"));
        }
        if *n == 0 {
            return Err("invalid declared vector type: word count must be positive".into());
        }
        count = count
            .checked_mul(*n)
            .ok_or("invalid declared vector type: word count overflows usize")?;
        element = inner;
    }
    let bits = match element {
        ScType::Bool | ScType::Bitstream { length: 1.. } => 1,
        ScType::Bitstream { length: 0 } => {
            return Err("invalid declared stream type: temporal window must be positive".into())
        }
        ScType::Rate => 16,
        ScType::UInt { width } | ScType::SInt { width } => *width as usize,
        ScType::FixedPoint { width, frac } => {
            if frac >= width {
                return Err(
                    "invalid declared scalar type: fixed-point word requires fraction smaller than word width".into(),
                );
            }
            *width as usize
        }
        ScType::Vec { .. } => unreachable!(),
    };
    let total = count
        .checked_mul(bits)
        .ok_or("invalid declared type: physical word width overflows usize")?;
    if total == 0 || total > i32::MAX as usize {
        Err("invalid declared scalar type: physical word width must fit positive Verilog indexing range".into())
    } else {
        Ok(total)
    }
}

/// Return false if later passes must stop before consuming malformed declarations.
pub(crate) fn verify(graph: &ScGraph, errors: &mut Vec<VerifyError>) -> bool {
    let before = errors.len();
    for (op_index, op) in graph.ops.iter().enumerate() {
        let ty = match op {
            ScOp::Input { ty, .. } | ScOp::Constant { ty, .. } => ty,
            ScOp::Word { op, .. } => op.result_type(),
            _ => continue,
        };
        if let Err(message) = width(ty) {
            errors.push(VerifyError { op_index, message });
        }
    }
    errors.len() == before
}
