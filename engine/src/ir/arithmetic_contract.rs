// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Compound scalar and reduction geometry

//! Shared lane interpretation for both hardware emitters; lane zero occupies the low bits.

use super::graph::*;

/// Physical geometry of homogeneous numeric lanes.
pub(crate) struct Lanes {
    /// Number of physical bits in one numeric lane.
    pub width: usize,
    /// Number of contiguous lanes, ordered from low to high bits.
    pub count: usize,
    /// Whether each lane has a two's-complement interpretation.
    pub signed: bool,
}

/// Infer a declared operand or a previously computed scalar arithmetic result.
pub(crate) fn lanes(graph: &ScGraph, id: ValueId) -> Result<Lanes, String> {
    let ty = super::stream_contract::value_type(graph, id)?;
    let mut element = &ty;
    let mut count = 1usize;
    while let ScType::Vec {
        element: inner,
        count: n,
    } = element
    {
        count = count
            .checked_mul(*n)
            .ok_or("numeric lane count overflows usize")?;
        element = inner;
    }
    let (width, signed) = match element {
        ScType::Bool => (1, false),
        ScType::Rate => (16, true),
        ScType::FixedPoint { width, .. } | ScType::SInt { width } => (*width as usize, true),
        ScType::UInt { width } => (*width as usize, false),
        _ => {
            return Err(format!(
                "numeric arithmetic requires scalar lanes, got {element}"
            ))
        }
    };
    if width
        .checked_mul(count)
        .filter(|w| *w > 0 && *w <= i32::MAX as usize)
        .is_none()
    {
        return Err("numeric lane geometry exceeds the supported hardware indexing range".into());
    }
    Ok(Lanes {
        width,
        count,
        signed,
    })
}

/// Validate scalar coefficient arithmetic and numeric lane reductions before hardware emission.
pub(crate) fn validate(graph: &ScGraph, op: &ScOp) -> Result<(), String> {
    match op {
        ScOp::Scale { input, factor, .. }
        | ScOp::Offset {
            input,
            offset: factor,
            ..
        } => {
            coefficient(*factor)?;
            if lanes(graph, *input)?.count != 1 {
                return Err("Scale and Offset require a numeric scalar".into());
            }
        }
        ScOp::Reduce { input, .. } => {
            lanes(graph, *input)?;
        }
        _ => {}
    }
    Ok(())
}

/// Quantize a finite coefficient toward zero to the retained signed Q8.8 contract.
pub(crate) fn coefficient(value: f64) -> Result<i64, String> {
    let raw = (value * 256.0).trunc();
    if !raw.is_finite() || raw < -2.0f64.powi(63) || raw >= 2.0f64.powi(63) {
        return Err("arithmetic coefficient must fit a finite signed Q8.8 word".into());
    }
    Ok(raw as i64)
}
