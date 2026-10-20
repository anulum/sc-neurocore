// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Typed constant word conversion

//! Scalar layouts and exact-width constant admission shared by verification and emission.

use super::graph::{ScConst, ScType};

/// Hardware representation of one declared scalar.
#[derive(Clone, Copy)]
pub(crate) struct ScalarLayout {
    /// Total number of bits in each encoded word.
    pub width: u32,
    /// Binary point for floating constants and probability conversion.
    pub fraction: u32,
    /// Whether the raw word uses two's-complement signed interpretation.
    pub signed: bool,
}

/// Resolve a scalar type without silently substituting a default word.
pub(crate) fn scalar_layout(ty: &ScType) -> Result<ScalarLayout, String> {
    let (width, fraction, signed) = match ty {
        ScType::Rate => (16, 8, true),
        ScType::FixedPoint { width, frac } => (*width, *frac, true),
        ScType::SInt { width } => (*width, 0, true),
        ScType::UInt { width } => (*width, 0, false),
        ScType::Bool => (1, 0, false),
        _ => return Err(format!("expected a numeric scalar, got {ty}")),
    };
    if !(1..=64).contains(&width) || fraction >= width {
        return Err(format!(
            "invalid scalar layout {ty}: width must be 1..64 and fraction < width"
        ));
    }
    Ok(ScalarLayout {
        width,
        fraction,
        signed,
    })
}

impl ScalarLayout {
    fn integer(self, value: i128) -> Result<u64, String> {
        let (min, max) = if self.signed {
            (
                -(1_i128 << (self.width - 1)),
                (1_i128 << (self.width - 1)) - 1,
            )
        } else {
            (0, (1_i128 << self.width) - 1)
        };
        if value < min || value > max {
            return Err(format!(
                "constant word {value} does not fit {} bits (signed={})",
                self.width, self.signed
            ));
        }
        Ok((value & ((1_i128 << self.width) - 1)) as u64)
    }

    fn float(self, value: f64) -> Result<u64, String> {
        let scaled = (value * 2_f64.powi(self.fraction as i32)).trunc();
        let upper = 2_f64.powi((self.width - u32::from(self.signed)) as i32);
        let lower = if self.signed { -upper } else { 0.0 };
        if !scaled.is_finite() || scaled < lower || scaled >= upper {
            return Err(format!(
                "constant {value} is not representable in the declared scalar layout"
            ));
        }
        self.integer(scaled as i128)
    }

    /// Render a validated raw word as a sized Verilog literal.
    pub(crate) fn literal(self, word: u64) -> String {
        if self.signed {
            let signed = if word & (1_u64 << (self.width - 1)) == 0 {
                i128::from(word)
            } else {
                i128::from(word) - (1_i128 << self.width)
            };
            if signed < 0 {
                format!("-{}'sd{}", self.width, -signed)
            } else {
                format!("{}'sd{}", self.width, signed)
            }
        } else {
            format!("{}'d{word}", self.width)
        }
    }
}

/// Quantize floats toward zero; integer constants already denote raw words.
pub(crate) fn constant_words(
    value: &ScConst,
    ty: &ScType,
) -> Result<(ScalarLayout, Vec<u64>), String> {
    let (element, count, vector) = match ty {
        ScType::Vec { element, count } => (element.as_ref(), *count, true),
        _ => (ty, 1, false),
    };
    let layout = scalar_layout(element)?;
    count
        .checked_mul(layout.width as usize)
        .filter(|bits| *bits <= i32::MAX as usize)
        .ok_or_else(|| "constant bus width exceeds Verilog indexing range".to_string())?;
    let words = match (value, vector) {
        (ScConst::F64(value), false) => vec![layout.float(*value)?],
        (ScConst::I64(value), false) => vec![layout.integer(i128::from(*value))?],
        (ScConst::U64(value), false) => vec![layout.integer(i128::from(*value))?],
        (ScConst::F64Vec(values), true) => values
            .iter()
            .map(|v| layout.float(*v))
            .collect::<Result<Vec<_>, _>>()?,
        (ScConst::I64Vec(values), true) => values
            .iter()
            .map(|v| layout.integer(i128::from(*v)))
            .collect::<Result<Vec<_>, _>>()?,
        _ => return Err(format!("constant scalar/vector shape does not match {ty}")),
    };
    if count == 0 || words.len() != count {
        return Err(format!(
            "constant length {} does not match nonempty declared shape {ty}",
            words.len()
        ));
    }
    Ok((layout, words))
}
