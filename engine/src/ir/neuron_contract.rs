// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Shared compound neuron admission

//! Validate state words and delay geometry before either hardware emitter forms dimensions.

use super::{graph::*, stream_contract::value_type};

/// Accept a declared fixed-point interpretation or an explicitly encoded signed raw word.
fn signed_word(g: &ScGraph, id: ValueId, width: u32, fraction: u32) -> Result<(), String> {
    let ty = value_type(g, id)?;
    let valid = match ty {
        ScType::FixedPoint { width: w, frac } => w == width && frac == fraction,
        ScType::SInt { width: w } => w == width,
        ScType::Rate => width == 16 && fraction == 8,
        _ => false,
    };
    if valid {
        Ok(())
    } else {
        Err(format!("{id} requires a signed {width}-bit Q{fraction} word or an explicit signed raw word, got {ty}"))
    }
}

/// Refuse unrepresentable LIF parameters and inconsistent runtime binary points.
pub(crate) fn lif(g: &ScGraph, operands: [ValueId; 4], p: &LifParams) -> Result<(), String> {
    if !(1..=64).contains(&p.data_width) || p.fraction >= p.data_width {
        return Err("LifStep requires a 1..64-bit word and a smaller fraction".into());
    }
    let limit = 1i128 << (p.data_width - 1);
    if [p.v_rest, p.v_reset, p.v_threshold]
        .iter()
        .any(|v| i128::from(*v) < -limit || i128::from(*v) >= limit)
    {
        return Err("LifStep rest, reset and threshold must fit its signed state word".into());
    }
    for operand in operands {
        signed_word(g, operand, p.data_width, p.fraction)?;
    }
    Ok(())
}

/// Refuse invalid delay addresses and ensure every dynamic tent operand shares its Q format.
pub(crate) fn dcls(g: &ScGraph, operands: [ValueId; 4], p: &DclsParams) -> Result<(), String> {
    let [spike, weights, centre, sigma] = operands;
    if !(2..=64).contains(&p.data_width) || p.fraction >= p.data_width || p.n_taps == 0 {
        return Err(
            "DclsLayer requires nonempty taps and a 2..64-bit fixed-point word with a representable positive sigma".into(),
        );
    }
    if p.delay_depth == 0
        || p.delay_depth > i32::MAX as usize
        || p.ptr_width == 0
        || p.ptr_width > 32
        || p.delay_depth as u64 > 1u64 << p.ptr_width
        || p.tap_offsets.len() != p.n_taps
        || p.tap_offsets
            .iter()
            .any(|offset| *offset as usize >= p.delay_depth)
    {
        return Err(
            "DclsLayer tap offsets and pointer width must address its nonempty delay line".into(),
        );
    }
    if !matches!(
        value_type(g, spike)?,
        ScType::Bool | ScType::Bitstream { .. }
    ) {
        return Err("DclsLayer requires one Boolean or temporal spike bit".into());
    }
    signed_word(g, centre, p.data_width, p.fraction)?;
    signed_word(g, sigma, p.data_width, p.fraction)?;
    let ty = value_type(g, weights)?;
    let ScType::Vec { element, count } = ty else {
        return Err("DclsLayer requires a packed vector of signed tap weights".into());
    };
    let compatible = match *element {
        ScType::FixedPoint { width, frac } => width == p.data_width && frac == p.fraction,
        ScType::SInt { width } => width == p.data_width,
        ScType::Rate => p.data_width == 16 && p.fraction == 8,
        _ => false,
    };
    if count != p.n_taps
        || !compatible
        || p.n_taps
            .checked_mul(p.data_width as usize)
            .filter(|w| *w <= i32::MAX as usize)
            .is_none()
    {
        return Err("DclsLayer weight lanes must match its tap count and signed Q format".into());
    }
    Ok(())
}
