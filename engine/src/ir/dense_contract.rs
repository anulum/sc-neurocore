// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Dense hardware operand and parameter contracts

//! Dense matrix geometry, signed state words, and probability input contracts.

use super::constant_words::{scalar_layout, ScalarLayout};
use super::graph::*;

/// Find a producer's declared type; untyped arithmetic is not a probability word.
pub(crate) fn value_type(graph: &ScGraph, id: ValueId) -> Result<&ScType, String> {
    graph
        .ops
        .iter()
        .find_map(|op| match op {
            ScOp::Input { id: result, ty, .. } | ScOp::Constant { id: result, ty, .. }
                if *result == id =>
            {
                Some(ty)
            }
            ScOp::Decode { id: result, .. } if *result == id => Some(&ScType::Rate),
            _ => None,
        })
        .ok_or_else(|| format!("{id} needs a declared numeric input or constant type"))
}

/// Resolve signed probability words, including Rate's Q8.8 representation.
pub(crate) fn probability_layout(ty: &ScType) -> Result<ScalarLayout, String> {
    match ty {
        ScType::Rate | ScType::FixedPoint { .. } => scalar_layout(ty),
        _ => Err(format!(
            "probability requires rate or signed fixed-point, got {ty}"
        )),
    }
}

fn probability_vector(graph: &ScGraph, id: ValueId, count: usize) -> Result<(), String> {
    match value_type(graph, id)? {
        ScType::Vec {
            element,
            count: actual,
        } if *actual == count => {
            probability_layout(element)?;
            Ok(())
        }
        actual => Err(format!(
            "{id} requires a probability vector of length {count}, got {actual}"
        )),
    }
}

fn state_word(graph: &ScGraph, id: ValueId, width: u32) -> Result<(), String> {
    let ty = value_type(graph, id)?;
    match ty {
        ScType::FixedPoint {
            width: actual,
            frac: 8,
        }
        | ScType::SInt { width: actual }
            if *actual == width =>
        {
            Ok(())
        }
        ScType::Rate if width == 16 => Ok(()),
        _ => Err(format!(
            "{id} requires a {width}-bit signed Q8 state word, got {ty}"
        )),
    }
}

fn seed_sequence(base: u16, stride: u16, count: usize) -> Result<(), String> {
    if count > usize::from(u16::MAX)
        || (0..count).any(|i| base.wrapping_add((i as u16).wrapping_mul(stride)) == 0)
    {
        return Err("encoder seed sequence contains the absorbing zero LFSR state".into());
    }
    Ok(())
}

/// Validate before emitting any ports, dimensions, or narrowed current literals.
pub(crate) fn validate_dense(
    graph: &ScGraph,
    inputs: ValueId,
    weights: ValueId,
    leak: ValueId,
    gain: ValueId,
    p: &DenseParams,
) -> Result<(), String> {
    if p.n_inputs == 0 || p.n_neurons == 0 {
        return Err("DenseForward requires positive input and neuron counts".into());
    }
    let weights_count = p
        .n_inputs
        .checked_mul(p.n_neurons)
        .filter(|n| *n <= i32::MAX as usize / 16)
        .ok_or_else(|| {
            "DenseForward matrix bus width exceeds Verilog indexing range".to_string()
        })?;
    if !(10..=64).contains(&p.data_width) {
        return Err(
            "DenseForward signed Q8 state width must be 10..64 to represent threshold 256".into(),
        );
    }
    if p.stream_length == 0 || p.stream_length > u32::MAX as usize {
        return Err("DenseForward stream length must be 1..4294967295".into());
    }
    let limit = 1_i128 << (p.data_width - 1);
    if [p.y_min, p.y_max]
        .iter()
        .any(|v| i128::from(*v) < -limit || i128::from(*v) >= limit)
    {
        return Err(format!(
            "DenseForward current endpoints must fit signed {}-bit words",
            p.data_width
        ));
    }
    probability_vector(graph, inputs, p.n_inputs)?;
    probability_vector(graph, weights, weights_count)?;
    state_word(graph, leak, p.data_width)?;
    state_word(graph, gain, p.data_width)?;
    seed_sequence(p.input_seed_base, 7, p.n_inputs)?;
    seed_sequence(p.weight_seed_base, 13, weights_count)
}
