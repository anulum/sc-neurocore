// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Delay-coded tent layer lowering

//! Fixed tap offsets share an equivalent asynchronously reset spike history.

use super::{
    ops::{Net, Value},
    value_wire,
};
use crate::ir::graph::*;

/// Preserve the complete delayed tent sum until saturation to the declared output word.
pub(super) fn emit(
    text: &mut String,
    graph: &ScGraph,
    id: ValueId,
    operands: [ValueId; 4],
    p: &DclsParams,
) {
    let [spike, weights, centre, sigma] = operands;
    let w = p.data_width as usize;
    let count_bits = usize::BITS as usize - (p.n_taps - 1).leading_zeros() as usize;
    let wide = (w + p.fraction as usize + count_bits + 4).max(64);
    let spike = Value {
        name: value_wire(graph, spike),
        width: 1,
        signed: false,
    };
    let weights = Value {
        name: value_wire(graph, weights),
        width: w * p.n_taps,
        signed: true,
    };
    let centre = Value {
        name: value_wire(graph, centre),
        width: w,
        signed: true,
    };
    let sigma = Value {
        name: value_wire(graph, sigma),
        width: w,
        signed: true,
    };
    let mut n = Net::new(text, id.0);
    let zero_bit = n.constant(0, 1, false);
    let zero = n.constant(0, wide, true);
    let one = n.constant(1, wide, true);
    let shift = n.constant(i64::from(p.fraction), wide, false);
    let unit = n.binary("shl", &one, &shift);
    let centre = n.resize(&centre, wide, true);
    let sigma = n.resize(&sigma, wide, true);
    let valid_sigma = n.compare("sgt", &sigma, &zero);
    let divisor = n.select(&valid_sigma, &sigma, &one);
    let mut history = vec![spike.clone()];
    let depth = p.tap_offsets.iter().copied().max().unwrap_or(0);
    for i in 0..depth as usize {
        let (state, next) = n.register(&zero_bit);
        n.bind(&next, &history[i]);
        history.push(state);
    }
    let mut accumulator = zero.clone();
    for (i, offset) in p.tap_offsets.iter().enumerate() {
        let raw_weight = n.extract(&weights, i * w, w);
        let weight = n.resize(
            &Value {
                signed: true,
                ..raw_weight
            },
            wide,
            true,
        );
        let delay = n.constant(i as i64, wide, true);
        let delay = n.binary("shl", &delay, &shift);
        let after = n.compare("sge", &delay, &centre);
        let forward = n.binary("sub", &delay, &centre);
        let backward = n.binary("sub", &centre, &delay);
        let distance = n.select(&after, &forward, &backward);
        let inside = n.compare("slt", &distance, &sigma);
        let active = n.binary("and", &inside, &history[*offset as usize]);
        let numerator = n.binary("sub", &sigma, &distance);
        let numerator = n.binary("shl", &numerator, &shift);
        let quotient = n.binary("divs", &numerator, &divisor);
        let above_unit = n.compare("sgt", &quotient, &unit);
        let gate = n.select(&above_unit, &unit, &quotient);
        let contribution = n.binary("mul", &weight, &gate);
        let contribution = n.select(&active, &contribution, &zero);
        accumulator = n.binary("add", &accumulator, &contribution);
    }
    let max = n.constant(((1u128 << (w - 1)) - 1) as i64, w, true);
    let min = n.constant(-(1i128 << (w - 1)) as i64, w, true);
    let upper_word = n.resize(&max, wide, true);
    let lower_word = n.resize(&min, wide, true);
    let upper = n.binary("shl", &upper_word, &shift);
    let lower = n.binary("shl", &lower_word, &shift);
    let above = n.compare("sgt", &accumulator, &upper);
    let below = n.compare("slt", &accumulator, &lower);
    let result = n.extract(&accumulator, p.fraction as usize, w);
    let limited_high = n.select(&above, &max, &result);
    let limited = n.select(&below, &min, &limited_high);
    let zero = n.constant(0, w, true);
    let result = n.select(&valid_sigma, &limited, &zero);
    n.bind(&format!("%v{}", id.0), &result);
}
