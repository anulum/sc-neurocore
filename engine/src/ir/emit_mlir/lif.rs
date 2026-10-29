// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Stateful LIF lowering to HW/Comb/Seq

//! Preserve sc_lif_neuron's signed products, wrap points, reset and refractory cycles.

use super::{
    ops::{Net, Value},
    value_wire,
};
use crate::ir::graph::*;

/// Lower all four verified runtime operands to a Boolean spike; output-port padding is separate.
pub(super) fn emit(
    text: &mut String,
    graph: &ScGraph,
    id: ValueId,
    operands: [ValueId; 4],
    p: &LifParams,
) {
    let w = p.data_width as usize;
    let mut wires = Vec::new();
    for operand in operands {
        wires.push(Value {
            name: value_wire(graph, operand),
            width: w,
            signed: true,
        });
    }
    let mut n = Net::new(text, id.0);
    let spike = lower(&mut n, &wires, p);
    n.bind(&format!("%v{}_spike", id.0), &spike);
}

/// Build one register bank for already validated signed current, leak, gain and noise words.
pub(super) fn lower(n: &mut Net<'_>, wires: &[Value], p: &LifParams) -> Value {
    let w = p.data_width as usize;
    let rest = n.constant(p.v_rest, w, true);
    let reset = n.constant(p.v_reset, w, true);
    let threshold = n.constant(p.v_threshold, w, true);
    let zero = n.constant(0, 32, false);
    let one = n.constant(1, 32, false);
    let period = n.constant(i64::from(p.refractory_period), 32, false);
    let false_bit = n.constant(0, 1, false);
    let (membrane, next_membrane) = n.register(&rest);
    let (counter, next_counter) = n.register(&zero);
    let (spike, next_spike) = n.register(&false_bit);
    let rest_wide = n.resize(&rest, 2 * w, true);
    let membrane_wide = n.resize(&membrane, 2 * w, true);
    let delta = n.binary("sub", &rest_wide, &membrane_wide);
    let leak = n.resize(&wires[1], 2 * w, true);
    let leak_product = n.binary("mul", &delta, &leak);
    let leak_increment = n.extract(&leak_product, p.fraction as usize, w);
    let current = n.resize(&wires[0], 2 * w, true);
    let gain = n.resize(&wires[2], 2 * w, true);
    let input_product = n.binary("mul", &current, &gain);
    let input_increment = n.extract(&input_product, p.fraction as usize, w);
    let leaking = n.binary("add", &membrane, &leak_increment);
    let driven = n.binary("add", &leaking, &input_increment);
    let potential = n.binary("add", &driven, &wires[3]);
    let crossing = n.compare("sge", &potential, &threshold);
    let refractory = n.compare("ugt", &counter, &zero);
    let candidate = n.select(&crossing, &reset, &potential);
    let next_v = n.select(&refractory, &rest, &candidate);
    let counted_down = n.binary("sub", &counter, &one);
    let loaded = n.select(&crossing, &period, &counter);
    let next_c = n.select(&refractory, &counted_down, &loaded);
    let next_s = n.select(&refractory, &false_bit, &crossing);
    n.bind(&next_membrane, &next_v);
    n.bind(&next_counter, &next_c);
    n.bind(&next_spike, &next_s);
    spike
}
