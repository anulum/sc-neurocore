// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Complete stochastic matrix layer CIRCT lowering

//! Shared input encoders, row-major independent weight encoders and signed current mapping.

use super::{
    lif,
    ops::{Net, Value},
    value_wire,
};
use crate::ir::{
    dense_contract::{probability_layout, value_type},
    graph::*,
};

/// Convert a declared signed fixed-point probability to the existing strict encoder threshold.
fn threshold(n: &mut Net<'_>, raw: &Value, fraction: u32) -> Value {
    let zero_raw = n.constant(0, raw.width, true);
    let nonpositive = n.compare("sle", raw, &zero_raw);
    let extended = n.resize(raw, raw.width + 17, true);
    let scale = n.constant(65535, extended.width, true);
    let product = n.binary("mul", &extended, &scale);
    let fractional = n.extract(&product, fraction as usize, 16);
    let positive = if fraction as usize + 1 < raw.width {
        let unit = n.constant(1i64 << fraction, raw.width, true);
        let full = n.constant(65535, 16, false);
        let at_least_one = n.compare("sge", raw, &unit);
        n.select(&at_least_one, &full, &fractional)
    } else {
        fractional
    };
    let zero = n.constant(0, 16, false);
    n.select(&nonpositive, &zero, &positive)
}

/// Register the same maximal-period LFSR and strict comparison as sc_bitstream_encoder.
fn encoder(n: &mut Net<'_>, threshold: &Value, seed: u16) -> Value {
    let reset = n.constant(i64::from(seed), 16, false);
    let false_bit = n.constant(0, 1, false);
    let (lfsr, next_lfsr) = n.register(&reset);
    let (sample, next_sample) = n.register(&false_bit);
    let taps = [15, 13, 12, 10].map(|bit| n.extract(&lfsr, bit, 1));
    let mut feedback = taps[0].clone();
    for tap in &taps[1..] {
        feedback = n.binary("xor", &feedback, tap);
    }
    let low = n.extract(&lfsr, 0, 15);
    let next = n.concat(&[low, feedback]);
    let comparison = n.compare("ult", &lfsr, threshold);
    n.bind(&next_lfsr, &next);
    n.bind(&next_sample, &comparison);
    sample
}

/// Expand a probability vector into decorrelated registered samples, lane zero at the LSB.
fn samples(
    n: &mut Net<'_>,
    g: &ScGraph,
    input: ValueId,
    count: usize,
    seed: u16,
    stride: u16,
) -> Result<Vec<Value>, String> {
    let ty = value_type(g, input)?;
    let ScType::Vec { element, .. } = ty else {
        return Err("dense encoder requires a probability vector".into());
    };
    let layout = probability_layout(&element)?;
    let source = Value {
        name: value_wire(g, input),
        width: layout.width as usize * count,
        signed: true,
    };
    let mut result = Vec::new();
    for i in 0..count {
        let lane = n.extract(&source, i * layout.width as usize, layout.width as usize);
        let lane = Value {
            signed: true,
            ..lane
        };
        let threshold = threshold(n, &lane, layout.fraction);
        let seed = seed.wrapping_add((i as u16).wrapping_mul(stride));
        result.push(encoder(n, &threshold, seed));
    }
    Ok(result)
}

/// Map a row's positive population count through either signed endpoint order without overflow.
fn current(n: &mut Net<'_>, bits: &[Value], p: &DenseParams) -> Value {
    let count_width = usize::BITS as usize - p.n_inputs.leading_zeros() as usize;
    let product_width = p.data_width as usize + count_width + 2;
    let mut count = n.constant(0, count_width, false);
    for bit in bits {
        let extended = n.resize(bit, count_width, false);
        count = n.binary("add", &count, &extended);
    }
    let count = n.resize(&count, product_width, true);
    let min = n.constant(p.y_min, p.data_width as usize, true);
    let max = n.constant(p.y_max, p.data_width as usize, true);
    let min_wide = n.resize(&min, product_width, true);
    let max = n.resize(&max, product_width, true);
    let range = n.binary("sub", &max, &min_wide);
    let product = n.binary("mul", &range, &count);
    let divisor = n.constant(p.n_inputs as i64, product_width, true);
    let scaled = n.binary("divs", &product, &divisor);
    let scaled = n.resize(&scaled, p.data_width as usize, true);
    n.binary("add", &min, &scaled)
}

/// Emit the observable matrix-layer spike bus with all runtime inputs and configuration.
pub(super) fn emit(
    text: &mut String,
    g: &ScGraph,
    id: ValueId,
    operands: [ValueId; 4],
    p: &DenseParams,
) -> Result<(), String> {
    let [inputs, weights, leak, gain] = operands;
    // Public graph verification has already checked dimensions, signed state words and seeds.
    let mut n = Net::new(text, id.0);
    let inputs = samples(&mut n, g, inputs, p.n_inputs, p.input_seed_base, 7)?;
    let weights = samples(
        &mut n,
        g,
        weights,
        p.n_inputs * p.n_neurons,
        p.weight_seed_base,
        13,
    )?;
    let leak = Value {
        name: value_wire(g, leak),
        width: p.data_width as usize,
        signed: true,
    };
    let gain = Value {
        name: value_wire(g, gain),
        width: p.data_width as usize,
        signed: true,
    };
    let noise = n.constant(0, p.data_width as usize, true);
    let neuron = LifParams {
        data_width: p.data_width,
        fraction: 8,
        v_rest: 0,
        v_reset: 0,
        v_threshold: 256,
        refractory_period: 0,
    };
    let mut spikes = Vec::new();
    for row in weights.chunks_exact(p.n_inputs) {
        let mut bits = Vec::new();
        for (input, weight) in inputs.iter().zip(row) {
            bits.push(n.binary("and", input, weight));
        }
        let current = current(&mut n, &bits, p);
        spikes.push(lif::lower(
            &mut n,
            &[current, leak.clone(), gain.clone(), noise.clone()],
            &neuron,
        ));
    }
    spikes.reverse();
    let result = if spikes.len() == 1 {
        spikes[0].clone()
    } else {
        n.concat(&spikes)
    };
    n.bind(&format!("%v{}", id.0), &result);
    Ok(())
}
