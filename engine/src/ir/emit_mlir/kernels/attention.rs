// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Scaled dot-product attention CIRCT lowering

//! Preserve stable row maxima, the maintained exp grid and deferred integer normalization.

use super::{constant, mac, pack, values, Net, Value};
use crate::ir::graph::{ScGraph, ValueId};

/// Implement the existing 256-entry exp table over the symmetric one-eighth grid.
fn exponential(n: &mut Net<'_>, argument: &Value) -> Value {
    let argument = n.resize(argument, 25, true);
    let offset = n.constant(1048576, 25, true);
    let shifted = n.binary("add", &argument, &offset);
    let shift = n.constant(13, 25, false);
    let raw = n.binary("shrs", &shifted, &shift);
    let zero = n.constant(0, 25, true);
    let last = n.constant(255, 25, true);
    let below = n.compare("slt", &raw, &zero);
    let above = n.compare("sgt", &raw, &last);
    let bounded_low = n.select(&below, &zero, &raw);
    let bounded = n.select(&above, &last, &bounded_low);
    let index = n.resize(&bounded, 8, false);
    let table = (0..256)
        .map(|i| (((-16.0 + f64::from(i) / 8.0).exp() * 65536.0).round() as i64).min(8388607))
        .collect::<Vec<_>>();
    n.lookup(&index, &table, 24)
}

/// Lower rectangular Q/K/V geometry using the same signed Q8.16 kernel contract.
pub(super) fn emit(
    text: &mut String,
    g: &ScGraph,
    id: ValueId,
    operands: [ValueId; 3],
    dim: usize,
) -> Result<(), String> {
    if dim == 0 {
        return Err("SoftmaxAttention requires a positive key dimension".into());
    }
    let [q, k, v] = operands;
    let q = values(g, q, "SoftmaxAttention queries")?;
    let k = values(g, k, "SoftmaxAttention keys")?;
    let v = values(g, v, "SoftmaxAttention values")?;
    if q.is_empty() || k.is_empty() || q.len() % dim != 0 || k.len() % dim != 0 {
        return Err(
            "SoftmaxAttention query and key lengths must contain nonempty complete rows".into(),
        );
    }
    let queries = q.len() / dim;
    let keys = k.len() / dim;
    if v.is_empty() || v.len() % keys != 0 {
        return Err("SoftmaxAttention values must contain one row per key".into());
    }
    let columns = v.len() / keys;
    let mut n = Net::new(text, id.0);
    let q = q
        .iter()
        .map(|v| constant(&mut n, *v))
        .collect::<Result<Vec<_>, _>>()?;
    let k = k
        .iter()
        .map(|v| constant(&mut n, *v))
        .collect::<Result<Vec<_>, _>>()?;
    let v = v
        .iter()
        .map(|v| constant(&mut n, *v))
        .collect::<Result<Vec<_>, _>>()?;
    let inverse_temperature = constant(&mut n, 1.0 / (dim as f64).sqrt())?;
    let inverse_temperature = n.resize(&inverse_temperature, 80, true);
    let mut output = Vec::new();
    for query in 0..queries {
        let mut scores = Vec::new();
        for key in 0..keys {
            let mut dot = n.constant(0, 56, true);
            for d in 0..dim {
                dot = mac(&mut n, &dot, &q[query * dim + d], &k[key * dim + d]);
            }
            let dot = n.resize(&dot, 80, true);
            let product = n.binary("mul", &dot, &inverse_temperature);
            let raw = n.extract(&product, 32, 24);
            scores.push(Value {
                signed: true,
                ..raw
            });
        }
        let mut maximum = scores[0].clone();
        for score in &scores[1..] {
            let greater = n.compare("sgt", score, &maximum);
            maximum = n.select(&greater, score, &maximum);
        }
        let mut exponentials = Vec::new();
        let mut sum = n.constant(0, 32, true);
        for score in &scores {
            let difference = n.binary("sub", score, &maximum);
            let value = exponential(&mut n, &difference);
            let extended = n.resize(&value, 32, true);
            sum = n.binary("add", &sum, &extended);
            exponentials.push(value);
        }
        let sum = n.resize(&sum, 80, true);
        let shift = n.constant(16, 80, false);
        let mut weights = Vec::new();
        for exponential in &exponentials {
            let exponential = n.resize(exponential, 80, true);
            let numerator = n.binary("shl", &exponential, &shift);
            let quotient = n.binary("divs", &numerator, &sum);
            weights.push(n.resize(&quotient, 24, true));
        }
        for column in 0..columns {
            let mut accumulator = n.constant(0, 56, true);
            for key in 0..keys {
                accumulator = mac(
                    &mut n,
                    &accumulator,
                    &weights[key],
                    &v[key * columns + column],
                );
            }
            let result = n.extract(&accumulator, 16, 24);
            output.push(Value {
                signed: true,
                ..result
            });
        }
    }
    pack(&mut n, id, output);
    Ok(())
}
