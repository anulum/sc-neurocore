// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Degree-normalized graph aggregation lowering

//! Preserve signed row degrees, deferred rounding and zero-degree cancellation.

use super::{constant, mac, pack, values, Net};
use crate::ir::graph::{ScGraph, ValueId};

/// Emit every node/feature accumulator with the maintained 56-bit product headroom.
pub(super) fn emit(
    text: &mut String,
    g: &ScGraph,
    id: ValueId,
    operands: [ValueId; 2],
    nodes: usize,
    features: usize,
) -> Result<(), String> {
    if nodes == 0 || features == 0 {
        return Err("GraphForward requires positive node and feature counts".into());
    }
    let [input, adjacency] = operands;
    let input = values(g, input, "GraphForward features")?;
    let adjacency = values(g, adjacency, "GraphForward adjacency")?;
    if nodes.checked_mul(features) != Some(input.len())
        || nodes.checked_mul(nodes) != Some(adjacency.len())
    {
        return Err("GraphForward constant lengths must match its node/feature geometry".into());
    }
    let mut n = Net::new(text, id.0);
    let input = input
        .iter()
        .map(|v| constant(&mut n, *v))
        .collect::<Result<Vec<_>, _>>()?;
    let adjacency = adjacency
        .iter()
        .map(|v| constant(&mut n, *v))
        .collect::<Result<Vec<_>, _>>()?;
    let zero_degree = n.constant(0, 32, true);
    let one = n.constant(1, 56, true);
    let shift = n.constant(16, 56, false);
    let mut result = Vec::new();
    for row in 0..nodes {
        let mut degree = zero_degree.clone();
        for edge in &adjacency[row * nodes..(row + 1) * nodes] {
            let edge = n.resize(edge, 32, true);
            degree = n.binary("add", &degree, &edge);
        }
        let empty = n.compare("eq", &degree, &zero_degree);
        let degree = n.resize(&degree, 56, true);
        let divisor = n.select(&empty, &one, &degree);
        for feature in 0..features {
            let mut accumulator = n.constant(0, 56, true);
            for node in 0..nodes {
                accumulator = mac(
                    &mut n,
                    &accumulator,
                    &adjacency[row * nodes + node],
                    &input[node * features + feature],
                );
            }
            let normalized = n.binary("divs", &accumulator, &divisor);
            let unnormalized = n.binary("shrs", &accumulator, &shift);
            let word = n.select(&empty, &unnormalized, &normalized);
            result.push(n.resize(&word, 24, true));
        }
    }
    pack(&mut n, id, result);
    Ok(())
}
