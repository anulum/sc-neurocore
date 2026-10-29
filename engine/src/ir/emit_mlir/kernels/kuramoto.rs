// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Quantized Kuramoto phase-step CIRCT lowering

//! Preserve the 64-entry sine grid, deferred coupling rounding and single-modulus wrap.

use super::{constant, mac, pack, values, Net, Value};
use crate::ir::graph::{ScGraph, ValueId};

/// Apply the maintained core's one-modulus phase or signed-difference correction.
fn wrap(n: &mut Net<'_>, value: &Value, delta: bool) -> Value {
    let modulus = n.constant(411775, 24, true);
    let upper = n.constant(if delta { 205887 } else { 411775 }, 24, true);
    let lower = n.constant(if delta { -205887 } else { 0 }, 24, true);
    let above = n.compare(if delta { "sgt" } else { "sge" }, value, &upper);
    let below = n.compare("slt", value, &lower);
    let forward = n.binary("sub", value, &modulus);
    let backward = n.binary("add", value, &modulus);
    let low = n.select(&below, &backward, value);
    n.select(&above, &forward, &low)
}

/// Evaluate the same quantized sine lookup on a wrapped signed phase difference.
fn sine(n: &mut Net<'_>, difference: &Value) -> Value {
    let phase = wrap(n, difference, false);
    let phase = n.resize(&phase, 32, true);
    let size = n.constant(64, 32, true);
    let modulus = n.constant(411775, 32, true);
    let scaled = n.binary("mul", &phase, &size);
    let index = n.binary("divs", &scaled, &modulus);
    let index = n.resize(&index, 6, false);
    let table = (0..64)
        .map(|i| ((std::f64::consts::TAU * f64::from(i) / 64.0).sin() * 65536.0).round() as i64)
        .collect::<Vec<_>>();
    n.lookup(&index, &table, 24)
}

/// Emit the full pairwise coupling matrix and one deterministic Euler phase update.
pub(super) fn emit(
    text: &mut String,
    g: &ScGraph,
    id: ValueId,
    operands: [ValueId; 3],
    dt: f64,
) -> Result<(), String> {
    let [phases, omega, coupling] = operands;
    let phases = values(g, phases, "KuramotoStep phases")?;
    let omega = values(g, omega, "KuramotoStep omega")?;
    let coupling = values(g, coupling, "KuramotoStep coupling")?;
    let count = phases.len();
    if count == 0 || omega.len() != count || count.checked_mul(count) != Some(coupling.len()) {
        return Err(
            "KuramotoStep requires nonempty phase/omega vectors and a square coupling matrix"
                .into(),
        );
    }
    let mut n = Net::new(text, id.0);
    let phases = phases
        .iter()
        .map(|v| constant(&mut n, v.rem_euclid(std::f64::consts::TAU)))
        .collect::<Result<Vec<_>, _>>()?;
    let omega = omega
        .iter()
        .map(|v| constant(&mut n, *v))
        .collect::<Result<Vec<_>, _>>()?;
    let coupling = coupling
        .iter()
        .map(|v| constant(&mut n, *v))
        .collect::<Result<Vec<_>, _>>()?;
    let dt = constant(&mut n, dt)?;
    let dt = n.resize(&dt, 48, true);
    let mut output = Vec::new();
    for i in 0..count {
        let mut accumulator = n.constant(0, 56, true);
        for j in 0..count {
            let difference = n.binary("sub", &phases[j], &phases[i]);
            let difference = wrap(&mut n, &difference, true);
            let sine = sine(&mut n, &difference);
            accumulator = mac(&mut n, &accumulator, &coupling[i * count + j], &sine);
        }
        let term = n.extract(&accumulator, 16, 24);
        let velocity = n.binary("add", &omega[i], &term);
        let velocity = Value {
            signed: true,
            ..velocity
        };
        let velocity = n.resize(&velocity, 48, true);
        let product = n.binary("mul", &velocity, &dt);
        let delta = n.extract(&product, 16, 24);
        let next = n.binary("add", &phases[i], &delta);
        output.push(wrap(&mut n, &next, false));
    }
    pack(&mut n, id, output);
    Ok(())
}
