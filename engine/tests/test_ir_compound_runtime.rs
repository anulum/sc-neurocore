// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Compound circuits against independent integer trajectories

//! Compare public graph outputs, including composed operations, through real RTL.

#[path = "common/ir_stream.rs"]
mod ir_stream;

/// Interpret physical words without depending on compiler arithmetic helpers.
fn signed(raw: u128, width: u32) -> i128 {
    ((raw << (128 - width)) as i128) >> (128 - width)
}

/// Produce reproducible full-width operands without a platform RNG contract.
fn random(state: &mut u128, width: u32) -> u128 {
    *state ^= *state << 23;
    *state ^= *state >> 17;
    *state ^= *state << 26;
    *state & (u128::MAX >> (128 - width))
}

/// Compare padded hexadecimal observations, keeping unknown RTL bits visible.
fn observation(value: u128, width: u32) -> String {
    let value = value & (u128::MAX >> (128 - width));
    format!("OBS {value:0digits$x}", digits = width.div_ceil(4) as usize)
}

#[path = "common/compound_arithmetic.rs"]
mod arithmetic;
#[path = "common/compound_kernels.rs"]
mod kernels;
#[path = "common/compound_neurons.rs"]
mod neurons;
