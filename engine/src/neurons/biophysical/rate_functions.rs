// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Shared Hodgkin-Huxley rate functions

//! Shared rate functions of the conductance-based neuron models.

#![deny(missing_docs)]

/// Hodgkin-Huxley-type rate `number_factor * d / (1 - exp(-d / denom_scale))` with
/// `d = v + v_offset`.
///
/// The expression is 0/0 at `d = 0`; within `1e-7` of it `fallback` is returned instead.
/// Every caller in the engine passes the limit `number_factor * denom_scale` there.
pub fn safe_rate(
    number_factor: f64,
    v_offset: f64,
    v: f64,
    denom_scale: f64,
    fallback: f64,
) -> f64 {
    let d = v + v_offset;
    if d.abs() < 1e-7 {
        fallback
    } else {
        number_factor * d / (1.0 - (-d / denom_scale).exp())
    }
}
