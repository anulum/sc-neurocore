// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Simple-spiking PyO3 binding composition

use pyo3::prelude::*;

mod balanced_resonate_and_fire;
mod butera_respiratory;
mod chay;
mod chay_keizer;
#[path = "../fitzhugh_nagumo.rs"]
mod fitzhugh_nagumo;
#[path = "../fitzhugh_rinzel.rs"]
mod fitzhugh_rinzel;
mod gutkin_ermentrout;
#[path = "../hindmarsh_rose.rs"]
mod hindmarsh_rose;
mod learnable_neuron_model;
#[path = "../mckean.rs"]
mod mckean;
mod morris_lecar;
#[path = "../pernarowski.rs"]
mod pernarowski;
#[path = "../resonate_and_fire.rs"]
mod resonate_and_fire;
mod sherman_rinzel_keizer;
#[path = "../terman_wang.rs"]
mod terman_wang;
#[path = "../wilson_hr.rs"]
mod wilson_hr;

/// Register default simple-spiking bindings that precede custom classes.
pub(crate) fn register_primary(module: &Bound<'_, PyModule>) -> PyResult<()> {
    fitzhugh_nagumo::register(module)?;
    morris_lecar::register(module)?;
    hindmarsh_rose::register(module)?;
    resonate_and_fire::register(module)?;
    balanced_resonate_and_fire::register(module)?;
    fitzhugh_rinzel::register(module)?;
    mckean::register(module)?;
    terman_wang::register(module)?;
    Ok(())
}

/// Register conductance-based defaults that follow the oscillator classes.
pub(crate) fn register_conductance_models(module: &Bound<'_, PyModule>) -> PyResult<()> {
    gutkin_ermentrout::register(module)?;
    wilson_hr::register(module)?;
    chay::register(module)?;
    chay_keizer::register(module)?;
    sherman_rinzel_keizer::register(module)?;
    butera_respiratory::register(module)?;
    Ok(())
}

/// Register the default trainable and beta-cell model bindings.
pub(crate) fn register_tail(module: &Bound<'_, PyModule>) -> PyResult<()> {
    learnable_neuron_model::register(module)?;
    pernarowski::register(module)?;
    Ok(())
}
