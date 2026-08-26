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
#[path = "../sc_triangular_mckean.rs"]
mod sc_triangular_mckean;
mod sc_unit_capacitance_respiratory;
mod sherman_rinzel_keizer;
#[path = "../terman_wang.rs"]
mod terman_wang;
#[path = "../wilson_hr.rs"]
mod wilson_hr;

pub use balanced_resonate_and_fire::PyBalancedResonateAndFireNeuron;
pub use butera_respiratory::PyButeraRespiratoryNeuron;
pub use chay::PyChayNeuron;
pub use chay_keizer::PyChayKeizerNeuron;
pub use fitzhugh_nagumo::PyFitzHughNagumoNeuron;
pub use fitzhugh_rinzel::PyFitzHughRinzelNeuron;
pub use gutkin_ermentrout::PyGutkinErmentroutNeuron;
pub use hindmarsh_rose::PyHindmarshRoseNeuron;
pub use learnable_neuron_model::PyLearnableNeuronModel;
pub use mckean::PyMcKeanNeuron;
pub use morris_lecar::PyMorrisLecarNeuron;
pub use pernarowski::PyPernarowskiNeuron;
pub use resonate_and_fire::PyResonateAndFireNeuron;
pub use sc_triangular_mckean::PySCTriangularMcKeanNeuron;
pub use sc_unit_capacitance_respiratory::PySCUnitCapacitanceRespiratoryNeuron;
pub use sherman_rinzel_keizer::PyShermanRinzelKeizerNeuron;
pub use terman_wang::PyTermanWangOscillator;
pub use wilson_hr::PyWilsonHRNeuron;

/// Register default simple-spiking bindings that precede custom classes.
pub(crate) fn register_primary(module: &Bound<'_, PyModule>) -> PyResult<()> {
    fitzhugh_nagumo::register(module)?;
    morris_lecar::register(module)?;
    hindmarsh_rose::register(module)?;
    resonate_and_fire::register(module)?;
    balanced_resonate_and_fire::register(module)?;
    fitzhugh_rinzel::register(module)?;
    mckean::register(module)?;
    sc_triangular_mckean::register(module)?;
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
    sc_unit_capacitance_respiratory::register(module)?;
    Ok(())
}

/// Register the default trainable and beta-cell model bindings.
pub(crate) fn register_tail(module: &Bound<'_, PyModule>) -> PyResult<()> {
    learnable_neuron_model::register(module)?;
    pernarowski::register(module)?;
    Ok(())
}
