// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Motor neuron PyO3 binding composition

use pyo3::prelude::*;

mod alpha_motor_neuron;
mod gamma_motor_neuron;
mod motor_unit;
mod renshaw_cell;
mod upper_motor_neuron;

/// Register the five model-owned motor neuron classes in stable ABI order.
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    alpha_motor_neuron::register(module)?;
    gamma_motor_neuron::register(module)?;
    upper_motor_neuron::register(module)?;
    renshaw_cell::register(module)?;
    motor_unit::register(module)?;
    Ok(())
}
