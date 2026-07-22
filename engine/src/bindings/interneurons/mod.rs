// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Interneuron PyO3 binding composition

use pyo3::prelude::*;

mod cerebellar_basket_neuron;
mod chandelier_neuron;
mod martinotti_neuron;
mod pv_fast_spiking_neuron;
mod sst_neuron;
mod vip_neuron;

pub use cerebellar_basket_neuron::PyCerebellarBasketNeuron;
pub use chandelier_neuron::PyChandelierNeuron;
pub use martinotti_neuron::PyMartinottiNeuron;
pub use pv_fast_spiking_neuron::PyPVFastSpikingNeuron;
pub use sst_neuron::PySSTNeuron;
pub use vip_neuron::PyVIPNeuron;

/// Register the six model-owned interneuron classes in stable ABI order.
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    pv_fast_spiking_neuron::register(module)?;
    sst_neuron::register(module)?;
    vip_neuron::register(module)?;
    chandelier_neuron::register(module)?;
    cerebellar_basket_neuron::register(module)?;
    martinotti_neuron::register(module)?;
    Ok(())
}
