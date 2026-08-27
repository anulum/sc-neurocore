// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Ion-channel neuron PyO3 binding composition

use pyo3::prelude::*;

mod a_type_k_neuron;
mod bk_neuron;
mod ih_neuron;
mod nmda_neuron;
mod persistent_na_neuron;
mod sc_wb_nmda_magnesium_block;
mod sk_neuron;
mod t_type_ca_neuron;

pub use a_type_k_neuron::PyATypeKNeuron;
pub use bk_neuron::PyBKNeuron;
pub use ih_neuron::PyIhNeuron;
pub use nmda_neuron::PyNMDANeuron;
pub use persistent_na_neuron::PyPersistentNaNeuron;
pub use sc_wb_nmda_magnesium_block::PySCWBNMDAMagnesiumBlockNeuron;
pub use sk_neuron::PySKNeuron;
pub use t_type_ca_neuron::PyTTypeCaNeuron;

/// Register the seven model-owned ion-channel neuron classes in stable ABI order.
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    persistent_na_neuron::register(module)?;
    ih_neuron::register(module)?;
    t_type_ca_neuron::register(module)?;
    a_type_k_neuron::register(module)?;
    bk_neuron::register(module)?;
    sk_neuron::register(module)?;
    nmda_neuron::register(module)?;
    sc_wb_nmda_magnesium_block::register(module)?;
    Ok(())
}
