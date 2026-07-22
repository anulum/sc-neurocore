// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Miscellaneous neuron PyO3 binding composition

use pyo3::prelude::*;

mod cardiac_purkinje_fibre;
mod endocrine_beta_cell;
mod frankenhaeuser_huxley_axon;
mod gap_junction_neuron;
mod graded_synapse_neuron;
mod myelinated_axon;
mod node_of_ranvier;
mod smooth_muscle_cell;

pub use cardiac_purkinje_fibre::PyCardiacPurkinjeFibre;
pub use endocrine_beta_cell::PyEndocrineBetaCell;
pub use frankenhaeuser_huxley_axon::PyFHAxon;
pub use gap_junction_neuron::PyGapJunctionNeuron;
pub use graded_synapse_neuron::PyGradedSynapseNeuron;
pub use myelinated_axon::PyMyelinatedAxon;
pub use node_of_ranvier::PyNodeOfRanvier;
pub use smooth_muscle_cell::PySmoothMuscleCell;

/// Register the eight model-owned miscellaneous neuron classes in stable ABI order.
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    graded_synapse_neuron::register(module)?;
    gap_junction_neuron::register(module)?;
    frankenhaeuser_huxley_axon::register(module)?;
    node_of_ranvier::register(module)?;
    myelinated_axon::register(module)?;
    cardiac_purkinje_fibre::register(module)?;
    smooth_muscle_cell::register(module)?;
    endocrine_beta_cell::register(module)?;
    Ok(())
}
