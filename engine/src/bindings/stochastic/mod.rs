// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Stochastic neuron PyO3 binding composition

use pyo3::prelude::*;

mod benda_herz;
mod galves_locherbach;
mod gamma_renewal;
mod glm;
mod inhomogeneous_poisson;
mod sc_stochastic_rate_adaptation;
mod spike_response;
mod stochastic_if;
mod stochastic_lif;

pub use benda_herz::PyBendaHerzNeuron;
pub use galves_locherbach::PyGalvesLocherbachNeuron;
pub use gamma_renewal::PyGammaRenewalNeuron;
pub use glm::PyGLMNeuron;
pub use inhomogeneous_poisson::PyInhomogeneousPoissonNeuron;
pub use sc_stochastic_rate_adaptation::PySCStochasticRateAdaptationNeuron;
pub use spike_response::PySpikeResponseNeuron;
pub use stochastic_if::PyStochasticIFNeuron;
pub use stochastic_lif::PyStochasticLIFNeuron;

pub(crate) fn register_adaptation(module: &Bound<'_, PyModule>) -> PyResult<()> {
    benda_herz::register(module)?;
    sc_stochastic_rate_adaptation::register(module)?;
    Ok(())
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    inhomogeneous_poisson::register(module)?;
    gamma_renewal::register(module)?;
    stochastic_if::register(module)?;
    stochastic_lif::register(module)?;
    galves_locherbach::register(module)?;
    spike_response::register(module)?;
    glm::register(module)?;
    Ok(())
}
