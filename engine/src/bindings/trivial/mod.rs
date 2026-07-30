// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Simple integrate-and-fire PyO3 binding composition

use pyo3::prelude::*;

#[path = "../adaptive_threshold_if.rs"]
mod adaptive_threshold_if;
mod closed_form_continuous;
mod complementary_lif;
mod energy_lif;
mod gated_lif;
mod inhibitory_lif;
mod klif;
mod mat;
mod non_resetting_lif;
mod nonlinear_lif;
mod parametric_lif;
mod perfect_integrator;
mod quadratic_if;
mod sc_non_resetting_adaptive_lif;
mod sc_resetting_mat;
mod sfa;
mod sigma_delta;
mod theta;

pub use adaptive_threshold_if::PyAdaptiveThresholdIFNeuron;
pub use closed_form_continuous::PyClosedFormContinuousNeuron;
pub use complementary_lif::PyComplementaryLIFNeuron;
pub use energy_lif::PyEnergyLIFNeuron;
pub use gated_lif::PyGatedLIFNeuron;
pub use inhibitory_lif::PyInhibitoryLIFNeuron;
pub use klif::PyKLIFNeuron;
pub use mat::PyMATNeuron;
pub use non_resetting_lif::PyNonResettingLIFNeuron;
pub use nonlinear_lif::PyNonlinearLIFNeuron;
pub use parametric_lif::PyParametricLIFNeuron;
pub use perfect_integrator::PyPerfectIntegratorNeuron;
pub use quadratic_if::PyQuadraticIFNeuron;
pub use sc_non_resetting_adaptive_lif::PySCNonResettingAdaptiveLIFNeuron;
pub use sc_resetting_mat::PySCResettingMATNeuron;
pub use sfa::PySFANeuron;
pub use sigma_delta::PySigmaDeltaNeuron;
pub use theta::PyThetaNeuron;

/// Register the eighteen model-owned simple integrate-and-fire classes in stable ABI order.
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    quadratic_if::register(module)?;
    theta::register(module)?;
    perfect_integrator::register(module)?;
    gated_lif::register(module)?;
    nonlinear_lif::register(module)?;
    sfa::register(module)?;
    mat::register(module)?;
    sc_resetting_mat::register(module)?;
    sc_non_resetting_adaptive_lif::register(module)?;
    klif::register(module)?;
    inhibitory_lif::register(module)?;
    complementary_lif::register(module)?;
    parametric_lif::register(module)?;
    non_resetting_lif::register(module)?;
    adaptive_threshold_if::register(module)?;
    sigma_delta::register(module)?;
    energy_lif::register(module)?;
    closed_form_continuous::register(module)?;
    Ok(())
}
