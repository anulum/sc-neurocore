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
mod sfa;
mod sigma_delta;
mod theta;

/// Register the sixteen model-owned simple integrate-and-fire classes in stable ABI order.
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    quadratic_if::register(module)?;
    theta::register(module)?;
    perfect_integrator::register(module)?;
    gated_lif::register(module)?;
    nonlinear_lif::register(module)?;
    sfa::register(module)?;
    mat::register(module)?;
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
