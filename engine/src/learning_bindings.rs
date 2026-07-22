// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Learning binding registration facade

//! Registers the responsibility-specific differentiable-learning bindings.

use pyo3::prelude::*;

#[path = "bindings/differentiable_dense.rs"]
mod differentiable_dense_binding;
#[path = "bindings/stochastic_attention.rs"]
mod stochastic_attention_binding;
#[path = "bindings/stochastic_graph.rs"]
mod stochastic_graph_binding;
#[path = "bindings/surrogate.rs"]
mod surrogate_binding;

/// Register differentiable and stochastic learning bindings.
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    surrogate_binding::register(module)?;
    differentiable_dense_binding::register(module)?;
    stochastic_attention_binding::register(module)?;
    stochastic_graph_binding::register(module)?;
    Ok(())
}
