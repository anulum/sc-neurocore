// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Perfect integrator neuron PyO3 binding

use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::neurons;

py_neuron_default!("PerfectIntegratorNeuron", PyPerfectIntegratorNeuron, neurons::PerfectIntegratorNeuron, state v);

/// Register the perfect integrator neuron class.
pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyPerfectIntegratorNeuron>()?;
    Ok(())
}
