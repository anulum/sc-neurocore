// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — preserved six-state thalamocortical PyO3 binding

use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::neurons;

py_neuron_default!("SCSixStateThalamocorticalNeuron", PySCSixStateThalamocorticalNeuron, neurons::SCSixStateThalamocorticalNeuron, state v, state h_na, state n_k, state m_h, state h_t, state na_i);

/// Register the preserved six-state thalamocortical class.
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PySCSixStateThalamocorticalNeuron>()?;
    Ok(())
}
