// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Hill-Tononi neuron PyO3 binding

use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::neurons;

py_neuron_default!("HillTononiNeuron", PyHillTononiNeuron, neurons::HillTononiNeuron, state v, state theta, state d_k, state m_h, state m_t, state h_t, state spike_timer);

/// Register the Hill-Tononi neuron class.
pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyHillTononiNeuron>()?;
    Ok(())
}
