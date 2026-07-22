// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Adaptive-threshold mixture-of-experts neuron PyO3 binding

use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::neurons;

py_neuron_default!("RustAdaptiveThresholdMoENeuron", PyAdaptiveThresholdMoENeuron, neurons::AdaptiveThresholdMoENeuron, state v, state v_th);

/// Register the adaptive-threshold mixture-of-experts neuron class.
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyAdaptiveThresholdMoENeuron>()?;
    Ok(())
}
