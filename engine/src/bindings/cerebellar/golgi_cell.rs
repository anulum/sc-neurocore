// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Golgi cell PyO3 binding

use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::neurons;

py_neuron_default!("GolgiCell", PyGolgiCell, neurons::GolgiCell, state v, state m, state h, state p_na, state n, state a, state b, state w, state m_t, state s, state c_n, state r, state ca);

/// Register the Golgi cell class.
pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyGolgiCell>()?;
    Ok(())
}
