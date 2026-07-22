// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Cortical column PyO3 binding

//! Python binding for the layered cortical-column simulator.

use numpy::{IntoPyArray, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::cortical_column;

/// Register the layered cortical-column simulator with the extension module.
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyCorticalColumn>()?;
    Ok(())
}

#[pyclass(
    name = "CorticalColumnRust",
    module = "sc_neurocore_engine.sc_neurocore_engine"
)]
pub struct PyCorticalColumn {
    inner: cortical_column::CorticalColumnRust,
}

#[pymethods]
impl PyCorticalColumn {
    #[new]
    fn new(n: usize, tau: f64, dt: f64, threshold: f64, w_exc: f64, w_inh: f64, seed: u64) -> Self {
        Self {
            inner: cortical_column::CorticalColumnRust::new(
                n, tau, dt, threshold, w_exc, w_inh, seed,
            ),
        }
    }

    fn step<'py>(
        &mut self,
        py: Python<'py>,
        thalamic_input: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Py<PyDict>> {
        let input = thalamic_input.as_slice()?;
        let spikes = self.inner.step(input);
        let dict = PyDict::new(py);
        let names = ["l4", "l23_exc", "l23_inh", "l5", "l6"];
        for (i, name) in names.iter().enumerate() {
            dict.set_item(*name, spikes[i].clone().into_pyarray(py))?;
        }
        Ok(dict.into())
    }

    fn reset(&mut self) {
        self.inner.reset();
    }
}
