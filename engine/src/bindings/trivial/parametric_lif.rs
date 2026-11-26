// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Parametric LIF neuron PyO3 binding

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::neurons;

/// Python class for the parametric LIF neuron, built from the fields of the Python model.
///
/// `v`, `a` and `threshold` take the Python model's defaults and are checked as it checks
/// them: `v` and `a` finite, `threshold` finite and positive. The Python model's `dt` does not
/// enter the PLIF step, so the engine class has none.
#[pyclass(
    name = "ParametricLIFNeuron",
    module = "sc_neurocore_engine.sc_neurocore_engine"
)]
#[derive(Clone)]
pub struct PyParametricLIFNeuron {
    inner: neurons::ParametricLIFNeuron,
}

#[pymethods]
impl PyParametricLIFNeuron {
    #[new]
    #[pyo3(signature = (v = 0.0, a = 0.0, threshold = 1.0))]
    fn new(v: f64, a: f64, threshold: f64) -> PyResult<Self> {
        if !v.is_finite() {
            return Err(PyValueError::new_err("v must be finite"));
        }
        if !a.is_finite() {
            return Err(PyValueError::new_err("a must be finite"));
        }
        if !threshold.is_finite() || threshold <= 0.0 {
            return Err(PyValueError::new_err(
                "threshold must be finite and positive",
            ));
        }
        let mut inner = neurons::ParametricLIFNeuron::new(a, threshold);
        inner.v = v;
        Ok(Self { inner })
    }

    fn step(&mut self, current: f64) -> i32 {
        self.inner.step(current)
    }

    fn reset(&mut self) {
        self.inner.reset();
    }

    fn get_state(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let d = PyDict::new(py);
        d.set_item("v", self.inner.v)?;
        Ok(d.into_any().unbind())
    }
}

/// Register the parametric LIF neuron class.
pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyParametricLIFNeuron>()?;
    Ok(())
}
