// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Two-compartment LIF PyO3 bindings

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::neurons;

#[pyclass(
    name = "TwoCompartmentLIFNeuron",
    module = "sc_neurocore_engine.sc_neurocore_engine"
)]
#[derive(Clone)]
pub struct PyTwoCompartmentLIFNeuron {
    inner: neurons::TwoCompartmentLIFNeuron,
}

#[pymethods]
impl PyTwoCompartmentLIFNeuron {
    #[new]
    fn new() -> Self {
        Self {
            inner: neurons::TwoCompartmentLIFNeuron::new(),
        }
    }

    /// Advance one step with the single external dendritic input; raises
    /// `ValueError` with the state unchanged on any invalid input.
    fn step(&mut self, i_ext: f64) -> PyResult<i32> {
        self.inner.try_step(i_ext).map_err(PyValueError::new_err)
    }

    /// Restore the dynamic state to zero, preserving parameters.
    fn reset(&mut self) {
        self.inner.reset();
    }

    /// Return the complete dynamic state as a Python dictionary.
    fn get_state(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let state = PyDict::new(py);
        state.set_item("u_d", self.inner.u_d)?;
        state.set_item("u_s", self.inner.u_s)?;
        state.set_item("s_prev", self.inner.s_prev)?;
        Ok(state.into_any().unbind())
    }
}

#[pyclass(
    name = "SCExponentialTwoCompartmentLIF",
    module = "sc_neurocore_engine.sc_neurocore_engine"
)]
#[derive(Clone)]
pub struct PySCExponentialTwoCompartmentLIF {
    inner: neurons::SCExponentialTwoCompartmentLIF,
}

#[pymethods]
impl PySCExponentialTwoCompartmentLIF {
    #[new]
    fn new() -> Self {
        Self {
            inner: neurons::SCExponentialTwoCompartmentLIF::new(),
        }
    }

    /// Advance one step with the historical two-current recurrence,
    /// preserved verbatim from the pre-2026-08-27 engine.
    #[pyo3(signature = (i_soma, i_dend=0.0))]
    fn step(&mut self, i_soma: f64, i_dend: f64) -> i32 {
        self.inner.step(i_soma, i_dend)
    }

    /// Restore both compartments to the rest potential.
    fn reset(&mut self) {
        self.inner.reset();
    }

    /// Return the complete dynamic state as a Python dictionary.
    fn get_state(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let state = PyDict::new(py);
        state.set_item("v_s", self.inner.v_s)?;
        state.set_item("v_d", self.inner.v_d)?;
        Ok(state.into_any().unbind())
    }
}

/// Register both two-compartment LIF classes.
pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyTwoCompartmentLIFNeuron>()?;
    module.add_class::<PySCExponentialTwoCompartmentLIF>()?;
    Ok(())
}
