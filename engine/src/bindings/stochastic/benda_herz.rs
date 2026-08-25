// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — source-faithful Benda-Herz PyO3 binding

use numpy::{IntoPyArray, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::neurons;

#[pyclass(
    name = "BendaHerzNeuron",
    module = "sc_neurocore_engine.sc_neurocore_engine"
)]
#[derive(Clone)]
pub struct PyBendaHerzNeuron {
    inner: neurons::BendaHerzNeuron,
}

#[pymethods]
impl PyBendaHerzNeuron {
    #[new]
    fn new() -> Self {
        Self {
            inner: neurons::BendaHerzNeuron::new(),
        }
    }

    fn step(&mut self, current: f64) -> i32 {
        self.inner.step(current)
    }

    fn reset(&mut self) {
        self.inner.reset();
    }

    fn get_state(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let d = PyDict::new(py);
        d.set_item("a", self.inner.a)?;
        d.set_item("phase", self.inner.phase)?;
        Ok(d.into_any().unbind())
    }
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyBendaHerzNeuron>()?;
    module.add_function(wrap_pyfunction!(py_benda_herz_simulate, module)?)?;
    Ok(())
}

#[pyfunction]
#[pyo3(signature=(a,phase,onset_gain,rheobase,adaptation_slope,tau_a,dt,currents))]
#[allow(clippy::too_many_arguments)]
fn py_benda_herz_simulate<'py>(
    py: Python<'py>,
    a: f64,
    phase: f64,
    onset_gain: f64,
    rheobase: f64,
    adaptation_slope: f64,
    tau_a: f64,
    dt: f64,
    currents: PyReadonlyArray1<'py, f64>,
) -> PyResult<Py<PyAny>> {
    let mut neuron = neurons::BendaHerzNeuron {
        a,
        phase,
        onset_gain,
        rheobase,
        adaptation_slope,
        tau_a,
        dt,
    };
    if !neuron.valid() {
        return Err(PyValueError::new_err("invalid Benda-Herz configuration"));
    }
    let mut adaptation = Vec::with_capacity(currents.len()?);
    let mut phases = Vec::with_capacity(currents.len()?);
    let mut events = Vec::with_capacity(currents.len()?);
    for &current in currents.as_slice()? {
        let event = neuron.step(current);
        if event < 0 {
            return Err(PyValueError::new_err("invalid Benda-Herz transition"));
        }
        adaptation.push(neuron.a);
        phases.push(neuron.phase);
        events.push(event);
    }
    let d = PyDict::new(py);
    d.set_item("adaptation", adaptation.into_pyarray(py))?;
    d.set_item("phases", phases.into_pyarray(py))?;
    d.set_item("events", events.into_pyarray(py))?;
    d.set_item("a_final", neuron.a)?;
    d.set_item("phase_final", neuron.phase)?;
    Ok(d.into_any().unbind())
}
