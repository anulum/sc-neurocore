// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Lapicque neuron PyO3 binding

use numpy::{IntoPyArray, PyArray1};
use pyo3::exceptions::{PyFloatingPointError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::neuron;

#[pyclass(
    name = "LapicqueNeuron",
    module = "sc_neurocore_engine.sc_neurocore_engine"
)]
#[derive(Clone)]
pub struct PyLapicqueNeuron {
    inner: neuron::LapicqueNeuron,
}

#[pymethods]
impl PyLapicqueNeuron {
    #[new]
    #[pyo3(signature = (tau=20.0, resistance=1.0, threshold=1.0, dt=1.0))]
    fn new(tau: f64, resistance: f64, threshold: f64, dt: f64) -> PyResult<Self> {
        let inner = neuron::LapicqueNeuron::new(tau, resistance, threshold, dt);
        if !inner.valid() {
            return Err(PyValueError::new_err(
                "Lapicque SC compatibility parameters violate the finite positive-RC contract",
            ));
        }
        Ok(Self { inner })
    }
    fn step(&mut self, current: f64) -> PyResult<i32> {
        self.inner
            .try_step(current)
            .map_err(|error| PyFloatingPointError::new_err(error.to_string()))
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

/// Register the Lapicque neuron class.
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyLapicqueNeuron>()?;
    module.add_function(wrap_pyfunction!(lapicque_simulate_complete, module)?)?;
    Ok(())
}

type LapicqueCompletePacket<'py> = (
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<u8>>,
    f64,
    bool,
);

/// Execute either the source or preserved SC profile in one checked boundary.
#[pyfunction]
#[pyo3(signature = (
    v, v_rest, v_reset, v_threshold, tau, resistance, dt, capacitance,
    series_resistance, polarization_resistance, excited, source_profile,
    n_steps, drive
))]
#[allow(clippy::too_many_arguments)]
fn lapicque_simulate_complete<'py>(
    py: Python<'py>,
    v: f64,
    v_rest: f64,
    v_reset: f64,
    v_threshold: f64,
    tau: f64,
    resistance: f64,
    dt: f64,
    capacitance: f64,
    series_resistance: f64,
    polarization_resistance: f64,
    excited: bool,
    source_profile: bool,
    n_steps: usize,
    drive: f64,
) -> PyResult<LapicqueCompletePacket<'py>> {
    let model = neuron::LapicqueNeuron {
        v,
        v_rest,
        v_reset,
        v_threshold,
        tau,
        resistance,
        dt,
        capacitance,
        series_resistance,
        polarization_resistance,
        excited,
        source_profile,
    };
    let (voltage, events, final_v, final_excited) =
        model.simulate_complete(n_steps, drive).map_err(|error| {
            PyFloatingPointError::new_err(format!("Lapicque batch rejected: {error}"))
        })?;
    Ok((
        voltage.into_pyarray(py),
        events.into_pyarray(py),
        final_v,
        final_excited,
    ))
}
