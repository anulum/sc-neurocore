// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Sampled APSDM native Python binding

//! PyO3 exposure for the sampled APSDM contract.

use numpy::{PyArrayMethods, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::fixed_point_lif_binding::array_allocation::zeros_1d;
use crate::neurons;

/// Python-owned sampled APSDM neuron with complete source configuration.
#[pyclass(
    name = "SigmaDeltaNeuron",
    module = "sc_neurocore_engine.sc_neurocore_engine"
)]
#[derive(Clone)]
pub struct PySigmaDeltaNeuron {
    inner: neurons::SigmaDeltaNeuron,
}

#[pymethods]
impl PySigmaDeltaNeuron {
    /// Construct a configured integrating prefilter and reconstruction state.
    #[new]
    #[pyo3(signature=(sigma=0.0,reconstruction=0.0,delta=1.0,tau_reconstruction=10.0,dt=0.1))]
    fn new(
        sigma: f64,
        reconstruction: f64,
        delta: f64,
        tau_reconstruction: f64,
        dt: f64,
    ) -> PyResult<Self> {
        let inner = neurons::SigmaDeltaNeuron {
            sigma,
            reconstruction,
            delta,
            tau_reconstruction,
            dt,
        };
        if !inner.validate() {
            return Err(PyValueError::new_err(
                "invalid SigmaDelta state or configuration",
            ));
        }
        Ok(Self { inner })
    }
    /// Advance one atomic sampled transition and return its unipolar event.
    fn step(&mut self, current: f64) -> PyResult<i32> {
        self.inner.try_step(current).map_err(PyValueError::new_err)
    }
    /// Clear both dynamic states while retaining the configured parameters.
    fn reset(&mut self) {
        self.inner.reset();
    }
    /// Return a detached dictionary of both dynamic state values.
    fn get_state(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let d = PyDict::new(py);
        d.set_item("sigma", self.inner.sigma)?;
        d.set_item("reconstruction", self.inner.reconstruction)?;
        Ok(d.into_any().unbind())
    }
}

/// Simulate APSDM into owning arrays, propagating layout and allocation errors.
#[pyfunction]
#[pyo3(signature=(sigma,reconstruction,delta,tau_reconstruction,dt,currents))]
fn py_sigma_delta_simulate<'py>(
    py: Python<'py>,
    sigma: f64,
    reconstruction: f64,
    delta: f64,
    tau_reconstruction: f64,
    dt: f64,
    currents: PyReadonlyArray1<'py, f64>,
) -> PyResult<Py<PyAny>> {
    let mut n = neurons::SigmaDeltaNeuron {
        sigma,
        reconstruction,
        delta,
        tau_reconstruction,
        dt,
    };
    if !n.validate() {
        return Err(PyValueError::new_err(
            "invalid SigmaDelta state or configuration",
        ));
    }
    let inputs = currents.as_slice()?;
    let sigmas = zeros_1d::<f64>(py, inputs.len())?;
    let reconstructions = zeros_1d::<f64>(py, inputs.len())?;
    let events = zeros_1d::<i32>(py, inputs.len())?;
    // SAFETY: The three fresh contiguous arrays have disjoint writable storage
    // and no external aliases. Their owners outlive these mutable slices.
    let (sigma_values, reconstruction_values, event_values) = unsafe {
        (
            sigmas.as_slice_mut()?,
            reconstructions.as_slice_mut()?,
            events.as_slice_mut()?,
        )
    };
    for (index, &current) in inputs.iter().enumerate() {
        event_values[index] = n.try_step(current).map_err(PyValueError::new_err)?;
        sigma_values[index] = n.sigma;
        reconstruction_values[index] = n.reconstruction;
    }
    let d = PyDict::new(py);
    d.set_item("sigma", sigmas)?;
    d.set_item("reconstruction", reconstructions)?;
    d.set_item("events", events)?;
    d.set_item("sigma_final", n.sigma)?;
    d.set_item("reconstruction_final", n.reconstruction)?;
    Ok(d.into_any().unbind())
}
/// Register the sampled source APSDM class and batch simulator.
pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PySigmaDeltaNeuron>()?;
    module.add_function(wrap_pyfunction!(py_sigma_delta_simulate, module)?)?;
    Ok(())
}
