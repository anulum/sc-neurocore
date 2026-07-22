// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — conductance-based LIF PyO3 binding

//! Python binding for the Brette et al. 2007 conductance-based LIF cell.

use numpy::{IntoPyArray, PyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::neurons;

/// Register the COBA-LIF simulator with the extension module.
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyCOBALIFNeuron>()?;
    module.add_function(wrap_pyfunction!(py_coba_lif_simulate, module)?)?;
    Ok(())
}

// COBALIFNeuron: step(current, delta_ge, delta_gi)
#[pyclass(
    name = "COBALIFNeuron",
    module = "sc_neurocore_engine.sc_neurocore_engine"
)]
#[derive(Clone)]
pub struct PyCOBALIFNeuron {
    inner: neurons::COBALIFNeuron,
}

#[pymethods]
impl PyCOBALIFNeuron {
    #[new]
    #[pyo3(signature = (
        v=-60.0, g_e=0.0, g_i=0.0, refractory_time=0.0, c_m=200.0,
        g_l=10.0, e_l=-60.0, e_e=0.0, e_i=-80.0, tau_e=5.0,
        tau_i=10.0, v_threshold=-50.0, v_reset=-60.0,
        refractory_period=5.0, dt=0.1
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        v: f64,
        g_e: f64,
        g_i: f64,
        refractory_time: f64,
        c_m: f64,
        g_l: f64,
        e_l: f64,
        e_e: f64,
        e_i: f64,
        tau_e: f64,
        tau_i: f64,
        v_threshold: f64,
        v_reset: f64,
        refractory_period: f64,
        dt: f64,
    ) -> PyResult<Self> {
        let inner = neurons::COBALIFNeuron {
            v,
            g_e,
            g_i,
            refractory_time,
            c_m,
            g_l,
            e_l,
            e_e,
            e_i,
            tau_e,
            tau_i,
            v_threshold,
            v_reset,
            refractory_period,
            dt,
        };
        inner.validate().map_err(PyValueError::new_err)?;
        Ok(Self { inner })
    }
    #[pyo3(signature = (current, delta_ge=0.0, delta_gi=0.0))]
    fn step(&mut self, current: f64, delta_ge: f64, delta_gi: f64) -> PyResult<i32> {
        self.inner
            .try_step(current, delta_ge, delta_gi)
            .map_err(PyValueError::new_err)
    }
    fn reset(&mut self) {
        self.inner.reset();
    }
    fn get_state(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let d = PyDict::new(py);
        d.set_item("v", self.inner.v)?;
        d.set_item("g_e", self.inner.g_e)?;
        d.set_item("g_i", self.inner.g_i)?;
        d.set_item("refractory_time", self.inner.refractory_time)?;
        Ok(d.into_any().unbind())
    }
}

/// Full-contract parity surface for the Brette et al. 2007 COBA LIF cell.
#[pyfunction]
#[pyo3(signature = (
    v0, g_e0, g_i0, refractory_time0, c_m, g_l, e_l, e_e, e_i,
    tau_e, tau_i, v_threshold, v_reset, refractory_period, dt, n_steps,
    current, delta_ge, delta_gi
))]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn py_coba_lif_simulate<'py>(
    py: Python<'py>,
    v0: f64,
    g_e0: f64,
    g_i0: f64,
    refractory_time0: f64,
    c_m: f64,
    g_l: f64,
    e_l: f64,
    e_e: f64,
    e_i: f64,
    tau_e: f64,
    tau_i: f64,
    v_threshold: f64,
    v_reset: f64,
    refractory_period: f64,
    dt: f64,
    n_steps: usize,
    current: f64,
    delta_ge: f64,
    delta_gi: f64,
) -> PyResult<(Bound<'py, PyArray1<f64>>, i64, f64, f64, f64, f64)> {
    let mut neuron = crate::neurons::COBALIFNeuron {
        v: v0,
        g_e: g_e0,
        g_i: g_i0,
        refractory_time: refractory_time0,
        c_m,
        g_l,
        e_l,
        e_e,
        e_i,
        tau_e,
        tau_i,
        v_threshold,
        v_reset,
        refractory_period,
        dt,
    };
    neuron.validate().map_err(PyValueError::new_err)?;
    if !current.is_finite()
        || !delta_ge.is_finite()
        || delta_ge < 0.0
        || !delta_gi.is_finite()
        || delta_gi < 0.0
    {
        return Err(PyValueError::new_err("invalid COBA LIF simulation input"));
    }
    let mut trace = Vec::with_capacity(n_steps);
    let mut spikes = 0_i64;
    for _ in 0..n_steps {
        spikes += i64::from(
            neuron
                .try_step(current, delta_ge, delta_gi)
                .map_err(PyValueError::new_err)?,
        );
        trace.push(neuron.v);
    }
    Ok((
        trace.into_pyarray(py),
        spikes,
        neuron.v,
        neuron.g_e,
        neuron.g_i,
        neuron.refractory_time,
    ))
}
