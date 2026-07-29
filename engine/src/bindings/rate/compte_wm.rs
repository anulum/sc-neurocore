// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Compte pyramidal-cell PyO3 binding

//! Python boundary for the complete Compte cell and channel state.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::neurons;

/// PyO3 owner for one configured Compte pyramidal cell.
#[pyclass(
    name = "CompteWMNeuron",
    module = "sc_neurocore_engine.sc_neurocore_engine"
)]
#[derive(Clone)]
pub struct PyCompteWMNeuron {
    inner: neurons::CompteWMNeuron,
}

#[pymethods]
impl PyCompteWMNeuron {
    /// Construct a configured cell; omitted arguments use source defaults.
    #[new]
    #[pyo3(signature = (
        v=-70.0, s_ampa=0.0, s_nmda=0.0, x_nmda=0.0, s_gaba=0.0,
        ref_remaining=0.0, g_l=0.025, g_ampa=0.0031, g_nmda=0.000381,
        g_gaba=0.001336, e_l=-70.0, e_exc=0.0, e_inh=-70.0, c_m=0.5,
        mg=1.0, tau_ampa=2.0, tau_nmda=100.0, tau_x=2.0, tau_gaba=10.0,
        alpha_nmda=0.5, v_threshold=-50.0, v_reset=-60.0, tau_ref=2.0,
        dt=0.02
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        v: f64,
        s_ampa: f64,
        s_nmda: f64,
        x_nmda: f64,
        s_gaba: f64,
        ref_remaining: f64,
        g_l: f64,
        g_ampa: f64,
        g_nmda: f64,
        g_gaba: f64,
        e_l: f64,
        e_exc: f64,
        e_inh: f64,
        c_m: f64,
        mg: f64,
        tau_ampa: f64,
        tau_nmda: f64,
        tau_x: f64,
        tau_gaba: f64,
        alpha_nmda: f64,
        v_threshold: f64,
        v_reset: f64,
        tau_ref: f64,
        dt: f64,
    ) -> PyResult<Self> {
        let inner = neurons::CompteWMNeuron {
            v,
            s_ampa,
            s_nmda,
            x_nmda,
            s_gaba,
            ref_remaining,
            g_l,
            g_ampa,
            g_nmda,
            g_gaba,
            e_l,
            e_exc,
            e_inh,
            c_m,
            mg,
            tau_ampa,
            tau_nmda,
            tau_x,
            tau_gaba,
            alpha_nmda,
            v_threshold,
            v_reset,
            tau_ref,
            dt,
        };
        if !inner.validate() {
            return Err(PyValueError::new_err(
                "invalid Compte state or configuration",
            ));
        }
        Ok(Self { inner })
    }

    /// Advance one atomic source-level step.
    #[pyo3(signature = (
        current, recurrent_event=false, external_event=false, inhibitory_event=false
    ))]
    fn step(
        &mut self,
        current: f64,
        recurrent_event: bool,
        external_event: bool,
        inhibitory_event: bool,
    ) -> PyResult<i32> {
        self.inner
            .step_events(current, recurrent_event, external_event, inhibitory_event)
            .map_err(PyValueError::new_err)
    }

    /// Reset dynamic state while retaining configuration.
    fn reset(&mut self) {
        self.inner.reset();
    }

    /// Return the complete named dynamic state.
    fn get_state(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let d = PyDict::new(py);
        let values = self.inner.get_state();
        for (name, value) in ["v", "s_ampa", "s_nmda", "x_nmda", "s_gaba", "ref_remaining"]
            .into_iter()
            .zip(values)
        {
            d.set_item(name, value)?;
        }
        Ok(d.into_any().unbind())
    }
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyCompteWMNeuron>()?;
    Ok(())
}
