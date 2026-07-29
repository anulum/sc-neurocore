// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — source-complete Brunel-Wang PyO3 binding

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::neurons;

/// Python boundary for the configurable Brunel-Wang pyramidal cell.
#[pyclass(
    name = "BrunelWangNeuron",
    module = "sc_neurocore_engine.sc_neurocore_engine"
)]
#[derive(Clone)]
pub struct PyBrunelWangNeuron {
    inner: neurons::BrunelWangNeuron,
}

#[pymethods]
impl PyBrunelWangNeuron {
    #[new]
    #[pyo3(signature = (v=-70.0, v_rest=-70.0, v_reset=-55.0, v_threshold=-50.0, tau_m=20.0, tau_ref=2.0, g_ampa_ext=2.08, g_ampa_rec=0.104, g_nmda=0.327, g_gaba=1.25, v_ampa=0.0, v_nmda=0.0, v_gaba=-70.0, c_m=0.5, mg_conc=1.0, dt=0.1, ref_remaining=0.0))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        v: f64,
        v_rest: f64,
        v_reset: f64,
        v_threshold: f64,
        tau_m: f64,
        tau_ref: f64,
        g_ampa_ext: f64,
        g_ampa_rec: f64,
        g_nmda: f64,
        g_gaba: f64,
        v_ampa: f64,
        v_nmda: f64,
        v_gaba: f64,
        c_m: f64,
        mg_conc: f64,
        dt: f64,
        ref_remaining: f64,
    ) -> PyResult<Self> {
        let mut inner = neurons::BrunelWangNeuron::new();
        inner.v = v;
        inner.v_rest = v_rest;
        inner.v_reset = v_reset;
        inner.v_threshold = v_threshold;
        inner.tau_m = tau_m;
        inner.tau_ref = tau_ref;
        inner.g_ampa_ext = g_ampa_ext;
        inner.g_ampa_rec = g_ampa_rec;
        inner.g_nmda = g_nmda;
        inner.g_gaba = g_gaba;
        inner.v_ampa = v_ampa;
        inner.v_nmda = v_nmda;
        inner.v_gaba = v_gaba;
        inner.c_m = c_m;
        inner.mg_conc = mg_conc;
        inner.dt = dt;
        inner.ref_remaining = ref_remaining;
        inner
            .try_step_full(0.0, 0.0, 0.0, 0.0)
            .map_err(PyValueError::new_err)?;
        inner.v = v;
        inner.ref_remaining = ref_remaining;
        Ok(Self { inner })
    }

    /// Advance one atomic midpoint-RK2 step over four aggregate gates.
    #[pyo3(signature = (i_ampa_ext=0.0, s_ampa_rec=0.0, s_nmda_rec=0.0, s_gaba=0.0))]
    fn step(
        &mut self,
        i_ampa_ext: f64,
        s_ampa_rec: f64,
        s_nmda_rec: f64,
        s_gaba: f64,
    ) -> PyResult<i32> {
        self.inner
            .try_step_full(i_ampa_ext, s_ampa_rec, s_nmda_rec, s_gaba)
            .map_err(PyValueError::new_err)
    }

    /// Reset membrane and refractory state while preserving configuration.
    fn reset(&mut self) {
        self.inner.reset();
    }

    /// Return `(voltage, refractory_remaining_ms)`.
    fn get_state(&self) -> (f64, f64) {
        self.inner.state()
    }
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyBrunelWangNeuron>()?;
    Ok(())
}
