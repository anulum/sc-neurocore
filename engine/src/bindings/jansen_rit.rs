// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Jansen–Rit PyO3 scalar and batch binding

//! Python boundary for the equation-(6) explicit-Euler contract.

use numpy::{IntoPyArray, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::neurons::JansenRitUnit;

/// Register the scalar class and batch function without growing `lib.rs`.
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyJansenRitUnit>()?;
    module.add_function(wrap_pyfunction!(py_jansen_rit_simulate, module)?)?;
    Ok(())
}

#[pyclass(
    name = "JansenRitUnit",
    module = "sc_neurocore_engine.sc_neurocore_engine"
)]
#[derive(Clone)]
pub struct PyJansenRitUnit {
    inner: JansenRitUnit,
}

#[pymethods]
impl PyJansenRitUnit {
    #[new]
    #[pyo3(signature = (
        y0=0.0, y3=0.0, y1=0.0, y4=0.0, y2=0.0, y5=0.0,
        a_exc=3.25, b_exc=22.0, a_rate=100.0, b_rate=50.0,
        c=135.0, e0=2.5, v0=6.0, r=0.56, dt=0.0001,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        y0: f64,
        y3: f64,
        y1: f64,
        y4: f64,
        y2: f64,
        y5: f64,
        a_exc: f64,
        b_exc: f64,
        a_rate: f64,
        b_rate: f64,
        c: f64,
        e0: f64,
        v0: f64,
        r: f64,
        dt: f64,
    ) -> PyResult<Self> {
        let inner = JansenRitUnit::with_parameters(
            y0, y3, y1, y4, y2, y5, a_exc, b_exc, a_rate, b_rate, c, e0, v0, r, dt,
        )
        .map_err(PyValueError::new_err)?;
        Ok(Self { inner })
    }

    #[pyo3(signature = (p_ext=220.0))]
    fn step(&mut self, p_ext: f64) -> PyResult<f64> {
        self.inner.step(p_ext).map_err(PyValueError::new_err)
    }

    fn reset(&mut self) {
        self.inner.reset();
    }

    fn get_state(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let mapping = PyDict::new(py);
        mapping.set_item("y0", self.inner.y[0])?;
        mapping.set_item("y3", self.inner.y[3])?;
        mapping.set_item("y1", self.inner.y[1])?;
        mapping.set_item("y4", self.inner.y[4])?;
        mapping.set_item("y2", self.inner.y[2])?;
        mapping.set_item("y5", self.inner.y[5])?;
        mapping.set_item("y", self.inner.y.to_vec())?;
        Ok(mapping.into_any().unbind())
    }
}

/// Simulate a complete Jansen–Rit batch from caller-owned external drive.
#[pyfunction]
#[pyo3(signature = (
    y0_init, y3_init, y1_init, y4_init, y2_init, y5_init,
    a_exc, b_exc, a_rate, b_rate, c, e0, v0, r, dt, p_ext,
))]
#[allow(clippy::too_many_arguments)]
fn py_jansen_rit_simulate<'py>(
    py: Python<'py>,
    y0_init: f64,
    y3_init: f64,
    y1_init: f64,
    y4_init: f64,
    y2_init: f64,
    y5_init: f64,
    a_exc: f64,
    b_exc: f64,
    a_rate: f64,
    b_rate: f64,
    c: f64,
    e0: f64,
    v0: f64,
    r: f64,
    dt: f64,
    p_ext: PyReadonlyArray1<'py, f64>,
) -> PyResult<Py<PyAny>> {
    let result = crate::neurons::jansen_rit::simulate(
        y0_init,
        y3_init,
        y1_init,
        y4_init,
        y2_init,
        y5_init,
        a_exc,
        b_exc,
        a_rate,
        b_rate,
        c,
        e0,
        v0,
        r,
        dt,
        p_ext.as_slice()?,
    )
    .map_err(PyValueError::new_err)?;
    let mapping = PyDict::new(py);
    mapping.set_item("y0", result.y0.into_pyarray(py))?;
    mapping.set_item("y3", result.y3.into_pyarray(py))?;
    mapping.set_item("y1", result.y1.into_pyarray(py))?;
    mapping.set_item("y4", result.y4.into_pyarray(py))?;
    mapping.set_item("y2", result.y2.into_pyarray(py))?;
    mapping.set_item("y5", result.y5.into_pyarray(py))?;
    mapping.set_item("eeg", result.eeg.into_pyarray(py))?;
    mapping.set_item("y0_final", result.final_state[0])?;
    mapping.set_item("y3_final", result.final_state[3])?;
    mapping.set_item("y1_final", result.final_state[1])?;
    mapping.set_item("y4_final", result.final_state[4])?;
    mapping.set_item("y2_final", result.final_state[2])?;
    mapping.set_item("y5_final", result.final_state[5])?;
    Ok(mapping.into_any().unbind())
}

#[cfg(test)]
mod tests {
    #[test]
    fn engine_batch_rejects_invalid_drive_without_partial_result() {
        let result = crate::neurons::jansen_rit::simulate(
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            3.25,
            22.0,
            100.0,
            50.0,
            135.0,
            2.5,
            6.0,
            0.56,
            0.0001,
            &[120.0, f64::NAN],
        );
        assert!(result.is_err());
    }
}
