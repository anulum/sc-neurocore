// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Integer QIF PyO3 binding

//! Python binding for the Wu et al. (2021) integer QIF neuron.

use numpy::{IntoPyArray, PyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::neurons;

/// Register the integer QIF batch simulator with the extension module.
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyIntegerQIFNeuron>()?;
    module.add_function(wrap_pyfunction!(py_iqif_simulate, module)?)?;
    Ok(())
}

#[pyclass(
    name = "IntegerQIFNeuron",
    module = "sc_neurocore_engine.sc_neurocore_engine"
)]
#[derive(Clone)]
pub struct PyIntegerQIFNeuron {
    inner: neurons::IntegerQIFNeuron,
}

#[pymethods]
impl PyIntegerQIFNeuron {
    #[new]
    #[pyo3(signature = (v=128, v_rest=128, v_threshold=200, v_reset=128, a=1, b=1, v_max=255, v_min=0))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        v: i32,
        v_rest: i32,
        v_threshold: i32,
        v_reset: i32,
        a: i32,
        b: i32,
        v_max: i32,
        v_min: i32,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: neurons::IntegerQIFNeuron::with_parameters(
                v,
                v_rest,
                v_threshold,
                v_reset,
                a,
                b,
                v_max,
                v_min,
            )
            .map_err(PyValueError::new_err)?,
        })
    }
    fn step(&mut self, current: i32) -> PyResult<i32> {
        self.inner.try_step(current).map_err(PyValueError::new_err)
    }
    fn reset(&mut self) {
        self.inner.reset();
    }
    fn get_state(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let d = PyDict::new(py);
        d.set_item("v", self.inner.v)?;
        d.set_item("v_rest", self.inner.v_rest)?;
        d.set_item("v_threshold", self.inner.v_threshold)?;
        d.set_item("v_reset", self.inner.v_reset)?;
        d.set_item("a", self.inner.a)?;
        d.set_item("b", self.inner.b)?;
        d.set_item("v_max", self.inner.v_max)?;
        d.set_item("v_min", self.inner.v_min)?;
        Ok(d.into_any().unbind())
    }
}

/// Full-contract Wu et al. (2021) IQIF integer batch.
#[pyfunction]
#[pyo3(signature = (v, v_rest, v_threshold, v_reset, a, b, v_max, v_min, n_steps, current))]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn py_iqif_simulate<'py>(
    py: Python<'py>,
    v: i32,
    v_rest: i32,
    v_threshold: i32,
    v_reset: i32,
    a: i32,
    b: i32,
    v_max: i32,
    v_min: i32,
    n_steps: usize,
    current: i32,
) -> PyResult<(Bound<'py, PyArray1<i64>>, i64, i64)> {
    let mut neuron = crate::neurons::IntegerQIFNeuron::with_parameters(
        v,
        v_rest,
        v_threshold,
        v_reset,
        a,
        b,
        v_max,
        v_min,
    )
    .map_err(PyValueError::new_err)?;
    let mut trace = Vec::with_capacity(n_steps);
    let mut spikes = 0_i64;
    for _ in 0..n_steps {
        spikes += i64::from(neuron.try_step(current).map_err(PyValueError::new_err)?);
        trace.push(neuron.v);
    }
    Ok((trace.into_pyarray(py), spikes, neuron.v))
}
