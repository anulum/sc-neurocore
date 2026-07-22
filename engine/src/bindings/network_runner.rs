// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Network runner PyO3 binding

//! Python bindings for heterogeneous network execution and named-model batches.

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::network_runner;

/// Register network execution surfaces with the extension module.
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyNetworkRunner>()?;
    module.add_function(wrap_pyfunction!(py_batch_simulate, module)?)?;
    Ok(())
}

#[pyclass(
    name = "NetworkRunner",
    module = "sc_neurocore_engine.sc_neurocore_engine"
)]
pub struct PyNetworkRunner {
    inner: network_runner::NetworkRunner,
}

#[pymethods]
impl PyNetworkRunner {
    #[new]
    fn new() -> Self {
        Self {
            inner: network_runner::NetworkRunner::new(),
        }
    }

    fn add_population(&mut self, model: &str, n: usize) -> PyResult<usize> {
        let pop = network_runner::create_population(model, n).map_err(PyValueError::new_err)?;
        Ok(self.inner.add_population(pop))
    }

    #[pyo3(signature = (src, tgt, row_offsets, col_indices, values, delay=0))]
    fn add_projection(
        &mut self,
        src: usize,
        tgt: usize,
        row_offsets: Vec<i64>,
        col_indices: Vec<i64>,
        values: Vec<f64>,
        delay: usize,
    ) {
        let ro: Vec<usize> = row_offsets.iter().map(|&x| x as usize).collect();
        let ci: Vec<usize> = col_indices.iter().map(|&x| x as usize).collect();
        let proj = network_runner::ProjectionRunner::new(src, tgt, ro, ci, values, delay);
        self.inner.add_projection(proj);
    }

    fn step_population<'py>(
        &mut self,
        py: Python<'py>,
        pop_index: usize,
        currents: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Py<PyAny>> {
        let currents = currents.as_slice()?;
        let (spikes, voltages) = self
            .inner
            .step_population_with_currents(pop_index, currents)
            .map_err(PyValueError::new_err)?;
        let dict = PyDict::new(py);
        dict.set_item("spikes", spikes.into_pyarray(py))?;
        dict.set_item("voltages", voltages.into_pyarray(py))?;
        Ok(dict.into_any().unbind())
    }

    fn run<'py>(&mut self, py: Python<'py>, n_steps: usize) -> PyResult<Py<PyAny>> {
        let results = self.inner.run(n_steps);
        let dict = PyDict::new(py);
        let spike_counts: Vec<u64> = results.spike_counts.iter().map(|&c| c as u64).collect();
        dict.set_item("spike_counts", spike_counts.into_pyarray(py))?;
        let spike_data: Vec<Py<PyArray1<u64>>> = results
            .spike_data
            .into_iter()
            .map(|v: Vec<u64>| v.into_pyarray(py).unbind())
            .collect();
        dict.set_item("spike_data", spike_data)?;
        let voltages: Vec<Py<PyArray1<f64>>> = results
            .voltages
            .into_iter()
            .map(|v: Vec<f64>| v.into_pyarray(py).unbind())
            .collect();
        dict.set_item("voltages", voltages)?;
        Ok(dict.into_any().unbind())
    }

    #[staticmethod]
    fn supported_models() -> Vec<&'static str> {
        network_runner::supported_models()
    }
}

/// Run a named neuron model for n_steps with a current trace, returning
/// voltage trace + spike indices. Entire simulation in Rust.
#[pyfunction]
fn py_batch_simulate<'py>(
    py: Python<'py>,
    model_name: &str,
    n_steps: usize,
    current_trace: PyReadonlyArray1<'py, f64>,
) -> PyResult<Py<PyAny>> {
    let mut neuron = network_runner::create_neuron(model_name).map_err(PyValueError::new_err)?;
    let currents = current_trace.as_slice()?;
    let steps = n_steps.min(currents.len());

    let mut voltages = vec![0.0f64; steps];
    let mut spikes: Vec<u64> = Vec::new();

    for t in 0..steps {
        let fired = neuron.step(currents[t]);
        voltages[t] = neuron.soma_voltage();
        if fired != 0 {
            spikes.push(t as u64);
        }
    }

    let d = PyDict::new(py);
    d.set_item("voltages", voltages.into_pyarray(py))?;
    d.set_item("spikes", spikes.into_pyarray(py))?;
    d.set_item("n_steps", steps)?;
    Ok(d.into_any().unbind())
}
