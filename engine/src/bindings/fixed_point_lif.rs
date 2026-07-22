// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — fixed-point LIF PyO3 bindings

//! Python bindings for sequential and parallel fixed-point LIF batch kernels.

use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::neuron;

/// Register fixed-point LIF batch kernels with the extension module.
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<FixedPointLif>()?;
    module.add_function(wrap_pyfunction!(batch_lif_run, module)?)?;
    module.add_function(wrap_pyfunction!(batch_lif_run_multi, module)?)?;
    module.add_function(wrap_pyfunction!(batch_lif_run_varying, module)?)?;
    Ok(())
}

#[pyclass(module = "sc_neurocore_engine.sc_neurocore_engine")]
pub struct FixedPointLif {
    inner: neuron::FixedPointLif,
}

#[pymethods]
impl FixedPointLif {
    #[new]
    #[pyo3(signature = (
        data_width=16,
        fraction=8,
        v_rest=0,
        v_reset=0,
        v_threshold=256,
        refractory_period=2
    ))]
    fn new(
        data_width: u32,
        fraction: u32,
        v_rest: i16,
        v_reset: i16,
        v_threshold: i16,
        refractory_period: i32,
    ) -> Self {
        Self {
            inner: neuron::FixedPointLif::new(
                data_width,
                fraction,
                v_rest,
                v_reset,
                v_threshold,
                refractory_period,
            ),
        }
    }

    #[pyo3(signature = (leak_k, gain_k, i_t, noise_in=0))]
    fn step(&mut self, leak_k: i16, gain_k: i16, i_t: i16, noise_in: i16) -> (i32, i16) {
        self.inner.step(leak_k, gain_k, i_t, noise_in)
    }

    fn reset(&mut self) {
        self.inner.reset();
    }

    fn reset_state(&mut self) {
        self.reset();
    }

    fn get_state(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let dict = PyDict::new(py);
        dict.set_item("v", self.inner.v)?;
        dict.set_item("refractory_counter", self.inner.refractory_counter)?;
        Ok(dict.into_any().unbind())
    }
}

/// Run a LIF neuron for N steps with constant inputs.
///
/// Returns (spikes: ndarray[i32], voltages: ndarray[i16]).
#[pyfunction]
#[pyo3(signature = (
    n_steps,
    leak_k,
    gain_k,
    i_t,
    noise_in=0,
    data_width=16,
    fraction=8,
    v_rest=0,
    v_reset=0,
    v_threshold=256,
    refractory_period=2
))]
#[allow(clippy::too_many_arguments)]
fn batch_lif_run<'py>(
    py: Python<'py>,
    n_steps: usize,
    leak_k: i16,
    gain_k: i16,
    i_t: i16,
    noise_in: i16,
    data_width: u32,
    fraction: u32,
    v_rest: i16,
    v_reset: i16,
    v_threshold: i16,
    refractory_period: i32,
) -> (Bound<'py, PyArray1<i32>>, Bound<'py, PyArray1<i16>>) {
    let mut lif = neuron::FixedPointLif::new(
        data_width,
        fraction,
        v_rest,
        v_reset,
        v_threshold,
        refractory_period,
    );
    let spikes_arr = PyArray1::<i32>::zeros(py, n_steps, false);
    let voltages_arr = PyArray1::<i16>::zeros(py, n_steps, false);

    // SAFETY: Arrays are newly allocated and contiguous.
    let spikes_slice = unsafe {
        spikes_arr
            .as_slice_mut()
            .expect("newly allocated spikes array must be contiguous")
    };
    // SAFETY: Arrays are newly allocated and contiguous.
    let voltages_slice = unsafe {
        voltages_arr
            .as_slice_mut()
            .expect("newly allocated voltages array must be contiguous")
    };

    for i in 0..n_steps {
        let (s, v) = lif.step(leak_k, gain_k, i_t, noise_in);
        spikes_slice[i] = s;
        voltages_slice[i] = v;
    }

    (spikes_arr, voltages_arr)
}

/// Run N independent LIF neurons in parallel, each with its own constant input.
///
/// Returns (spikes: ndarray[i32, (n_neurons, n_steps)],
///          voltages: ndarray[i16, (n_neurons, n_steps)]).
#[pyfunction]
#[pyo3(signature = (
    n_neurons,
    n_steps,
    leak_k,
    gain_k,
    currents,
    data_width=16,
    fraction=8,
    v_rest=0,
    v_reset=0,
    v_threshold=256,
    refractory_period=2
))]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::type_complexity)]
fn batch_lif_run_multi<'py>(
    py: Python<'py>,
    n_neurons: usize,
    n_steps: usize,
    leak_k: i16,
    gain_k: i16,
    currents: PyReadonlyArray1<'py, i16>,
    data_width: u32,
    fraction: u32,
    v_rest: i16,
    v_reset: i16,
    v_threshold: i16,
    refractory_period: i32,
) -> PyResult<(Bound<'py, PyArray2<i32>>, Bound<'py, PyArray2<i16>>)> {
    use rayon::prelude::*;

    let curr_slice = currents
        .as_slice()
        .map_err(|e| PyValueError::new_err(format!("Cannot read currents: {e}")))?;
    if curr_slice.len() != n_neurons {
        return Err(PyValueError::new_err(format!(
            "currents length {} does not match n_neurons {}.",
            curr_slice.len(),
            n_neurons
        )));
    }

    let spikes_arr = PyArray2::<i32>::zeros(py, [n_neurons, n_steps], false);
    let voltages_arr = PyArray2::<i16>::zeros(py, [n_neurons, n_steps], false);

    if n_neurons == 0 || n_steps == 0 {
        return Ok((spikes_arr, voltages_arr));
    }

    // SAFETY: Arrays are newly allocated and contiguous.
    let spikes_flat = unsafe {
        spikes_arr
            .as_slice_mut()
            .expect("newly allocated spikes array must be contiguous")
    };
    // SAFETY: Arrays are newly allocated and contiguous.
    let voltages_flat = unsafe {
        voltages_arr
            .as_slice_mut()
            .expect("newly allocated voltages array must be contiguous")
    };

    spikes_flat
        .par_chunks_mut(n_steps)
        .zip(voltages_flat.par_chunks_mut(n_steps))
        .zip(curr_slice.par_iter().copied())
        .for_each(|((spike_row, voltage_row), i_t)| {
            let mut lif = neuron::FixedPointLif::new(
                data_width,
                fraction,
                v_rest,
                v_reset,
                v_threshold,
                refractory_period,
            );
            for step in 0..n_steps {
                let (s, v) = lif.step(leak_k, gain_k, i_t, 0);
                spike_row[step] = s;
                voltage_row[step] = v;
            }
        });

    Ok((spikes_arr, voltages_arr))
}

/// Run a LIF neuron for N steps with per-step current and optional noise arrays.
#[pyfunction]
#[pyo3(signature = (
    leak_k,
    gain_k,
    currents,
    noises=None,
    data_width=16,
    fraction=8,
    v_rest=0,
    v_reset=0,
    v_threshold=256,
    refractory_period=2
))]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::type_complexity)]
fn batch_lif_run_varying<'py>(
    py: Python<'py>,
    leak_k: i16,
    gain_k: i16,
    currents: PyReadonlyArray1<'py, i16>,
    noises: Option<PyReadonlyArray1<'py, i16>>,
    data_width: u32,
    fraction: u32,
    v_rest: i16,
    v_reset: i16,
    v_threshold: i16,
    refractory_period: i32,
) -> PyResult<(Bound<'py, PyArray1<i32>>, Bound<'py, PyArray1<i16>>)> {
    let curr_slice = currents
        .as_slice()
        .map_err(|e| PyValueError::new_err(format!("Cannot read currents: {e}")))?;
    let noise_slice: Option<&[i16]> = match noises.as_ref() {
        Some(n) => Some(
            n.as_slice()
                .map_err(|e| PyValueError::new_err(format!("Cannot read noises: {e}")))?,
        ),
        None => None,
    };

    let n_steps = curr_slice.len();
    if let Some(ns) = noise_slice {
        if ns.len() != n_steps {
            return Err(PyValueError::new_err(format!(
                "noises length {} does not match currents length {}.",
                ns.len(),
                n_steps
            )));
        }
    }

    let mut lif = neuron::FixedPointLif::new(
        data_width,
        fraction,
        v_rest,
        v_reset,
        v_threshold,
        refractory_period,
    );
    let spikes_arr = PyArray1::<i32>::zeros(py, n_steps, false);
    let voltages_arr = PyArray1::<i16>::zeros(py, n_steps, false);

    // SAFETY: Arrays are newly allocated and contiguous.
    let spikes_slice = unsafe {
        spikes_arr
            .as_slice_mut()
            .expect("newly allocated spikes array must be contiguous")
    };
    // SAFETY: Arrays are newly allocated and contiguous.
    let voltages_slice = unsafe {
        voltages_arr
            .as_slice_mut()
            .expect("newly allocated voltages array must be contiguous")
    };

    for i in 0..n_steps {
        let noise_in = noise_slice.map_or(0, |ns| ns[i]);
        let (s, v) = lif.step(leak_k, gain_k, curr_slice[i], noise_in);
        spikes_slice[i] = s;
        voltages_slice[i] = v;
    }

    Ok((spikes_arr, voltages_arr))
}
