// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Engine Crate Root

#![allow(
    clippy::useless_conversion,
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    deprecated
)]

use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2,
    PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

pub mod adc_to_spike;
#[path = "bindings/adc_to_spike.rs"]
mod adc_to_spike_binding;
pub mod analysis;
pub mod attention;
pub mod bitstream;
#[path = "bindings/bitstream.rs"]
mod bitstream_binding;
#[path = "bindings/escape_rate.rs"]
mod escape_rate_binding;
#[path = "bindings/evolution.rs"]
mod evolution_binding;
#[path = "bindings/exp_if.rs"]
mod exp_if_binding;
#[path = "bindings/hdc.rs"]
mod hdc_binding;
pub use hdc_binding::PyBitStreamTensor;
pub mod brunel;
#[path = "bindings/cazelles_map.rs"]
mod cazelles_map_binding;
#[path = "bindings/chialvo_map.rs"]
mod chialvo_map_binding;
#[path = "bindings/coba_lif.rs"]
mod coba_lif_binding;
pub mod connectome;
pub mod conv;
pub mod cordiv;
#[path = "bindings/cordiv.rs"]
mod cordiv_binding;
pub mod cortical_column;
#[path = "bindings/cortical_column.rs"]
mod cortical_column_binding;
pub mod cortical_inject;
#[path = "bindings/cortical_inject.rs"]
mod cortical_inject_binding;
#[path = "bindings/courage_nekorkin_map.rs"]
mod courage_nekorkin_map_binding;
#[path = "bindings/dcls.rs"]
mod dcls_binding;
pub mod dna;
pub mod ei_network;
#[path = "bindings/ei_network.rs"]
mod ei_network_binding;
pub mod encoder;
#[path = "bindings/ermentrout_kopell_map.rs"]
mod ermentrout_kopell_map_binding;
pub mod evo;
pub mod fault;
#[path = "bindings/fault.rs"]
mod fault_binding;
#[path = "bindings/fitzhugh_nagumo.rs"]
mod fitzhugh_nagumo_binding;
#[path = "bindings/fitzhugh_rinzel.rs"]
mod fitzhugh_rinzel_binding;
#[path = "bindings/fixed_point_lif.rs"]
mod fixed_point_lif_binding;
pub mod fusion;
#[path = "bindings/glif.rs"]
mod glif_binding;
#[cfg(feature = "gpu")]
pub mod gpu;
pub mod grad;
pub mod graph;
#[path = "bindings/hindmarsh_rose.rs"]
mod hindmarsh_rose_binding;
#[path = "bindings/ibarz_tanaka_map.rs"]
mod ibarz_tanaka_map_binding;
#[path = "bindings/iqif.rs"]
mod iqif_binding;
pub mod ir;
#[path = "bindings/izhikevich2007.rs"]
mod izhikevich2007_binding;
#[path = "bindings/izhikevich.rs"]
mod izhikevich_binding;
pub mod layer;
pub(crate) mod learning_bindings;
pub mod lgssm;
#[path = "bindings/lgssm.rs"]
mod lgssm_binding;
#[path = "bindings/mckean.rs"]
mod mckean_binding;
#[path = "bindings/medvedev_map.rs"]
mod medvedev_map_binding;
#[path = "bindings/mihalas_niebur.rs"]
mod mihalas_niebur_binding;
#[path = "bindings/mixed_dense.rs"]
mod mixed_dense_binding;
pub mod network_runner;
#[path = "bindings/network_runner.rs"]
mod network_runner_binding;
pub mod neuron;
pub mod neurons;
#[path = "bindings/ollivier_ricci.rs"]
mod ollivier_ricci_binding;
pub mod optimizer;
#[path = "bindings/optimizer.rs"]
mod optimizer_binding;
pub mod partition;
#[path = "bindings/partition.rs"]
mod partition_binding;
#[path = "bindings/pernarowski.rs"]
mod pernarowski_binding;
pub mod phi;
#[path = "bindings/phi.rs"]
mod phi_binding;
pub mod photonic;
pub mod ping;
#[path = "bindings/ping.rs"]
mod ping_binding;
#[path = "bindings/poisson.rs"]
mod poisson_binding;
pub mod predictive_coding;
#[path = "bindings/predictive_coding.rs"]
mod predictive_coding_binding;
pub mod pyo3_neurons;
pub mod quantum;
pub mod rall_dendrite;
#[path = "bindings/rall_dendrite.rs"]
mod rall_dendrite_binding;
pub mod recorder;
pub mod recurrent;
pub mod rk4_neurons;
#[path = "bindings/rulkov_map.rs"]
mod rulkov_map_binding;
#[path = "bindings/runtime_control.rs"]
mod runtime_control_binding;
pub mod sc_inference;
#[path = "bindings/sc_inference.rs"]
mod sc_inference_binding;
pub mod scpn;
pub mod simd;
pub mod sobol;
#[path = "bindings/stdp_synapse.rs"]
mod stdp_synapse_binding;
#[cfg(feature = "z3")]
pub mod supervisor;
pub mod synapses;
#[path = "bindings/terman_wang.rs"]
mod terman_wang_binding;
pub mod topology;
pub mod wilson_cowan;
#[path = "bindings/wilson_cowan.rs"]
mod wilson_cowan_binding;
#[path = "bindings/wilson_hr.rs"]
mod wilson_hr_binding;
pub mod wong_wang;

// ── Brunel Network PyO3 wrapper ──────────────────────────────────────

#[pyclass(
    name = "BrunelNetwork",
    module = "sc_neurocore_engine.sc_neurocore_engine"
)]
pub struct PyBrunelNetwork {
    inner: brunel::BrunelNetwork,
}

#[pymethods]
impl PyBrunelNetwork {
    #[new]
    #[pyo3(signature = (
        n_neurons,
        w_indptr,
        w_indices,
        w_data,
        leak_k,
        gain_k,
        ext_lambda,
        ext_weight_fp,
        data_width=16,
        fraction=8,
        v_rest=0,
        v_reset=0,
        v_threshold=256,
        refractory_period=2,
        seed=42
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        n_neurons: usize,
        w_indptr: PyReadonlyArray1<'_, i64>,
        w_indices: PyReadonlyArray1<'_, i64>,
        w_data: PyReadonlyArray1<'_, i16>,
        leak_k: i16,
        gain_k: i16,
        ext_lambda: f64,
        ext_weight_fp: i16,
        data_width: u32,
        fraction: u32,
        v_rest: i16,
        v_reset: i16,
        v_threshold: i16,
        refractory_period: i32,
        seed: u64,
    ) -> PyResult<Self> {
        let indptr = w_indptr
            .as_slice()
            .map_err(|e| PyValueError::new_err(format!("Cannot read w_indptr: {e}")))?;
        let indices = w_indices
            .as_slice()
            .map_err(|e| PyValueError::new_err(format!("Cannot read w_indices: {e}")))?;
        let data = w_data
            .as_slice()
            .map_err(|e| PyValueError::new_err(format!("Cannot read w_data: {e}")))?;

        let row_offsets: Vec<usize> = indptr.iter().map(|&v| v as usize).collect();
        let col_indices: Vec<usize> = indices.iter().map(|&v| v as usize).collect();
        let values: Vec<i16> = data.to_vec();

        let inner = brunel::BrunelNetwork::new(
            n_neurons,
            row_offsets,
            col_indices,
            values,
            data_width,
            fraction,
            v_rest,
            v_reset,
            v_threshold,
            refractory_period,
            leak_k,
            gain_k,
            ext_lambda,
            ext_weight_fp,
            seed,
        )
        .map_err(PyValueError::new_err)?;

        Ok(Self { inner })
    }

    fn run<'py>(&mut self, py: Python<'py>, n_steps: usize) -> Bound<'py, PyArray1<u32>> {
        let counts = self.inner.run(n_steps);
        counts.into_pyarray(py)
    }
}

/// SC-NeuroCore ─ High-Performance Rust Engine

#[pymodule]
fn sc_neurocore_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    runtime_control_binding::register(m)?;
    bitstream_binding::register(m)?;
    fixed_point_lif_binding::register(m)?;
    dcls_binding::register(m)?;
    mixed_dense_binding::register(m)?;
    adc_to_spike_binding::register(m)?;
    sc_inference_binding::register(m)?;
    wilson_cowan_binding::register(m)?;
    m.add_class::<DenseLayer>()?;
    #[cfg(feature = "gpu")]
    m.add_class::<gpu::PyGpuDenseLayer>()?;
    #[cfg(feature = "gpu")]
    m.add_class::<gpu::PyGpuLifBatch>()?;
    #[cfg(feature = "gpu")]
    m.add_class::<gpu::PyGpuKuramoto>()?;
    #[cfg(feature = "gpu")]
    m.add_class::<gpu::PyGpuIzhikevichBatch>()?;
    stdp_synapse_binding::register(m)?;
    learning_bindings::register(m)?;
    m.add_class::<PyKuramotoSolver>()?;
    m.add_class::<PySCPNMetrics>()?;
    hdc_binding::register(m)?;
    m.add_class::<PyBrunelNetwork>()?;
    izhikevich_binding::register(m)?;
    ir::bindings::register(m)?;
    exp_if_binding::register(m)?;
    pyo3_neurons::register_neuron_classes(m)?;
    network_runner_binding::register(m)?;
    #[cfg(feature = "z3")]
    m.add_class::<supervisor::PySpikingControllerPool>()?;
    ei_network_binding::register(m)?;
    m.add_function(wrap_pyfunction!(rk4_neurons::py_rk4_neuron_simulate, m)?)?;
    cordiv_binding::register(m)?;
    predictive_coding_binding::register(m)?;
    phi_binding::register(m)?;
    cortical_column_binding::register(m)?;
    rall_dendrite_binding::register(m)?;
    analysis::bindings::register(m)?;
    dna::bindings::register(m)?;
    quantum::bindings::register(m)?;
    // Photonic NoC acceleration
    photonic::bindings::register(m)?;
    optimizer_binding::register(m)?;
    evolution_binding::register(m)?;
    lgssm_binding::register(m)?;
    ollivier_ricci_binding::register(m)?;
    chialvo_map_binding::register(m)?;
    cazelles_map_binding::register(m)?;
    courage_nekorkin_map_binding::register(m)?;
    mckean_binding::register(m)?;
    wilson_hr_binding::register(m)?;
    pernarowski_binding::register(m)?;
    terman_wang_binding::register(m)?;
    coba_lif_binding::register(m)?;
    escape_rate_binding::register(m)?;
    poisson_binding::register(m)?;
    iqif_binding::register(m)?;
    mihalas_niebur_binding::register(m)?;
    glif_binding::register(m)?;
    rulkov_map_binding::register(m)?;
    ibarz_tanaka_map_binding::register(m)?;
    medvedev_map_binding::register(m)?;
    ermentrout_kopell_map_binding::register(m)?;
    fitzhugh_nagumo_binding::register(m)?;
    hindmarsh_rose_binding::register(m)?;
    fitzhugh_rinzel_binding::register(m)?;
    izhikevich2007_binding::register(m)?;
    fault_binding::register(m)?;
    partition_binding::register(m)?;
    ping_binding::register(m)?;
    cortical_inject_binding::register(m)?;
    Ok(())
}

// ── DenseLayer ──────────────────────────────────────────────────

#[pyclass(module = "sc_neurocore_engine.sc_neurocore_engine")]
pub struct DenseLayer {
    inner: layer::DenseLayer,
}

#[pymethods]
impl DenseLayer {
    #[new]
    #[pyo3(signature = (n_inputs, n_neurons, length=1024, seed=24301))]
    fn new(n_inputs: usize, n_neurons: usize, length: usize, seed: u64) -> Self {
        Self {
            inner: layer::DenseLayer::new(n_inputs, n_neurons, length, seed),
        }
    }

    fn get_weights(&self) -> Vec<Vec<f64>> {
        self.inner.get_weights()
    }

    fn set_weights(&mut self, weights: Vec<Vec<f64>>) -> PyResult<()> {
        self.inner
            .set_weights(weights)
            .map_err(PyValueError::new_err)
    }

    fn refresh_packed_weights(&mut self) {
        self.inner.refresh_packed_weights();
    }

    #[pyo3(signature = (input_values, seed=44257))]
    fn forward(&self, input_values: Vec<f64>, seed: u64) -> PyResult<Vec<f64>> {
        self.inner
            .forward(&input_values, seed)
            .map_err(PyValueError::new_err)
    }

    #[pyo3(signature = (input_values, seed=44257))]
    fn forward_fast(&self, input_values: Vec<f64>, seed: u64) -> PyResult<Vec<f64>> {
        self.inner
            .forward_fast(&input_values, seed)
            .map_err(PyValueError::new_err)
    }

    /// Dense forward accepting numpy input and returning numpy output.
    ///
    /// This performs parallel encoding + parallel compute in one FFI call.
    #[pyo3(signature = (input_values, seed=44257))]
    fn forward_numpy<'py>(
        &self,
        py: Python<'py>,
        input_values: PyReadonlyArray1<'py, f64>,
        seed: u64,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let slice = input_values
            .as_slice()
            .map_err(|e| PyValueError::new_err(format!("Cannot read input array: {e}")))?;
        let out = self
            .inner
            .forward_numpy_inner(slice, seed)
            .map_err(PyValueError::new_err)?;
        Ok(out.into_pyarray(py))
    }

    /// Dense forward for a batch of input samples in one FFI call.
    ///
    /// `inputs` must be a contiguous float64 array of shape (n_samples, n_inputs).
    /// Returns float64 array of shape (n_samples, n_neurons).
    #[pyo3(signature = (inputs, seed=44257))]
    fn forward_batch_numpy<'py>(
        &self,
        py: Python<'py>,
        inputs: PyReadonlyArray2<'py, f64>,
        seed: u64,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let shape = inputs.shape();
        let n_samples = shape[0];
        let n_inputs = shape[1];
        if n_inputs != self.inner.n_inputs {
            return Err(PyValueError::new_err(format!(
                "Expected {} input features, got {}.",
                self.inner.n_inputs, n_inputs
            )));
        }

        let flat_inputs = inputs
            .as_slice()
            .map_err(|e| PyValueError::new_err(format!("Array not contiguous: {e}")))?;
        let out = PyArray2::<f64>::zeros(py, [n_samples, self.inner.n_neurons], false);
        // SAFETY: Newly allocated numpy arrays are contiguous.
        let out_slice = unsafe {
            out.as_slice_mut()
                .expect("newly allocated output array must be contiguous")
        };

        self.inner
            .forward_batch_into(flat_inputs, n_samples, seed, out_slice)
            .map_err(PyValueError::new_err)?;
        Ok(out)
    }

    /// Forward pass with pre-packed input bitstreams.
    ///
    /// Accepts either:
    /// - 2-D numpy array of dtype uint64 with shape (n_inputs, words)
    /// - list[list[int]]
    fn forward_prepacked(&self, packed_inputs: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
        if let Ok(arr) = packed_inputs.extract::<PyReadonlyArray2<u64>>() {
            let view = arr.as_array();
            let rows: Vec<Vec<u64>> = (0..view.nrows()).map(|i| view.row(i).to_vec()).collect();
            return self
                .inner
                .forward_prepacked(&rows)
                .map_err(PyValueError::new_err);
        }

        let rows = packed_inputs.extract::<Vec<Vec<u64>>>().map_err(|_| {
            PyValueError::new_err(
                "packed_inputs must be a 2-D numpy uint64 array or list[list[int]].",
            )
        })?;
        self.inner
            .forward_prepacked(&rows)
            .map_err(PyValueError::new_err)
    }

    /// Dense forward with pre-packed numpy 2-D input (true zero-copy).
    ///
    /// Accepts a contiguous numpy uint64 array of shape (n_inputs, words).
    /// This avoids all row-copying that the `forward_prepacked` method does.
    #[pyo3(signature = (packed_inputs,))]
    fn forward_prepacked_numpy<'py>(
        &self,
        py: Python<'py>,
        packed_inputs: PyReadonlyArray2<'py, u64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let shape = packed_inputs.shape();
        let n_inputs = shape[0];
        let words = shape[1];
        let flat = packed_inputs
            .as_slice()
            .map_err(|e| PyValueError::new_err(format!("Array not contiguous: {e}")))?;
        let out = self
            .inner
            .forward_prepacked_2d(flat, n_inputs, words)
            .map_err(PyValueError::new_err)?;
        Ok(out.into_pyarray(py))
    }
}

fn extract_matrix_f64(data: &Bound<'_, PyAny>, name: &str) -> PyResult<(Vec<f64>, usize, usize)> {
    if let Ok(rows) = data.extract::<Vec<Vec<f64>>>() {
        if rows.is_empty() {
            return Err(PyValueError::new_err(format!(
                "{} must not be an empty matrix.",
                name
            )));
        }
        let row_count = rows.len();
        let cols = rows[0].len();
        if cols == 0 {
            return Err(PyValueError::new_err(format!(
                "{} must not have zero columns.",
                name
            )));
        }
        if rows.iter().any(|r| r.len() != cols) {
            return Err(PyValueError::new_err(format!(
                "{} must be a rectangular matrix.",
                name
            )));
        }
        let out = rows.into_iter().flatten().collect::<Vec<f64>>();
        return Ok((out, row_count, cols));
    }

    if let Ok(flat) = data.extract::<Vec<f64>>() {
        if flat.is_empty() {
            return Err(PyValueError::new_err(format!(
                "{} must not be an empty vector.",
                name
            )));
        }
        let cols = flat.len();
        return Ok((flat, 1, cols));
    }

    Err(PyValueError::new_err(format!(
        "{} must be a 1-D or 2-D float array.",
        name
    )))
}

fn reshape_flat_to_rows(flat: Vec<f64>, rows: usize, cols: usize) -> Vec<Vec<f64>> {
    let mut out = Vec::with_capacity(rows);
    for i in 0..rows {
        out.push(flat[i * cols..(i + 1) * cols].to_vec());
    }
    out
}

#[pyclass(
    name = "KuramotoSolver",
    module = "sc_neurocore_engine.sc_neurocore_engine"
)]
pub struct PyKuramotoSolver {
    inner: scpn::KuramotoSolver,
}

fn validate_kuramoto_finite(name: &str, values: &[f64]) -> PyResult<()> {
    if values.iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(PyValueError::new_err(format!(
            "{name} values must be finite"
        )))
    }
}

fn validate_kuramoto_dt(dt: f64) -> PyResult<()> {
    if dt.is_finite() && dt > 0.0 {
        Ok(())
    } else {
        Err(PyValueError::new_err("dt must be finite and positive"))
    }
}

fn validate_kuramoto_matrix_shape(
    name: &str,
    values_len: usize,
    rows: usize,
    cols: usize,
    n: usize,
) -> PyResult<()> {
    let is_absent = rows == 0 && cols == 0 && values_len == 0;
    let is_flat = rows == 1 && values_len == n * n;
    let is_square = rows == n && cols == n;
    if is_absent || is_flat || is_square {
        Ok(())
    } else {
        Err(PyValueError::new_err(format!(
            "{name} must be shape ({n}, {n}) or flat length {}",
            n * n
        )))
    }
}

#[pymethods]
impl PyKuramotoSolver {
    #[getter]
    fn phases(&self) -> Vec<f64> {
        self.inner.phases.clone()
    }
    #[new]
    #[pyo3(signature = (omega, coupling, phases, noise_amp=0.1))]
    fn new(
        omega: Vec<f64>,
        coupling: &Bound<'_, PyAny>,
        phases: Vec<f64>,
        noise_amp: f64,
    ) -> PyResult<Self> {
        let n = omega.len();
        if n == 0 {
            return Err(PyValueError::new_err("omega must not be empty."));
        }
        if phases.len() != n {
            return Err(PyValueError::new_err(format!(
                "phases length mismatch: got {}, expected {}.",
                phases.len(),
                n
            )));
        }
        validate_kuramoto_finite("omega", &omega)?;
        validate_kuramoto_finite("initial_phases", &phases)?;
        if !(noise_amp.is_finite() && noise_amp >= 0.0) {
            return Err(PyValueError::new_err(
                "noise_amp must be finite and non-negative",
            ));
        }

        let (coupling_flat, rows, cols) = extract_matrix_f64(coupling, "coupling")?;
        if rows == 1 {
            if coupling_flat.len() != n * n {
                return Err(PyValueError::new_err(format!(
                    "Flat coupling length mismatch: got {}, expected {}.",
                    coupling_flat.len(),
                    n * n
                )));
            }
        } else if rows != n || cols != n {
            return Err(PyValueError::new_err(format!(
                "coupling must be shape ({}, {}) or flat length {}, got ({}, {}).",
                n,
                n,
                n * n,
                rows,
                cols
            )));
        }
        validate_kuramoto_finite("coupling", &coupling_flat)?;

        Ok(Self {
            inner: scpn::KuramotoSolver::new(omega, coupling_flat, phases, noise_amp),
        })
    }

    #[pyo3(signature = (dt, seed=0))]
    fn step(&mut self, dt: f64, seed: u64) -> PyResult<f64> {
        validate_kuramoto_dt(dt)?;
        Ok(self.inner.step(dt, seed))
    }

    #[pyo3(signature = (n_steps, dt, seed=0))]
    fn run(&mut self, n_steps: usize, dt: f64, seed: u64) -> PyResult<Vec<f64>> {
        validate_kuramoto_dt(dt)?;
        Ok(self.inner.run(n_steps, dt, seed))
    }

    fn set_field_pressure(&mut self, f: f64) -> PyResult<()> {
        if !f.is_finite() {
            return Err(PyValueError::new_err("field_pressure must be finite"));
        }
        self.inner.set_field_pressure(f);
        Ok(())
    }

    #[pyo3(signature = (
        dt,
        seed=0,
        W=None,
        sigma_g=0.0,
        h_munu=None,
        pgbo_weight=0.0,
    ))]
    #[allow(non_snake_case)]
    fn step_ssgf(
        &mut self,
        dt: f64,
        seed: u64,
        W: Option<&Bound<'_, PyAny>>,
        sigma_g: f64,
        h_munu: Option<&Bound<'_, PyAny>>,
        pgbo_weight: f64,
    ) -> PyResult<f64> {
        validate_kuramoto_dt(dt)?;
        if !sigma_g.is_finite() {
            return Err(PyValueError::new_err("sigma_g must be finite"));
        }
        if !pgbo_weight.is_finite() {
            return Err(PyValueError::new_err("pgbo_weight must be finite"));
        }
        let (w_flat, w_rows, w_cols) = match W {
            Some(w) => extract_matrix_f64(w, "W")?,
            None => (vec![], 0, 0),
        };
        let (h_flat, h_rows, h_cols) = match h_munu {
            Some(h) => extract_matrix_f64(h, "h_munu")?,
            None => (vec![], 0, 0),
        };
        validate_kuramoto_matrix_shape("W", w_flat.len(), w_rows, w_cols, self.inner.n)?;
        validate_kuramoto_matrix_shape("h_munu", h_flat.len(), h_rows, h_cols, self.inner.n)?;
        validate_kuramoto_finite("w_flat", &w_flat)?;
        validate_kuramoto_finite("h_flat", &h_flat)?;
        Ok(self
            .inner
            .step_ssgf(dt, seed, &w_flat, sigma_g, &h_flat, pgbo_weight))
    }

    #[pyo3(signature = (
        n_steps,
        dt,
        seed=0,
        W=None,
        sigma_g=0.0,
        h_munu=None,
        pgbo_weight=0.0,
    ))]
    #[allow(clippy::too_many_arguments, non_snake_case)]
    fn run_ssgf(
        &mut self,
        n_steps: usize,
        dt: f64,
        seed: u64,
        W: Option<&Bound<'_, PyAny>>,
        sigma_g: f64,
        h_munu: Option<&Bound<'_, PyAny>>,
        pgbo_weight: f64,
    ) -> PyResult<Vec<f64>> {
        validate_kuramoto_dt(dt)?;
        if !sigma_g.is_finite() {
            return Err(PyValueError::new_err("sigma_g must be finite"));
        }
        if !pgbo_weight.is_finite() {
            return Err(PyValueError::new_err("pgbo_weight must be finite"));
        }
        let (w_flat, w_rows, w_cols) = match W {
            Some(w) => extract_matrix_f64(w, "W")?,
            None => (vec![], 0, 0),
        };
        let (h_flat, h_rows, h_cols) = match h_munu {
            Some(h) => extract_matrix_f64(h, "h_munu")?,
            None => (vec![], 0, 0),
        };
        validate_kuramoto_matrix_shape("W", w_flat.len(), w_rows, w_cols, self.inner.n)?;
        validate_kuramoto_matrix_shape("h_munu", h_flat.len(), h_rows, h_cols, self.inner.n)?;
        validate_kuramoto_finite("w_flat", &w_flat)?;
        validate_kuramoto_finite("h_flat", &h_flat)?;
        Ok(self
            .inner
            .run_ssgf(n_steps, dt, seed, &w_flat, sigma_g, &h_flat, pgbo_weight))
    }

    fn order_parameter(&self) -> f64 {
        self.inner.order_parameter()
    }

    fn apply_phases(&mut self, phases: Vec<f64>) -> PyResult<()> {
        if phases.len() != self.inner.n {
            return Err(PyValueError::new_err(format!(
                "phases length mismatch: got {}, expected {}.",
                phases.len(),
                self.inner.n
            )));
        }
        validate_kuramoto_finite("phases", &phases)?;
        self.inner.set_phases(phases);
        Ok(())
    }

    fn set_phases(&mut self, phases: Vec<f64>) -> PyResult<()> {
        self.apply_phases(phases)
    }

    #[setter(phases)]
    fn set_phases_attr(&mut self, phases: Vec<f64>) -> PyResult<()> {
        self.apply_phases(phases)
    }

    fn set_coupling(&mut self, coupling: &Bound<'_, PyAny>) -> PyResult<()> {
        let n = self.inner.n;
        let (coupling_flat, rows, cols) = extract_matrix_f64(coupling, "coupling")?;
        if rows == 1 {
            if coupling_flat.len() != n * n {
                return Err(PyValueError::new_err(format!(
                    "Flat coupling length mismatch: got {}, expected {}.",
                    coupling_flat.len(),
                    n * n
                )));
            }
        } else if rows != n || cols != n {
            return Err(PyValueError::new_err(format!(
                "coupling must be shape ({}, {}) or flat length {}, got ({}, {}).",
                n,
                n,
                n * n,
                rows,
                cols
            )));
        }
        validate_kuramoto_finite("coupling", &coupling_flat)?;
        self.inner.set_coupling(coupling_flat);
        Ok(())
    }
}

#[pyclass(
    name = "SCPNMetrics",
    module = "sc_neurocore_engine.sc_neurocore_engine"
)]
pub struct PySCPNMetrics;

#[pymethods]
impl PySCPNMetrics {
    #[new]
    fn new() -> Self {
        Self
    }

    #[staticmethod]
    fn global_coherence(weights: [f64; 7], metrics: [f64; 7]) -> f64 {
        scpn::SCPNMetrics::global_coherence(&weights, &metrics)
    }

    #[staticmethod]
    fn consciousness_index(phases_l4: Vec<f64>, glyph_l7: [f64; 6]) -> f64 {
        scpn::SCPNMetrics::consciousness_index(&phases_l4, &glyph_l7)
    }
}
