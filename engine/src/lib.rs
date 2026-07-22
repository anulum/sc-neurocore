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
    PyReadwriteArray1, PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::IntoPyObject;

pub mod adc_to_spike;
pub mod analysis;
pub mod attention;
pub mod bitstream;
#[path = "bindings/bitstream.rs"]
mod bitstream_binding;
#[path = "bindings/escape_rate.rs"]
mod escape_rate_binding;
#[path = "bindings/evolution.rs"]
mod evolution_binding;
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
pub mod cortical_inject;
#[path = "bindings/courage_nekorkin_map.rs"]
mod courage_nekorkin_map_binding;
pub mod dna;
pub mod ei_network;
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
pub mod layer;
pub(crate) mod learning_bindings;
pub mod lgssm;
#[path = "bindings/mckean.rs"]
mod mckean_binding;
#[path = "bindings/medvedev_map.rs"]
mod medvedev_map_binding;
#[path = "bindings/mihalas_niebur.rs"]
mod mihalas_niebur_binding;
pub mod network_runner;
pub mod neuron;
pub mod neurons;
#[path = "bindings/ollivier_ricci.rs"]
mod ollivier_ricci_binding;
pub mod optimizer;
#[path = "bindings/optimizer.rs"]
mod optimizer_binding;
pub mod partition;
#[path = "bindings/pernarowski.rs"]
mod pernarowski_binding;
pub mod phi;
#[path = "bindings/phi.rs"]
mod phi_binding;
pub mod photonic;
pub mod ping;
#[path = "bindings/poisson.rs"]
mod poisson_binding;
pub mod predictive_coding;
#[path = "bindings/predictive_coding.rs"]
mod predictive_coding_binding;
pub mod pyo3_neurons;
pub mod quantum;
pub mod rall_dendrite;
pub mod recorder;
pub mod recurrent;
pub mod rk4_neurons;
#[path = "bindings/rulkov_map.rs"]
mod rulkov_map_binding;
pub mod sc_inference;
pub mod scpn;
pub mod simd;
pub mod sobol;
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

#[pyclass(
    name = "StdpSynapse",
    module = "sc_neurocore_engine.sc_neurocore_engine"
)]
pub struct StdpSynapse {
    inner: synapses::StdpSynapse,
}

#[pymethods]
impl StdpSynapse {
    #[new]
    #[pyo3(signature = (initial_weight, data_width=16, fraction=8))]
    fn new(initial_weight: i16, data_width: u32, fraction: u32) -> Self {
        Self {
            inner: synapses::StdpSynapse::new(initial_weight, data_width, fraction),
        }
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (pre_spike, post_spike, a_plus=16, a_minus=-16, decay=250, w_min=0, w_max=32767))]
    fn step(
        &mut self,
        pre_spike: bool,
        post_spike: bool,
        a_plus: i16,
        a_minus: i16,
        decay: i16,
        w_min: i16,
        w_max: i16,
    ) {
        let params = synapses::StdpParams {
            a_plus,
            a_minus,
            decay,
            w_min,
            w_max,
        };
        self.inner.step(pre_spike, post_spike, &params);
    }

    #[getter]
    fn weight(&self) -> i16 {
        self.inner.weight
    }

    #[setter]
    fn set_weight(&mut self, value: i16) {
        self.inner.weight = value;
    }
}

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

// ── NetworkRunner PyO3 wrapper ────────────────────────────────────────

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

// ── Batch model simulate PyO3 wrapper ────────────────────────────────

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

// ── E-I Network PyO3 wrapper ───────────────────────────────────────

#[pyfunction]
#[pyo3(signature = (
    n_exc=80, n_inh=20,
    w_ee=0.1, w_ei=0.4, w_ie=0.1, w_ii=0.4,
    p_conn=0.2, ext_rate=5.0,
    duration=200.0, dt=0.1, seed=42
))]
fn py_simulate_ei_network<'py>(
    py: Python<'py>,
    n_exc: usize,
    n_inh: usize,
    w_ee: f64,
    w_ei: f64,
    w_ie: f64,
    w_ii: f64,
    p_conn: f64,
    ext_rate: f64,
    duration: f64,
    dt: f64,
    seed: u64,
) -> PyResult<Py<PyAny>> {
    let r = ei_network::simulate_ei(
        n_exc, n_inh, w_ee, w_ei, w_ie, w_ii, p_conn, ext_rate, duration, dt, seed,
    );
    let n_spikes = r.spike_times.len();
    let d = PyDict::new(py);
    d.set_item("spike_times", r.spike_times.into_pyarray(py))?;
    d.set_item(
        "spike_neurons",
        r.spike_neurons
            .iter()
            .map(|&x| x as i64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    d.set_item("n_exc", r.n_exc)?;
    d.set_item("n_inh", r.n_inh)?;
    d.set_item("n_total", r.n_exc + r.n_inh)?;
    d.set_item("n_spikes", n_spikes)?;
    d.set_item("rate_time", r.rate_time.into_pyarray(py))?;
    d.set_item("exc_rates", r.exc_rates.into_pyarray(py))?;
    d.set_item("inh_rates", r.inh_rates.into_pyarray(py))?;
    d.set_item("duration", duration)?;
    d.set_item("dt", dt)?;
    d.set_item("mean_exc_rate", r.mean_exc_rate)?;
    d.set_item("mean_inh_rate", r.mean_inh_rate)?;
    Ok(d.into_any().unbind())
}

// ── DCLS-max Q8.8 tent kernel — batch PyO3 wrapper ───────────────────

/// Batched DCLS-max triangular (tent) contraction in bit-true Q8.8 arithmetic.
///
/// Parity contract with `sc_neurocore.scpn.dcls_tent_kernel`: this Rust path,
/// the Mojo, Julia and Go backends, and the Python floor all return
/// bit-identical arrays because the kernel is exact integer arithmetic.
///
/// `spikes` and `weights_q88` are row-major `n_channels * n_taps`; `centres_q88`
/// and `sigmas_q88` carry one learnable `(centre, sigma)` per output channel.
///
/// Returns a dict with keys `outputs_q88` (int16), `accumulators_q16_16`
/// (int32), `overflow` (bool), `active_tap_counts` (int64) and `max_gates_q88`
/// (int16), each a 1-D array of length `n_channels`.
#[pyfunction]
#[pyo3(signature = (spikes, weights_q88, centres_q88, sigmas_q88, n_taps))]
fn py_dcls_max_forward_batch_q88<'py>(
    py: Python<'py>,
    spikes: PyReadonlyArray1<'py, u8>,
    weights_q88: PyReadonlyArray1<'py, i16>,
    centres_q88: PyReadonlyArray1<'py, i16>,
    sigmas_q88: PyReadonlyArray1<'py, i16>,
    n_taps: usize,
) -> PyResult<Py<PyAny>> {
    let result = scpn::dcls_max_forward_batch_q88(
        spikes.as_slice()?,
        weights_q88.as_slice()?,
        centres_q88.as_slice()?,
        sigmas_q88.as_slice()?,
        n_taps,
    )
    .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let active_tap_counts: Vec<i64> = result.active_tap_counts.iter().map(|&c| c as i64).collect();
    let d = PyDict::new(py);
    d.set_item("outputs_q88", result.outputs_q88.into_pyarray(py))?;
    d.set_item(
        "accumulators_q16_16",
        result.accumulators_q16_16.into_pyarray(py),
    )?;
    d.set_item("overflow", result.overflow.into_pyarray(py))?;
    d.set_item("active_tap_counts", active_tap_counts.into_pyarray(py))?;
    d.set_item("max_gates_q88", result.max_gates_q88.into_pyarray(py))?;
    Ok(d.into_any().unbind())
}

// ── SC inference over pre-packed weights — PyO3 wrapper ──────────────

/// Stochastic forward pass over caller-owned packed weight bitstreams.
///
/// Parity contract with `sc_neurocore.accel.sc_forward`: this Rust path and the
/// NumPy fallback return bit-identical results for a fixed seed because the input
/// encoder is the deterministic 16-bit LFSR comparator.
///
/// `weights_packed` is row-major `n_out * n_in * n_words` (`n_words =
/// ceil(length / 64)`); `input_probs` is `n_in` float64 in `[0, 1]`. Returns an
/// `n_out` float64 array, the AND-then-popcount estimate of
/// `weights @ input_probs` divided by `length`.
#[pyfunction]
#[pyo3(signature = (weights_packed, n_out, n_in, n_words, input_probs, length, seed))]
#[allow(clippy::too_many_arguments)]
fn py_sc_forward_packed<'py>(
    py: Python<'py>,
    weights_packed: PyReadonlyArray1<'py, u64>,
    n_out: usize,
    n_in: usize,
    n_words: usize,
    input_probs: PyReadonlyArray1<'py, f64>,
    length: usize,
    seed: u64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let outputs = sc_inference::sc_forward_packed(
        weights_packed.as_slice()?,
        n_out,
        n_in,
        n_words,
        input_probs.as_slice()?,
        length,
        seed,
    )
    .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(outputs.into_pyarray(py))
}

// ── ADC-to-spike decimating rate-code — per-window PyO3 wrapper ───────

/// Encode raw ADC samples into per-window spike rate codes.
///
/// Parity contract with `sc_neurocore.sensors.adc_to_spike_kernel`: this Rust
/// path and the Julia, Go, Mojo and Python backends return bit-identical arrays
/// because the per-window quantise/average/rate-code arithmetic is exact integer.
///
/// `signed_input` is `0` for offset-binary or `1` for two's-complement ADC
/// samples. Returns a dict with `window_values_q` (int32), `spike_counts` (int32)
/// and `polarities` (bool), each of length `samples.len() / decimation`.
#[pyfunction]
#[pyo3(signature = (
    samples, adc_width, q_int, q_frac, decimation, signed_input, threshold_q,
))]
#[allow(clippy::too_many_arguments)]
fn py_adc_to_spike_windows<'py>(
    py: Python<'py>,
    samples: PyReadonlyArray1<'py, i64>,
    adc_width: u32,
    q_int: u32,
    q_frac: u32,
    decimation: u32,
    signed_input: i64,
    threshold_q: i64,
) -> PyResult<Py<PyAny>> {
    let result = adc_to_spike::adc_to_spike_windows(
        samples.as_slice()?,
        adc_width,
        q_int,
        q_frac,
        decimation,
        signed_input != 0,
        threshold_q,
    )
    .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let d = PyDict::new(py);
    d.set_item("window_values_q", result.window_values_q.into_pyarray(py))?;
    d.set_item("spike_counts", result.spike_counts.into_pyarray(py))?;
    d.set_item("polarities", result.polarities.into_pyarray(py))?;
    Ok(d.into_any().unbind())
}

// ── Mixed-precision Q8.8 × Q16.16 dense MAC — batch PyO3 wrapper ──────

/// Batched integer mixed-precision Q8.8 × Q16.16 dense MAC.
///
/// Parity contract with `sc_neurocore.compiler.mixed_dense_kernel`: this Rust
/// path and the Julia, Go, Mojo and Python backends return bit-identical arrays
/// because the integer branch (divisor equal to the Q8.8 weight scale) is exact.
///
/// `weights_q88` is row-major `n_outputs * n_inputs`; `inputs_q1616` is row-major
/// `n_batch * n_inputs`. Returns a dict with `outputs_q1616` (int32), `overflow`
/// (bool) and `underflow` (bool), each a 1-D array of length `n_batch * n_outputs`.
#[pyfunction]
#[pyo3(signature = (weights_q88, inputs_q1616, n_outputs, n_inputs))]
fn py_mixed_dense_forward_batch_q88_q1616<'py>(
    py: Python<'py>,
    weights_q88: PyReadonlyArray1<'py, i16>,
    inputs_q1616: PyReadonlyArray1<'py, i32>,
    n_outputs: usize,
    n_inputs: usize,
) -> PyResult<Py<PyAny>> {
    let result = crate::ir::qformat::mixed_dense_forward_batch_q88_q1616(
        weights_q88.as_slice()?,
        inputs_q1616.as_slice()?,
        n_outputs,
        n_inputs,
    )
    .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let d = PyDict::new(py);
    d.set_item("outputs_q1616", result.outputs_q1616.into_pyarray(py))?;
    d.set_item("overflow", result.overflow.into_pyarray(py))?;
    d.set_item("underflow", result.underflow.into_pyarray(py))?;
    Ok(d.into_any().unbind())
}

/// SC-NeuroCore ─ High-Performance Rust Engine

#[pymodule]
fn sc_neurocore_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(simd_tier, m)?)?;
    m.add_function(wrap_pyfunction!(set_num_threads, m)?)?;
    m.add_function(wrap_pyfunction!(pack_bitstream, m)?)?;
    m.add_function(wrap_pyfunction!(unpack_bitstream, m)?)?;
    bitstream_binding::register(m)?;
    m.add_function(wrap_pyfunction!(pack_bitstream_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(popcount_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(unpack_bitstream_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(batch_lif_run, m)?)?;
    m.add_function(wrap_pyfunction!(batch_lif_run_multi, m)?)?;
    m.add_function(wrap_pyfunction!(batch_lif_run_varying, m)?)?;
    m.add_function(wrap_pyfunction!(batch_encode, m)?)?;
    m.add_function(wrap_pyfunction!(batch_encode_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(py_dcls_max_forward_batch_q88, m)?)?;
    m.add_function(wrap_pyfunction!(py_mixed_dense_forward_batch_q88_q1616, m)?)?;
    m.add_function(wrap_pyfunction!(py_adc_to_spike_windows, m)?)?;
    m.add_function(wrap_pyfunction!(py_sc_forward_packed, m)?)?;
    wilson_cowan_binding::register(m)?;
    m.add_class::<Lfsr16>()?;
    m.add_class::<BitstreamEncoder>()?;
    m.add_class::<FixedPointLif>()?;
    m.add_class::<DenseLayer>()?;
    #[cfg(feature = "gpu")]
    m.add_class::<gpu::PyGpuDenseLayer>()?;
    #[cfg(feature = "gpu")]
    m.add_class::<gpu::PyGpuLifBatch>()?;
    #[cfg(feature = "gpu")]
    m.add_class::<gpu::PyGpuKuramoto>()?;
    #[cfg(feature = "gpu")]
    m.add_class::<gpu::PyGpuIzhikevichBatch>()?;
    m.add_class::<StdpSynapse>()?;
    learning_bindings::register(m)?;
    m.add_class::<PyKuramotoSolver>()?;
    m.add_class::<PySCPNMetrics>()?;
    hdc_binding::register(m)?;
    m.add_class::<PyBrunelNetwork>()?;
    m.add_class::<PyIzhikevich>()?;
    m.add_class::<PyBitstreamAverager>()?;
    ir::bindings::register(m)?;
    m.add_class::<PyAdExNeuron>()?;
    m.add_class::<PyExpIFNeuron>()?;
    m.add_class::<PyLapicqueNeuron>()?;
    pyo3_neurons::register_neuron_classes(m)?;
    m.add_class::<PyNetworkRunner>()?;
    #[cfg(feature = "z3")]
    m.add_class::<supervisor::PySpikingControllerPool>()?;
    m.add_function(wrap_pyfunction!(py_simulate_ei_network, m)?)?;
    m.add_function(wrap_pyfunction!(py_batch_simulate, m)?)?;
    m.add_function(wrap_pyfunction!(rk4_neurons::py_rk4_neuron_simulate, m)?)?;
    cordiv_binding::register(m)?;
    predictive_coding_binding::register(m)?;
    phi_binding::register(m)?;
    m.add_class::<PyCorticalColumn>()?;
    m.add_class::<PyRallDendrite>()?;
    analysis::bindings::register(m)?;
    dna::bindings::register(m)?;
    quantum::bindings::register(m)?;
    // Photonic NoC acceleration
    photonic::bindings::register(m)?;
    optimizer_binding::register(m)?;
    evolution_binding::register(m)?;
    // LGSSM Kalman filter (predictive_model)
    m.add_function(wrap_pyfunction!(py_lgssm_kalman_filter, m)?)?;
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
    m.add_function(wrap_pyfunction!(py_izhikevich2007_simulate, m)?)?;
    fault_binding::register(m)?;
    // Hierarchical partitioner KL refine
    m.add_function(wrap_pyfunction!(py_kl_refine, m)?)?;
    // PINGCircuit per-step kernel
    m.add_function(wrap_pyfunction!(py_ping_step, m)?)?;
    // CorticalColumn block-CSR per-row-parallel spmv
    m.add_function(wrap_pyfunction!(py_parallel_csr_spmv_add, m)?)?;
    m.add_function(wrap_pyfunction!(py_parallel_csr_multi_spmv_add, m)?)?;
    Ok(())
}

// ── Cortical column ──────────────────────────────────────────────────

#[pyclass(
    name = "CorticalColumnRust",
    module = "sc_neurocore_engine.sc_neurocore_engine"
)]
pub struct PyCorticalColumn {
    inner: cortical_column::CorticalColumnRust,
}

#[pymethods]
impl PyCorticalColumn {
    #[new]
    fn new(n: usize, tau: f64, dt: f64, threshold: f64, w_exc: f64, w_inh: f64, seed: u64) -> Self {
        Self {
            inner: cortical_column::CorticalColumnRust::new(
                n, tau, dt, threshold, w_exc, w_inh, seed,
            ),
        }
    }

    fn step<'py>(
        &mut self,
        py: Python<'py>,
        thalamic_input: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Py<PyDict>> {
        let input = thalamic_input.as_slice()?;
        let spikes = self.inner.step(input);
        let dict = PyDict::new(py);
        let names = ["l4", "l23_exc", "l23_inh", "l5", "l6"];
        for (i, name) in names.iter().enumerate() {
            dict.set_item(*name, spikes[i].clone().into_pyarray(py))?;
        }
        Ok(dict.into())
    }

    fn reset(&mut self) {
        self.inner.reset();
    }
}

// ── Rall dendrite ────────────────────────────────────────────────────

#[pyclass(
    name = "RallDendriteRust",
    module = "sc_neurocore_engine.sc_neurocore_engine"
)]
pub struct PyRallDendrite {
    inner: rall_dendrite::RallDendriteRust,
}

#[pymethods]
impl PyRallDendrite {
    #[new]
    fn new(n_branches: usize, branch_length: usize, tau: f64, coupling: f64, dt: f64) -> Self {
        Self {
            inner: rall_dendrite::RallDendriteRust::new(
                n_branches,
                branch_length,
                tau,
                coupling,
                dt,
            ),
        }
    }

    fn step(&mut self, branch_inputs: PyReadonlyArray1<'_, f64>) -> PyResult<f64> {
        let inputs = branch_inputs.as_slice()?;
        Ok(self.inner.step(inputs))
    }

    fn reset(&mut self) {
        self.inner.reset();
    }

    #[getter]
    fn soma_v(&self) -> f64 {
        self.inner.soma_v
    }
}

/// Returns the highest SIMD tier available on this CPU.
#[pyfunction]
fn simd_tier() -> &'static str {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512vpopcntdq") {
            return "avx512-vpopcntdq";
        }
        if is_x86_feature_detected!("avx512bw") {
            return "avx512bw";
        }
        if is_x86_feature_detected!("avx512f") {
            return "avx512f";
        }
        if is_x86_feature_detected!("avx2") {
            return "avx2";
        }
        if is_x86_feature_detected!("popcnt") {
            return "popcnt";
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        return "neon";
    }
    "portable"
}

/// Set the number of threads in the global rayon thread pool.
///
/// Must be called before any parallel operation.
/// Passing 0 uses rayon's default (number of CPU cores).
#[pyfunction]
fn set_num_threads(n: usize) -> PyResult<()> {
    if n == 0 {
        return Ok(());
    }
    rayon::ThreadPoolBuilder::new()
        .num_threads(n)
        .build_global()
        .map_err(|e| PyValueError::new_err(format!("Cannot set thread pool: {e}")))
}

#[pyfunction]
fn pack_bitstream(py: Python<'_>, bits: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    if let Ok(rows) = bits.extract::<Vec<Vec<u8>>>() {
        let packed_rows: Vec<Vec<u64>> = rows.iter().map(|row| bitstream::pack(row).data).collect();
        return Ok(packed_rows
            .into_pyobject(py)
            .map_err(|e| PyValueError::new_err(e.to_string()))?
            .into_any()
            .unbind());
    }

    let flat = bits
        .extract::<Vec<u8>>()
        .map_err(|_| PyValueError::new_err("Expected a 1-D or 2-D array of uint8 bits."))?;
    Ok(bitstream::pack(&flat)
        .data
        .into_pyobject(py)
        .map_err(|e| PyValueError::new_err(e.to_string()))?
        .into_any()
        .unbind())
}

#[pyfunction]
#[pyo3(signature = (packed, original_length, original_shape=None))]
fn unpack_bitstream(
    py: Python<'_>,
    packed: &Bound<'_, PyAny>,
    original_length: usize,
    original_shape: Option<(usize, usize)>,
) -> PyResult<Py<PyAny>> {
    if let Ok(rows) = packed.extract::<Vec<Vec<u64>>>() {
        let batch = rows.len();
        let per_batch_len = if let Some((expected_batch, length)) = original_shape {
            if expected_batch != batch {
                return Err(PyValueError::new_err(format!(
                    "original_shape batch {} does not match packed batch {}.",
                    expected_batch, batch
                )));
            }
            length
        } else {
            original_length.checked_div(batch).unwrap_or(0)
        };

        let unpacked_rows: Vec<Vec<u8>> = rows
            .into_iter()
            .map(|row| {
                bitstream::unpack(&bitstream::BitStreamTensor::from_words(row, per_batch_len))
            })
            .collect();
        return Ok(unpacked_rows
            .into_pyobject(py)
            .map_err(|e| PyValueError::new_err(e.to_string()))?
            .into_any()
            .unbind());
    }

    let words = packed.extract::<Vec<u64>>().map_err(|_| {
        PyValueError::new_err("Expected packed uint64 words as 1-D or 2-D sequence.")
    })?;
    let tensor = bitstream::BitStreamTensor::from_words(words, original_length);
    Ok(bitstream::unpack(&tensor)
        .into_pyobject(py)
        .map_err(|e| PyValueError::new_err(e.to_string()))?
        .into_any()
        .unbind())
}

/// Pack a 1-D numpy uint8 array into packed u64 words, returning a numpy array.
/// Zero-copy input, single-allocation output.
#[pyfunction]
fn pack_bitstream_numpy<'py>(
    py: Python<'py>,
    bits: PyReadonlyArray1<'py, u8>,
) -> PyResult<Bound<'py, PyArray1<u64>>> {
    let slice = bits
        .as_slice()
        .map_err(|e| PyValueError::new_err(format!("Cannot read numpy array: {e}")))?;
    let tensor = simd::pack_dispatch(slice);
    Ok(tensor.data.into_pyarray(py))
}

/// Popcount on a numpy uint64 array — zero-copy input.
#[pyfunction]
fn popcount_numpy(packed: PyReadonlyArray1<'_, u64>) -> PyResult<u64> {
    let words = packed
        .as_slice()
        .map_err(|e| PyValueError::new_err(format!("Cannot read numpy array: {e}")))?;
    Ok(simd::popcount_dispatch(words))
}

/// Unpack a numpy uint64 array back to a numpy uint8 array.
#[pyfunction]
fn unpack_bitstream_numpy<'py>(
    py: Python<'py>,
    packed: PyReadonlyArray1<'py, u64>,
    original_length: usize,
) -> PyResult<Bound<'py, PyArray1<u8>>> {
    let words = packed
        .as_slice()
        .map_err(|e| PyValueError::new_err(format!("Cannot read numpy array: {e}")))?;
    let tensor = bitstream::BitStreamTensor::from_words(words.to_vec(), original_length);
    let bits = bitstream::unpack(&tensor);
    Ok(bits.into_pyarray(py))
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

/// Bernoulli-encode a numpy float64 array into packed bitstream words.
///
/// Returns nested packed words with shape (n_probs, ceil(length / 64)).
#[pyfunction]
#[pyo3(signature = (probs, length=1024, seed=0xACE1))]
fn batch_encode<'py>(
    _py: Python<'py>,
    probs: PyReadonlyArray1<'py, f64>,
    length: usize,
    seed: u64,
) -> PyResult<Vec<Vec<u64>>> {
    let prob_slice = probs
        .as_slice()
        .map_err(|e| PyValueError::new_err(format!("Cannot read probs: {e}")))?;
    let words = length.div_ceil(64);

    use rand::SeedableRng;
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);

    let packed: Vec<Vec<u64>> = prob_slice
        .iter()
        .map(|&p| {
            let mut data = bitstream::bernoulli_packed(p, length, &mut rng);
            data.resize(words, 0);
            data
        })
        .collect();

    Ok(packed)
}

/// Bernoulli-encode a numpy float64 array into a 2-D numpy uint64 array.
///
/// Returns shape `(n_probs, ceil(length / 64))`.
#[pyfunction]
#[pyo3(signature = (probs, length=1024, seed=0xACE1))]
fn batch_encode_numpy<'py>(
    py: Python<'py>,
    probs: PyReadonlyArray1<'py, f64>,
    length: usize,
    seed: u64,
) -> PyResult<Bound<'py, PyArray2<u64>>> {
    use rayon::prelude::*;

    let prob_slice = probs
        .as_slice()
        .map_err(|e| PyValueError::new_err(format!("Cannot read probs: {e}")))?;
    let words = length.div_ceil(64);
    let n_probs = prob_slice.len();

    let rows: Vec<Vec<u64>> = prob_slice
        .par_iter()
        .enumerate()
        .map(|(idx, &p)| {
            use rand::SeedableRng;

            let prob_seed = seed.wrapping_add(idx as u64);
            let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(prob_seed);
            let mut row = bitstream::bernoulli_packed_simd(p, length, &mut rng);
            row.resize(words, 0);
            row
        })
        .collect();

    let mut flat = Vec::with_capacity(n_probs * words);
    for row in &rows {
        flat.extend_from_slice(row);
    }

    let arr = ndarray::Array2::from_shape_vec((n_probs, words), flat)
        .map_err(|e| PyValueError::new_err(format!("Shape construction failed: {e}")))?;
    Ok(arr.into_pyarray(py))
}

#[pyclass(module = "sc_neurocore_engine.sc_neurocore_engine")]
pub struct Lfsr16 {
    inner: encoder::Lfsr16,
    seed_init: u16,
}

#[pymethods]
impl Lfsr16 {
    #[new]
    #[pyo3(signature = (seed=0xACE1))]
    fn new(seed: u16) -> PyResult<Self> {
        if seed == 0 {
            return Err(PyValueError::new_err("LFSR seed must be non-zero."));
        }
        Ok(Self {
            inner: encoder::Lfsr16::new(seed),
            seed_init: seed,
        })
    }

    fn step(&mut self) -> u16 {
        self.inner.step()
    }

    #[getter]
    fn reg(&self) -> u16 {
        self.inner.reg
    }

    #[getter]
    fn width(&self) -> u32 {
        self.inner.width
    }

    #[pyo3(signature = (seed=None))]
    fn reset(&mut self, seed: Option<u16>) -> PyResult<()> {
        let next = seed.unwrap_or(self.seed_init);
        if next == 0 {
            return Err(PyValueError::new_err("LFSR seed must be non-zero."));
        }
        self.inner = encoder::Lfsr16::new(next);
        self.seed_init = next;
        Ok(())
    }
}

#[pyclass(module = "sc_neurocore_engine.sc_neurocore_engine")]
pub struct BitstreamEncoder {
    inner: encoder::BitstreamEncoder,
    seed_init: u16,
}

#[pymethods]
impl BitstreamEncoder {
    #[new]
    #[pyo3(signature = (data_width=16, seed=0xACE1))]
    fn new(data_width: u32, seed: u16) -> PyResult<Self> {
        if seed == 0 {
            return Err(PyValueError::new_err("LFSR seed must be non-zero."));
        }
        Ok(Self {
            inner: encoder::BitstreamEncoder::new(data_width, seed),
            seed_init: seed,
        })
    }

    fn step(&mut self, x_value: u16) -> u8 {
        self.inner.step(x_value)
    }

    #[getter]
    fn data_width(&self) -> u32 {
        self.inner.data_width
    }

    #[getter]
    fn reg(&self) -> u16 {
        self.inner.lfsr.reg
    }

    #[pyo3(signature = (seed=None))]
    fn reset(&mut self, seed: Option<u16>) -> PyResult<()> {
        let next = seed.unwrap_or(self.seed_init);
        if next == 0 {
            return Err(PyValueError::new_err("LFSR seed must be non-zero."));
        }
        self.inner.reset(Some(next));
        self.seed_init = next;
        Ok(())
    }
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

// ── Izhikevich PyO3 wrapper ─────────────────────────────────────

#[pyclass(
    name = "Izhikevich",
    module = "sc_neurocore_engine.sc_neurocore_engine"
)]
pub struct PyIzhikevich {
    inner: neuron::Izhikevich,
}

#[pymethods]
impl PyIzhikevich {
    #[new]
    #[pyo3(signature = (a=0.02, b=0.2, c=-65.0, d=8.0, dt=1.0))]
    fn new(a: f64, b: f64, c: f64, d: f64, dt: f64) -> Self {
        Self {
            inner: neuron::Izhikevich::new(a, b, c, d, dt),
        }
    }

    fn step(&mut self, current: f64) -> i32 {
        self.inner.step(current)
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
        dict.set_item("u", self.inner.u)?;
        Ok(dict.into_any().unbind())
    }
}

// ── BitstreamAverager PyO3 wrapper ──────────────────────────────

#[pyclass(
    name = "BitstreamAverager",
    module = "sc_neurocore_engine.sc_neurocore_engine"
)]
pub struct PyBitstreamAverager {
    inner: neuron::BitstreamAverager,
}

#[pymethods]
impl PyBitstreamAverager {
    #[new]
    #[pyo3(signature = (window=1024))]
    fn new(window: usize) -> Self {
        Self {
            inner: neuron::BitstreamAverager::new(window),
        }
    }

    fn push(&mut self, bit: u8) {
        self.inner.push(bit);
    }

    fn estimate(&self) -> f64 {
        self.inner.estimate()
    }

    fn reset(&mut self) {
        self.inner.reset();
    }

    #[getter]
    fn window(&self) -> usize {
        self.inner.window()
    }
}

// ── AdEx, ExpIF, Lapicque PyO3 wrappers ────────────────────────

#[pyclass(
    name = "AdExNeuron",
    module = "sc_neurocore_engine.sc_neurocore_engine"
)]
#[derive(Clone)]
pub struct PyAdExNeuron {
    inner: neuron::AdExNeuron,
}

#[pymethods]
impl PyAdExNeuron {
    #[new]
    fn new() -> Self {
        Self {
            inner: neuron::AdExNeuron::new(),
        }
    }
    fn step(&mut self, current: f64) -> i32 {
        self.inner.step(current)
    }
    fn reset(&mut self) {
        self.inner.reset();
    }
    fn get_state(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let d = PyDict::new(py);
        d.set_item("v", self.inner.v)?;
        d.set_item("w", self.inner.w)?;
        Ok(d.into_any().unbind())
    }
}

#[pyclass(
    name = "ExpIFNeuron",
    module = "sc_neurocore_engine.sc_neurocore_engine"
)]
#[derive(Clone)]
pub struct PyExpIFNeuron {
    inner: neuron::ExpIfNeuron,
}

#[pymethods]
impl PyExpIFNeuron {
    #[new]
    fn new() -> Self {
        Self {
            inner: neuron::ExpIfNeuron::new(),
        }
    }
    fn step(&mut self, current: f64) -> i32 {
        self.inner.step(current)
    }
    fn reset(&mut self) {
        self.inner.reset();
    }
    fn get_state(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let d = PyDict::new(py);
        d.set_item("v", self.inner.v)?;
        d.set_item("refractory_remaining", self.inner.refractory_remaining)?;
        Ok(d.into_any().unbind())
    }
}

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
    fn new(tau: f64, resistance: f64, threshold: f64, dt: f64) -> Self {
        Self {
            inner: neuron::LapicqueNeuron::new(tau, resistance, threshold, dt),
        }
    }
    fn step(&mut self, current: f64) -> i32 {
        self.inner.step(current)
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

// ── LGSSM Kalman filter (predictive_model) ──────────────────────────

/// Forward Kalman filter for a Linear Gaussian State-Space Model.
///
/// Parity contract with `sc_neurocore.world_model.predictive_model.KalmanFilter`:
/// for the same model parameters and observation sequence, the
/// returned (means, covariances, log_likelihood) must agree with
/// the Python implementation to within float64 round-off.
///
/// All matrices are passed as flat row-major Vec<f64>; the caller
/// supplies their shapes explicitly. Returns a dict with keys:
///   - "means": Vec<Vec<f64>> shape (T, d)
///   - "covariances": Vec<Vec<Vec<f64>>> shape (T, d, d)
///   - "pred_means": Vec<Vec<f64>> shape (T, d)
///   - "pred_covariances": Vec<Vec<Vec<f64>>> shape (T, d, d)
///   - "log_likelihood": f64
///   - "backend": "rust"
#[pyfunction]
#[pyo3(signature = (
    obs_flat, controls_flat, t_len, p_dim, m_dim,
    a_flat, b_flat, c_flat, d_flat, q_flat, r_flat,
    mu_0, sigma_0_flat, d_dim,
))]
#[allow(clippy::too_many_arguments)]
fn py_lgssm_kalman_filter<'py>(
    py: Python<'py>,
    obs_flat: Vec<f64>,
    controls_flat: Vec<f64>,
    t_len: usize,
    p_dim: usize,
    m_dim: usize,
    a_flat: Vec<f64>,
    b_flat: Vec<f64>,
    c_flat: Vec<f64>,
    d_flat: Vec<f64>,
    q_flat: Vec<f64>,
    r_flat: Vec<f64>,
    mu_0: Vec<f64>,
    sigma_0_flat: Vec<f64>,
    d_dim: usize,
) -> PyResult<Py<PyAny>> {
    use ndarray::Array1;
    use ndarray::Array2;

    let to_2d = |flat: &[f64], rows: usize, cols: usize| -> Array2<f64> {
        Array2::from_shape_vec((rows, cols), flat.to_vec()).expect("shape")
    };
    let obs = to_2d(&obs_flat, t_len, p_dim);
    let controls = to_2d(&controls_flat, t_len, m_dim);
    let a = to_2d(&a_flat, d_dim, d_dim);
    let b = to_2d(&b_flat, d_dim, m_dim);
    let c = to_2d(&c_flat, p_dim, d_dim);
    let d = to_2d(&d_flat, p_dim, m_dim);
    let q = to_2d(&q_flat, d_dim, d_dim);
    let r = to_2d(&r_flat, p_dim, p_dim);
    let mu_0_arr = Array1::from(mu_0);
    let sigma_0 = to_2d(&sigma_0_flat, d_dim, d_dim);

    let result = lgssm::kalman_filter(
        obs.view(),
        controls.view(),
        a.view(),
        b.view(),
        c.view(),
        d.view(),
        q.view(),
        r.view(),
        mu_0_arr.view(),
        sigma_0.view(),
    );

    // Convert to Python-friendly nested Vec
    let means: Vec<Vec<f64>> = (0..t_len)
        .map(|t| (0..d_dim).map(|i| result.means[(t, i)]).collect())
        .collect();
    let covs: Vec<Vec<Vec<f64>>> = (0..t_len)
        .map(|t| {
            (0..d_dim)
                .map(|i| (0..d_dim).map(|j| result.covariances[(t, i, j)]).collect())
                .collect()
        })
        .collect();
    let pred_means: Vec<Vec<f64>> = (0..t_len)
        .map(|t| (0..d_dim).map(|i| result.pred_means[(t, i)]).collect())
        .collect();
    let pred_covs: Vec<Vec<Vec<f64>>> = (0..t_len)
        .map(|t| {
            (0..d_dim)
                .map(|i| {
                    (0..d_dim)
                        .map(|j| result.pred_covariances[(t, i, j)])
                        .collect()
                })
                .collect()
        })
        .collect();

    let dict = PyDict::new(py);
    dict.set_item("means", means)?;
    dict.set_item("covariances", covs)?;
    dict.set_item("pred_means", pred_means)?;
    dict.set_item("pred_covariances", pred_covs)?;
    dict.set_item("log_likelihood", result.log_likelihood)?;
    dict.set_item("backend", "rust")?;
    Ok(dict.into_any().unbind())
}

/// Parity contract with
/// `sc_neurocore.neurons.models.izhikevich2007.Izhikevich2007Neuron.simulate`:
/// for the same parameters and constant input the returned `v` trace, spike
/// count, and final `(v, u)` state are bit-identical to the Python RK4 reference
/// (the NeuroML right-hand side `k (v-vr)(v-vt)/C` is exact arithmetic — products,
/// a sum and a division, no transcendental functions).
#[pyfunction]
#[pyo3(signature = (v0, u0, cap, k, vr, vt, vpeak, a, b, c, d, dt, n_steps, current))]
#[allow(clippy::too_many_arguments)]
fn py_izhikevich2007_simulate<'py>(
    py: Python<'py>,
    v0: f64,
    u0: f64,
    cap: f64,
    k: f64,
    vr: f64,
    vt: f64,
    vpeak: f64,
    a: f64,
    b: f64,
    c: f64,
    d: f64,
    dt: f64,
    n_steps: usize,
    current: f64,
) -> (Bound<'py, PyArray1<f64>>, i64, f64, f64) {
    let mut neuron = crate::rk4_neurons::Izhikevich2007Rk4 {
        v: v0,
        u: u0,
        cap,
        k,
        vr,
        vt,
        vpeak,
        a,
        b,
        c,
        d,
        dt,
    };
    let (trace, spikes) = neuron.simulate(n_steps, current);
    (trace.into_pyarray(py), spikes, neuron.v, neuron.u)
}

// ── Hierarchical partitioner — KL refine (PyO3) ──
//
// Caller passes flat numpy arrays (CSR adjacency + flat scc weights +
// flat vertex_weights + initial part_map). The kernel mutates a copy
// of part_map in-place and returns (new_part_map, num_moves).

#[pyfunction]
#[pyo3(signature = (
    adj_offsets, adj_neighbours, adj_scc_abs, vertex_weights,
    part_map, parts_concat, parts_offsets,
    n_parts, kl_iterations, correlation_penalty,
))]
#[allow(clippy::too_many_arguments)]
fn py_kl_refine<'py>(
    py: Python<'py>,
    adj_offsets: PyReadonlyArray1<'_, i64>,
    adj_neighbours: PyReadonlyArray1<'_, i32>,
    adj_scc_abs: PyReadonlyArray1<'_, f64>,
    vertex_weights: PyReadonlyArray1<'_, f64>,
    part_map: PyReadonlyArray1<'_, i32>,
    parts_concat: PyReadonlyArray1<'_, i32>,
    parts_offsets: PyReadonlyArray1<'_, i64>,
    n_parts: i32,
    kl_iterations: i32,
    correlation_penalty: f64,
) -> PyResult<(Py<PyArray1<i32>>, u64)> {
    let mut pm = part_map.as_slice()?.to_vec();
    let moves = partition::kl_refine(
        adj_offsets.as_slice()?,
        adj_neighbours.as_slice()?,
        adj_scc_abs.as_slice()?,
        vertex_weights.as_slice()?,
        &mut pm,
        parts_concat.as_slice()?,
        parts_offsets.as_slice()?,
        n_parts,
        kl_iterations,
        correlation_penalty,
    );
    Ok((pm.into_pyarray(py).into(), moves))
}

// ── PINGCircuit per-step kernel ─────────────────────────────────────
//
// Mirrors `PINGCircuit.step()` in
// `src/sc_neurocore/network/gamma_oscillation.py`. Caller hands over
// the per-instance state arrays + this-step `xi` noise samples (drawn
// on the Python side from the per-instance RNG so seed determinism is
// preserved), and the kernel writes the new state in-place plus a
// boolean spike vector per population. Returns `(n_e_spikes,
// n_i_spikes)` so the caller can do the synaptic-conductance update
// step (which needs the cross-population summary, not the per-cell
// detail).

#[pyfunction]
#[pyo3(signature = (
    v_e, g_ampa_e, g_gaba_e, refrac_e, i_drive_e, xi_e, spikes_e_out,
    v_i, g_ampa_i, g_gaba_i, refrac_i, i_drive_i, xi_i, spikes_i_out,
    e_l, e_ampa, e_gaba, g_l, c_m, v_threshold, v_reset, t_refrac,
    tau_ampa, tau_gaba, sigma_e, sigma_i, dt,
))]
#[allow(clippy::too_many_arguments)]
fn py_ping_step<'py>(
    _py: Python<'py>,
    v_e: PyReadwriteArray1<'_, f64>,
    g_ampa_e: PyReadwriteArray1<'_, f64>,
    g_gaba_e: PyReadwriteArray1<'_, f64>,
    refrac_e: PyReadwriteArray1<'_, f64>,
    i_drive_e: PyReadonlyArray1<'_, f64>,
    xi_e: PyReadonlyArray1<'_, f64>,
    spikes_e_out: PyReadwriteArray1<'_, u8>,
    v_i: PyReadwriteArray1<'_, f64>,
    g_ampa_i: PyReadwriteArray1<'_, f64>,
    g_gaba_i: PyReadwriteArray1<'_, f64>,
    refrac_i: PyReadwriteArray1<'_, f64>,
    i_drive_i: PyReadonlyArray1<'_, f64>,
    xi_i: PyReadonlyArray1<'_, f64>,
    spikes_i_out: PyReadwriteArray1<'_, u8>,
    e_l: f64,
    e_ampa: f64,
    e_gaba: f64,
    g_l: f64,
    c_m: f64,
    v_threshold: f64,
    v_reset: f64,
    t_refrac: f64,
    tau_ampa: f64,
    tau_gaba: f64,
    sigma_e: f64,
    sigma_i: f64,
    dt: f64,
) -> PyResult<(u32, u32)> {
    let mut v_e = v_e;
    let mut g_ampa_e = g_ampa_e;
    let mut g_gaba_e = g_gaba_e;
    let mut refrac_e = refrac_e;
    let mut spikes_e_out = spikes_e_out;
    let mut v_i = v_i;
    let mut g_ampa_i = g_ampa_i;
    let mut g_gaba_i = g_gaba_i;
    let mut refrac_i = refrac_i;
    let mut spikes_i_out = spikes_i_out;
    let (ne, ni) = ping::step_kernel(
        v_e.as_slice_mut()?,
        g_ampa_e.as_slice_mut()?,
        g_gaba_e.as_slice_mut()?,
        refrac_e.as_slice_mut()?,
        i_drive_e.as_slice()?,
        xi_e.as_slice()?,
        spikes_e_out.as_slice_mut()?,
        v_i.as_slice_mut()?,
        g_ampa_i.as_slice_mut()?,
        g_gaba_i.as_slice_mut()?,
        refrac_i.as_slice_mut()?,
        i_drive_i.as_slice()?,
        xi_i.as_slice()?,
        spikes_i_out.as_slice_mut()?,
        e_l,
        e_ampa,
        e_gaba,
        g_l,
        c_m,
        v_threshold,
        v_reset,
        t_refrac,
        tau_ampa,
        tau_gaba,
        sigma_e,
        sigma_i,
        dt,
    );
    Ok((ne, ni))
}

// ── CorticalColumn block-CSR spmv (per-row-parallel) ────────────────
//
// `y += W @ x` where `W` is a CSR matrix described by
// `(indptr, indices, data)`. Rows are processed in parallel via
// rayon — bit-identical to scipy single-threaded for matching
// inputs because the per-row reduction is local. Used by
// `CorticalColumn._inject_block(dt)` once per `(source-type, bin)`
// pair, replacing scipy's single-threaded csr_matvec for that step.

#[pyfunction]
#[pyo3(signature = (indptr, indices, data, x, y))]
fn py_parallel_csr_spmv_add(
    indptr: PyReadonlyArray1<'_, i32>,
    indices: PyReadonlyArray1<'_, i32>,
    data: PyReadonlyArray1<'_, f64>,
    x: PyReadonlyArray1<'_, f64>,
    y: PyReadwriteArray1<'_, f64>,
) -> PyResult<()> {
    let mut y = y;
    cortical_inject::parallel_csr_spmv_add(
        indptr.as_slice()?,
        indices.as_slice()?,
        data.as_slice()?,
        x.as_slice()?,
        y.as_slice_mut()?,
    );
    Ok(())
}

// Batched multi-spmv. The Python side passes lists of numpy arrays
// (one per block) for indptrs / indices / data / xs and a single
// mutable output. ONE FFI call per step replaces N (= n_delay_bins
// for E + same for I, typically 10) per-block calls. At scale=0.1
// the per-call savings amortise over a 600 ms simulation.
#[pyfunction]
#[pyo3(signature = (indptrs, indices_list, data_list, xs, y))]
fn py_parallel_csr_multi_spmv_add(
    indptrs: Vec<PyReadonlyArray1<'_, i32>>,
    indices_list: Vec<PyReadonlyArray1<'_, i32>>,
    data_list: Vec<PyReadonlyArray1<'_, f64>>,
    xs: Vec<PyReadonlyArray1<'_, f64>>,
    y: PyReadwriteArray1<'_, f64>,
) -> PyResult<()> {
    let mut y = y;
    let indptr_slices: Vec<&[i32]> = indptrs
        .iter()
        .map(|a| a.as_slice())
        .collect::<Result<_, _>>()?;
    let indices_slices: Vec<&[i32]> = indices_list
        .iter()
        .map(|a| a.as_slice())
        .collect::<Result<_, _>>()?;
    let data_slices: Vec<&[f64]> = data_list
        .iter()
        .map(|a| a.as_slice())
        .collect::<Result<_, _>>()?;
    let x_slices: Vec<&[f64]> = xs.iter().map(|a| a.as_slice()).collect::<Result<_, _>>()?;
    cortical_inject::parallel_csr_multi_spmv_add(
        &indptr_slices,
        &indices_slices,
        &data_slices,
        &x_slices,
        y.as_slice_mut()?,
    );
    Ok(())
}
