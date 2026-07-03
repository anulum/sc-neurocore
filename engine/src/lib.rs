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
pub mod brunel;
pub mod connectome;
pub mod conv;
pub mod cordiv;
pub mod cortical_column;
pub mod cortical_inject;
pub mod dna;
pub mod ei_network;
pub mod encoder;
pub mod evo;
pub mod fault;
pub mod fusion;
#[cfg(feature = "gpu")]
pub mod gpu;
pub mod grad;
pub mod graph;
pub mod ir;
pub mod layer;
pub mod lgssm;
pub mod network_runner;
pub mod neuron;
pub mod neurons;
pub mod optimizer;
pub mod partition;
pub mod phi;
pub mod photonic;
pub mod ping;
pub mod predictive_coding;
pub mod pyo3_neurons;
pub mod quantum;
pub mod rall_dendrite;
pub mod recorder;
pub mod recurrent;
pub mod rk4_neurons;
pub mod sc_inference;
pub mod scpn;
pub mod simd;
pub mod sobol;
#[cfg(feature = "z3")]
pub mod supervisor;
pub mod synapses;
pub mod topology;
pub mod wilson_cowan;
pub mod wong_wang;

// ── HDC / VSA PyO3 wrapper ───────────────────────────────────────────

#[pyclass(
    name = "BitStreamTensor",
    module = "sc_neurocore_engine.sc_neurocore_engine"
)]
pub struct PyBitStreamTensor {
    inner: bitstream::BitStreamTensor,
}

#[pymethods]
impl PyBitStreamTensor {
    /// Create a random binary vector of `dimension` bits.
    #[new]
    #[pyo3(signature = (dimension=10000, seed=0xACE1))]
    fn new(dimension: usize, seed: u64) -> Self {
        use rand::SeedableRng;
        let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(seed);
        let data = bitstream::bernoulli_packed(0.5, dimension, &mut rng);
        Self {
            inner: bitstream::BitStreamTensor::from_words(data, dimension),
        }
    }

    /// Create from pre-packed u64 words.
    #[staticmethod]
    fn from_packed(data: Vec<u64>, length: usize) -> PyResult<Self> {
        if length == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "bitstream length must be > 0",
            ));
        }
        Ok(Self {
            inner: bitstream::BitStreamTensor::from_words(data, length),
        })
    }

    /// In-place XOR (HDC bind).
    fn xor_inplace(&mut self, other: &PyBitStreamTensor) {
        self.inner.xor_inplace(&other.inner);
    }

    /// XOR returning a new tensor (HDC bind).
    fn xor(&self, other: &PyBitStreamTensor) -> PyBitStreamTensor {
        PyBitStreamTensor {
            inner: self.inner.xor(&other.inner),
        }
    }

    /// Cyclic right rotation by `shift` bits (HDC permute).
    fn rotate_right(&mut self, shift: usize) {
        self.inner.rotate_right(shift);
    }

    /// Normalized Hamming distance (0.0 = identical, 1.0 = opposite).
    fn hamming_distance(&self, other: &PyBitStreamTensor) -> f32 {
        self.inner.hamming_distance(&other.inner)
    }

    /// Majority-vote bundle of multiple tensors.
    #[staticmethod]
    fn bundle(vectors: Vec<PyRef<'_, PyBitStreamTensor>>) -> PyBitStreamTensor {
        let refs: Vec<&bitstream::BitStreamTensor> = vectors.iter().map(|v| &v.inner).collect();
        PyBitStreamTensor {
            inner: bitstream::BitStreamTensor::bundle(&refs),
        }
    }

    /// Count of set bits.
    fn popcount(&self) -> u64 {
        bitstream::popcount(&self.inner)
    }

    /// Packed u64 words (read-only copy).
    #[getter]
    fn data(&self) -> Vec<u64> {
        self.inner.data.clone()
    }

    /// Logical bit length.
    #[getter]
    fn length(&self) -> usize {
        self.inner.length
    }

    fn __len__(&self) -> usize {
        self.inner.length
    }

    fn __repr__(&self) -> String {
        format!(
            "BitStreamTensor(length={}, popcount={})",
            self.inner.length,
            bitstream::popcount(&self.inner)
        )
    }
}

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

// ── Wong-Wang 2006 decision unit — batch PyO3 wrapper ─────────────────

/// Simulate a single Wong-Wang unit for `stim1.len()` steps and return
/// the per-step state traces + firing-rate traces.
///
/// Parity contract with `sc_neurocore.neurons.models.wong_wang.WongWangUnit`:
/// the caller pre-draws `2 * n_steps` N(0, 1) samples so Python-side
/// `numpy.random` ordering is preserved bit-for-bit.
///
/// Returns a dict with keys: `s1`, `s2`, `r1`, `r2`
/// (1-D float64 arrays of length `n_steps`) and the final scalar
/// `s1_final`, `s2_final`.
#[pyfunction]
#[pyo3(signature = (
    s1_init, s2_init,
    tau_s, gamma, j_n, j_cross, i_0, sigma, dt,
    stim1, stim2, xi,
))]
#[allow(clippy::too_many_arguments)]
fn py_wong_wang_simulate<'py>(
    py: Python<'py>,
    s1_init: f64,
    s2_init: f64,
    tau_s: f64,
    gamma: f64,
    j_n: f64,
    j_cross: f64,
    i_0: f64,
    sigma: f64,
    dt: f64,
    stim1: PyReadonlyArray1<'py, f64>,
    stim2: PyReadonlyArray1<'py, f64>,
    xi: PyReadonlyArray1<'py, f64>,
) -> PyResult<Py<PyAny>> {
    let stim1 = stim1.as_slice()?;
    let stim2 = stim2.as_slice()?;
    let xi = xi.as_slice()?;
    let n = stim1.len();
    if stim2.len() != n {
        return Err(PyValueError::new_err(format!(
            "stim1 and stim2 length mismatch: {n} vs {}",
            stim2.len()
        )));
    }
    if xi.len() != 2 * n {
        return Err(PyValueError::new_err(format!(
            "xi length must be 2 * n_steps ({}): got {}",
            2 * n,
            xi.len()
        )));
    }
    let mut s1o = vec![0.0_f64; n];
    let mut s2o = vec![0.0_f64; n];
    let mut r1o = vec![0.0_f64; n];
    let mut r2o = vec![0.0_f64; n];
    let (s1_final, s2_final) = wong_wang::simulate(
        s1_init, s2_init, tau_s, gamma, j_n, j_cross, i_0, sigma, dt, stim1, stim2, xi, &mut s1o,
        &mut s2o, &mut r1o, &mut r2o,
    );
    let d = PyDict::new(py);
    d.set_item("s1", s1o.into_pyarray(py))?;
    d.set_item("s2", s2o.into_pyarray(py))?;
    d.set_item("r1", r1o.into_pyarray(py))?;
    d.set_item("r2", r2o.into_pyarray(py))?;
    d.set_item("s1_final", s1_final)?;
    d.set_item("s2_final", s2_final)?;
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

// ── Wilson-Cowan 1972 E/I rate model — batch PyO3 wrapper ─────────────

/// Simulate a single Wilson-Cowan E/I unit for `ext_input.len()` steps
/// and return per-step `e`, `i` traces plus final scalars.
///
/// Returns a dict with keys: `e`, `i` (1-D float64 arrays of length
/// `n_steps`) + the final scalars `e_final`, `i_final`.
#[pyfunction]
#[pyo3(signature = (
    e_init, i_init,
    w_ee, w_ei, w_ie, w_ii,
    tau_e, tau_i,
    a, theta, dt,
    ext_input,
))]
#[allow(clippy::too_many_arguments)]
fn py_wilson_cowan_simulate<'py>(
    py: Python<'py>,
    e_init: f64,
    i_init: f64,
    w_ee: f64,
    w_ei: f64,
    w_ie: f64,
    w_ii: f64,
    tau_e: f64,
    tau_i: f64,
    a: f64,
    theta: f64,
    dt: f64,
    ext_input: PyReadonlyArray1<'py, f64>,
) -> PyResult<Py<PyAny>> {
    let ext = ext_input.as_slice()?;
    let n = ext.len();
    let mut e_out = vec![0.0_f64; n];
    let mut i_out = vec![0.0_f64; n];
    let (e_final, i_final) = wilson_cowan::simulate(
        e_init, i_init, w_ee, w_ei, w_ie, w_ii, tau_e, tau_i, a, theta, dt, ext, &mut e_out,
        &mut i_out,
    );
    let d = PyDict::new(py);
    d.set_item("e", e_out.into_pyarray(py))?;
    d.set_item("i", i_out.into_pyarray(py))?;
    d.set_item("e_final", e_final)?;
    d.set_item("i_final", i_final)?;
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
    m.add_function(wrap_pyfunction!(popcount, m)?)?;
    m.add_function(wrap_pyfunction!(pack_bitstream_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(popcount_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(unpack_bitstream_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(batch_lif_run, m)?)?;
    m.add_function(wrap_pyfunction!(batch_lif_run_multi, m)?)?;
    m.add_function(wrap_pyfunction!(batch_lif_run_varying, m)?)?;
    m.add_function(wrap_pyfunction!(batch_encode, m)?)?;
    m.add_function(wrap_pyfunction!(batch_encode_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(py_wong_wang_simulate, m)?)?;
    m.add_function(wrap_pyfunction!(py_dcls_max_forward_batch_q88, m)?)?;
    m.add_function(wrap_pyfunction!(py_mixed_dense_forward_batch_q88_q1616, m)?)?;
    m.add_function(wrap_pyfunction!(py_adc_to_spike_windows, m)?)?;
    m.add_function(wrap_pyfunction!(py_sc_forward_packed, m)?)?;
    m.add_function(wrap_pyfunction!(py_wilson_cowan_simulate, m)?)?;
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
    m.add_class::<PySurrogateLif>()?;
    m.add_class::<PyDifferentiableDenseLayer>()?;
    m.add_class::<PyStochasticAttention>()?;
    m.add_class::<PyStochasticGraphLayer>()?;
    m.add_class::<PyKuramotoSolver>()?;
    m.add_class::<PySCPNMetrics>()?;
    m.add_class::<PyBitStreamTensor>()?;
    m.add_class::<PyBrunelNetwork>()?;
    m.add_class::<PyIzhikevich>()?;
    m.add_class::<PyBitstreamAverager>()?;
    m.add_class::<PyScGraph>()?;
    m.add_class::<PyScGraphBuilder>()?;
    m.add_function(wrap_pyfunction!(ir_verify, m)?)?;
    m.add_function(wrap_pyfunction!(ir_print, m)?)?;
    m.add_function(wrap_pyfunction!(ir_parse, m)?)?;
    m.add_function(wrap_pyfunction!(ir_emit_sv, m)?)?;
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
    m.add_function(wrap_pyfunction!(py_cordiv, m)?)?;
    m.add_function(wrap_pyfunction!(py_adaptive_length, m)?)?;
    m.add_function(wrap_pyfunction!(py_prediction_error, m)?)?;
    m.add_function(wrap_pyfunction!(py_predict_xor_ema, m)?)?;
    m.add_function(wrap_pyfunction!(py_predict_xor_lfsr, m)?)?;
    m.add_function(wrap_pyfunction!(py_recover_xor_ema, m)?)?;
    m.add_function(wrap_pyfunction!(py_recover_xor_lfsr, m)?)?;
    m.add_function(wrap_pyfunction!(py_phi_star, m)?)?;
    m.add_class::<PyCorticalColumn>()?;
    m.add_class::<PyRallDendrite>()?;
    // Analysis functions (P0-A: spike_stats)
    m.add_function(wrap_pyfunction!(py_spike_times, m)?)?;
    m.add_function(wrap_pyfunction!(py_isi, m)?)?;
    m.add_function(wrap_pyfunction!(py_firing_rate, m)?)?;
    m.add_function(wrap_pyfunction!(py_spike_count, m)?)?;
    m.add_function(wrap_pyfunction!(py_bin_spike_train, m)?)?;
    m.add_function(wrap_pyfunction!(py_instantaneous_rate, m)?)?;
    m.add_function(wrap_pyfunction!(py_psth, m)?)?;
    m.add_function(wrap_pyfunction!(py_cv_isi, m)?)?;
    m.add_function(wrap_pyfunction!(py_cv2, m)?)?;
    m.add_function(wrap_pyfunction!(py_local_variation, m)?)?;
    m.add_function(wrap_pyfunction!(py_fano_factor, m)?)?;
    m.add_function(wrap_pyfunction!(py_lempel_ziv_complexity, m)?)?;
    m.add_function(wrap_pyfunction!(py_permutation_entropy, m)?)?;
    m.add_function(wrap_pyfunction!(py_hurst_exponent, m)?)?;
    m.add_function(wrap_pyfunction!(py_approximate_entropy, m)?)?;
    m.add_function(wrap_pyfunction!(py_sample_entropy, m)?)?;
    // correlation
    m.add_function(wrap_pyfunction!(py_cross_correlation, m)?)?;
    m.add_function(wrap_pyfunction!(py_pairwise_correlation, m)?)?;
    m.add_function(wrap_pyfunction!(py_event_synchronization, m)?)?;
    m.add_function(wrap_pyfunction!(py_spike_train_coherence, m)?)?;
    m.add_function(wrap_pyfunction!(py_spike_time_tiling_coefficient, m)?)?;
    m.add_function(wrap_pyfunction!(py_covariance_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(py_autocorrelation_time, m)?)?;
    m.add_function(wrap_pyfunction!(py_noise_correlation, m)?)?;
    m.add_function(wrap_pyfunction!(py_signal_correlation, m)?)?;
    m.add_function(wrap_pyfunction!(py_spike_count_covariance, m)?)?;
    m.add_function(wrap_pyfunction!(py_joint_psth, m)?)?;
    m.add_function(wrap_pyfunction!(py_coincidence_index, m)?)?;
    // distance
    m.add_function(wrap_pyfunction!(py_van_rossum_distance, m)?)?;
    m.add_function(wrap_pyfunction!(py_victor_purpura_distance, m)?)?;
    m.add_function(wrap_pyfunction!(py_isi_distance, m)?)?;
    m.add_function(wrap_pyfunction!(py_spike_distance, m)?)?;
    m.add_function(wrap_pyfunction!(py_spike_sync, m)?)?;
    m.add_function(wrap_pyfunction!(py_spike_sync_profile, m)?)?;
    m.add_function(wrap_pyfunction!(py_spike_profile, m)?)?;
    m.add_function(wrap_pyfunction!(py_isi_profile, m)?)?;
    m.add_function(wrap_pyfunction!(py_adaptive_spike_distance, m)?)?;
    m.add_function(wrap_pyfunction!(py_schreiber_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(py_hunter_milton_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(py_earth_movers_distance, m)?)?;
    m.add_function(wrap_pyfunction!(py_multi_neuron_victor_purpura, m)?)?;
    m.add_function(wrap_pyfunction!(py_spike_distance_matrix, m)?)?;
    // information
    m.add_function(wrap_pyfunction!(py_mutual_information, m)?)?;
    m.add_function(wrap_pyfunction!(py_transfer_entropy, m)?)?;
    m.add_function(wrap_pyfunction!(py_spike_train_entropy, m)?)?;
    m.add_function(wrap_pyfunction!(py_noise_entropy, m)?)?;
    m.add_function(wrap_pyfunction!(py_stimulus_specific_information, m)?)?;
    m.add_function(wrap_pyfunction!(py_kozachenko_leonenko_mi, m)?)?;
    // causality
    m.add_function(wrap_pyfunction!(py_pairwise_granger_causality, m)?)?;
    m.add_function(wrap_pyfunction!(py_conditional_granger_causality, m)?)?;
    m.add_function(wrap_pyfunction!(py_spectral_granger_causality, m)?)?;
    m.add_function(wrap_pyfunction!(py_partial_directed_coherence, m)?)?;
    m.add_function(wrap_pyfunction!(py_directed_transfer_function, m)?)?;
    // decoding
    m.add_function(wrap_pyfunction!(py_population_vector_decode, m)?)?;
    m.add_function(wrap_pyfunction!(py_bayesian_decode, m)?)?;
    m.add_function(wrap_pyfunction!(py_maximum_likelihood_decode, m)?)?;
    m.add_function(wrap_pyfunction!(py_linear_discriminant_decode, m)?)?;
    m.add_function(wrap_pyfunction!(py_naive_bayes_decode, m)?)?;
    // neural_decoders (P1)
    m.add_function(wrap_pyfunction!(py_tokenise_spikes, m)?)?;
    m.add_function(wrap_pyfunction!(py_sinusoidal_position_encode, m)?)?;
    m.add_function(wrap_pyfunction!(py_scaled_dot_product_attention, m)?)?;
    m.add_function(wrap_pyfunction!(py_gaussian_attention, m)?)?;
    m.add_function(wrap_pyfunction!(py_infonce_loss, m)?)?;
    // network
    m.add_function(wrap_pyfunction!(py_functional_connectivity, m)?)?;
    m.add_function(wrap_pyfunction!(py_unitary_events, m)?)?;
    m.add_function(wrap_pyfunction!(py_cell_assembly_detection, m)?)?;
    m.add_function(wrap_pyfunction!(py_synfire_chain_detection, m)?)?;
    // surrogates
    m.add_function(wrap_pyfunction!(py_surrogate_isi_shuffle, m)?)?;
    m.add_function(wrap_pyfunction!(py_surrogate_dither, m)?)?;
    m.add_function(wrap_pyfunction!(py_homogeneous_poisson, m)?)?;
    m.add_function(wrap_pyfunction!(py_gamma_process, m)?)?;
    m.add_function(wrap_pyfunction!(py_compound_poisson_process, m)?)?;
    m.add_function(wrap_pyfunction!(py_surrogate_joint_isi, m)?)?;
    m.add_function(wrap_pyfunction!(py_surrogate_bin_shuffling, m)?)?;
    m.add_function(wrap_pyfunction!(py_surrogate_spike_train_shifting, m)?)?;
    // temporal
    m.add_function(wrap_pyfunction!(py_burst_detection, m)?)?;
    m.add_function(wrap_pyfunction!(py_first_spike_latency, m)?)?;
    m.add_function(wrap_pyfunction!(py_response_onset, m)?)?;
    m.add_function(wrap_pyfunction!(py_change_point_detection, m)?)?;
    // patterns
    m.add_function(wrap_pyfunction!(py_spike_directionality, m)?)?;
    m.add_function(wrap_pyfunction!(py_spike_train_order, m)?)?;
    m.add_function(wrap_pyfunction!(py_cubic_higher_order, m)?)?;
    // spectral
    m.add_function(wrap_pyfunction!(py_power_spectrum, m)?)?;
    // waveform
    m.add_function(wrap_pyfunction!(py_waveform_width, m)?)?;
    m.add_function(wrap_pyfunction!(py_waveform_amplitude, m)?)?;
    m.add_function(wrap_pyfunction!(py_waveform_repolarization_slope, m)?)?;
    m.add_function(wrap_pyfunction!(py_waveform_recovery_slope, m)?)?;
    m.add_function(wrap_pyfunction!(py_waveform_halfwidth, m)?)?;
    m.add_function(wrap_pyfunction!(py_waveform_pt_ratio, m)?)?;
    // point_process
    m.add_function(wrap_pyfunction!(py_conditional_intensity, m)?)?;
    m.add_function(wrap_pyfunction!(py_isi_hazard_function, m)?)?;
    m.add_function(wrap_pyfunction!(py_isi_survivor_function, m)?)?;
    m.add_function(wrap_pyfunction!(py_renewal_density, m)?)?;
    // stimulus
    m.add_function(wrap_pyfunction!(py_spike_triggered_average, m)?)?;
    m.add_function(wrap_pyfunction!(py_spike_triggered_covariance, m)?)?;
    m.add_function(wrap_pyfunction!(py_spatial_information, m)?)?;
    m.add_function(wrap_pyfunction!(py_place_field_detection, m)?)?;
    m.add_function(wrap_pyfunction!(py_tuning_curve, m)?)?;
    // lfp
    m.add_function(wrap_pyfunction!(py_phase_locking_value, m)?)?;
    m.add_function(wrap_pyfunction!(py_spike_field_coherence, m)?)?;
    m.add_function(wrap_pyfunction!(py_spike_phase_histogram, m)?)?;
    // sorting_quality
    m.add_function(wrap_pyfunction!(py_isolation_distance, m)?)?;
    m.add_function(wrap_pyfunction!(py_l_ratio, m)?)?;
    m.add_function(wrap_pyfunction!(py_silhouette_score, m)?)?;
    m.add_function(wrap_pyfunction!(py_d_prime, m)?)?;
    m.add_function(wrap_pyfunction!(py_isi_violation_rate, m)?)?;
    m.add_function(wrap_pyfunction!(py_presence_ratio, m)?)?;
    m.add_function(wrap_pyfunction!(py_amplitude_cutoff, m)?)?;
    m.add_function(wrap_pyfunction!(py_snr, m)?)?;
    m.add_function(wrap_pyfunction!(py_nn_hit_rate, m)?)?;
    m.add_function(wrap_pyfunction!(py_drift_metric, m)?)?;
    // dimensionality
    m.add_function(wrap_pyfunction!(py_spike_train_pca, m)?)?;
    m.add_function(wrap_pyfunction!(py_demixed_pca, m)?)?;
    m.add_function(wrap_pyfunction!(py_factor_analysis, m)?)?;
    m.add_function(wrap_pyfunction!(py_pca_components, m)?)?;
    m.add_function(wrap_pyfunction!(py_demixed_components, m)?)?;
    m.add_function(wrap_pyfunction!(py_factor_loadings, m)?)?;
    // gpfa
    m.add_function(wrap_pyfunction!(py_gpfa, m)?)?;
    m.add_function(wrap_pyfunction!(py_gpfa_em, m)?)?;
    m.add_function(wrap_pyfunction!(py_gpfa_transform, m)?)?;
    // spade
    m.add_function(wrap_pyfunction!(py_spade_detect, m)?)?;
    // dna
    m.add_function(wrap_pyfunction!(py_dna_design_sequence, m)?)?;
    m.add_function(wrap_pyfunction!(py_dna_design_orthogonal_set, m)?)?;
    m.add_function(wrap_pyfunction!(py_dna_check_cross_hybridization, m)?)?;
    m.add_function(wrap_pyfunction!(py_dna_simulate_kinetics, m)?)?;
    m.add_function(wrap_pyfunction!(py_dna_detect_hairpins, m)?)?;
    // Quantum annealing acceleration
    m.add_function(wrap_pyfunction!(py_qa_ising_energy, m)?)?;
    m.add_function(wrap_pyfunction!(py_qa_batch_ising_energy, m)?)?;
    m.add_function(wrap_pyfunction!(py_qa_simulated_annealing, m)?)?;
    m.add_function(wrap_pyfunction!(py_qa_gauge_transform, m)?)?;
    m.add_function(wrap_pyfunction!(py_qa_generate_gauges, m)?)?;
    m.add_function(wrap_pyfunction!(py_qa_greedy_partition, m)?)?;
    // Photonic NoC acceleration
    m.add_function(wrap_pyfunction!(py_ph_route_waveguides, m)?)?;
    m.add_function(wrap_pyfunction!(py_ph_mzi_transfer_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(py_ph_cascade_mzi, m)?)?;
    m.add_function(wrap_pyfunction!(py_ph_analyze_crosstalk, m)?)?;
    m.add_function(wrap_pyfunction!(py_ph_analyze_power_budget, m)?)?;
    m.add_function(wrap_pyfunction!(py_ph_analyze_crosstalk_bank, m)?)?;
    m.add_function(wrap_pyfunction!(py_ph_analyze_crosstalk_pairs, m)?)?;
    // SC-Optimizer acceleration
    m.add_function(wrap_pyfunction!(py_opt_sa_search, m)?)?;
    m.add_function(wrap_pyfunction!(py_opt_extract_pareto, m)?)?;
    // Evolutionary substrate acceleration
    m.add_function(wrap_pyfunction!(py_evo_batch_mutate, m)?)?;
    m.add_function(wrap_pyfunction!(py_evo_batch_fitness, m)?)?;
    m.add_function(wrap_pyfunction!(py_evo_batch_crossover, m)?)?;
    m.add_function(wrap_pyfunction!(py_evo_diversity, m)?)?;
    m.add_function(wrap_pyfunction!(py_evo_novelty, m)?)?;
    m.add_function(wrap_pyfunction!(py_evo_tournament, m)?)?;
    // LGSSM Kalman filter (predictive_model)
    m.add_function(wrap_pyfunction!(py_lgssm_kalman_filter, m)?)?;
    m.add_function(wrap_pyfunction!(py_ollivier_ricci_curvature, m)?)?;
    m.add_function(wrap_pyfunction!(py_cazelles_map_simulate, m)?)?;
    m.add_function(wrap_pyfunction!(py_courage_nekorkin_map_simulate, m)?)?;
    m.add_function(wrap_pyfunction!(py_mckean_simulate, m)?)?;
    m.add_function(wrap_pyfunction!(py_wilson_hr_simulate, m)?)?;
    m.add_function(wrap_pyfunction!(py_pernarowski_simulate, m)?)?;
    m.add_function(wrap_pyfunction!(py_terman_wang_simulate, m)?)?;
    m.add_function(wrap_pyfunction!(py_mihalas_niebur_simulate, m)?)?;
    m.add_function(wrap_pyfunction!(py_glif_simulate, m)?)?;
    m.add_function(wrap_pyfunction!(py_rulkov_map_simulate, m)?)?;
    m.add_function(wrap_pyfunction!(py_ibarz_tanaka_map_simulate, m)?)?;
    m.add_function(wrap_pyfunction!(py_medvedev_map_simulate, m)?)?;
    m.add_function(wrap_pyfunction!(py_ermentrout_kopell_map_simulate, m)?)?;
    m.add_function(wrap_pyfunction!(py_fitzhugh_nagumo_simulate, m)?)?;
    m.add_function(wrap_pyfunction!(py_hindmarsh_rose_simulate, m)?)?;
    m.add_function(wrap_pyfunction!(py_fitzhugh_rinzel_simulate, m)?)?;
    m.add_function(wrap_pyfunction!(py_izhikevich2007_simulate, m)?)?;
    // Byte-level fault injection (parity with FaultInjector.inject)
    m.add_function(wrap_pyfunction!(py_inject_bitflip_u8, m)?)?;
    m.add_function(wrap_pyfunction!(py_inject_stuck_at_0_u8, m)?)?;
    m.add_function(wrap_pyfunction!(py_inject_stuck_at_1_u8, m)?)?;
    m.add_function(wrap_pyfunction!(py_inject_dropout_u8, m)?)?;
    m.add_function(wrap_pyfunction!(py_inject_gaussian_u8, m)?)?;
    // Hierarchical partitioner KL refine
    m.add_function(wrap_pyfunction!(py_kl_refine, m)?)?;
    // PINGCircuit per-step kernel
    m.add_function(wrap_pyfunction!(py_ping_step, m)?)?;
    // CorticalColumn block-CSR per-row-parallel spmv
    m.add_function(wrap_pyfunction!(py_parallel_csr_spmv_add, m)?)?;
    m.add_function(wrap_pyfunction!(py_parallel_csr_multi_spmv_add, m)?)?;
    Ok(())
}

// ── CORDIV + adaptive length ─────────────────────────────────────────

#[pyfunction]
fn py_cordiv(
    py: Python<'_>,
    numerator: PyReadonlyArray1<'_, u8>,
    denominator: PyReadonlyArray1<'_, u8>,
) -> PyResult<Py<PyArray1<u8>>> {
    let num = numerator.as_slice()?;
    let den = denominator.as_slice()?;
    let result = cordiv::cordiv(num, den);
    Ok(result.into_pyarray(py).into())
}

#[pyfunction]
fn py_adaptive_length(epsilon: f64, confidence: f64) -> usize {
    cordiv::adaptive_length_hoeffding(epsilon, confidence)
}

// ── Predictive coding ────────────────────────────────────────────────

#[pyfunction]
fn py_prediction_error(
    _py: Python<'_>,
    predicted: PyReadonlyArray1<'_, u64>,
    actual: PyReadonlyArray1<'_, u64>,
    length: usize,
) -> PyResult<f64> {
    let pred = predicted
        .as_slice()
        .map_err(|e| PyValueError::new_err(format!("predicted array must be contiguous: {e}")))?;
    let act = actual
        .as_slice()
        .map_err(|e| PyValueError::new_err(format!("actual array must be contiguous: {e}")))?;
    Ok(predictive_coding::prediction_error_packed(
        pred, act, length,
    ))
}

// ── Spike codec predictors (EMA + LFSR) ─────────────────────────────

#[pyfunction]
fn py_predict_xor_ema(
    _py: Python<'_>,
    spikes: PyReadonlyArray1<'_, i8>,
    n_channels: usize,
    alpha: f64,
    threshold: f64,
) -> PyResult<(Py<PyArray1<i8>>, usize)> {
    let data = spikes
        .as_slice()
        .map_err(|e| PyValueError::new_err(format!("spikes must be contiguous: {e}")))?;
    let (errors, correct) =
        predictive_coding::predict_and_xor_ema(data, n_channels, alpha, threshold);
    Ok((PyArray1::from_vec(_py, errors).into(), correct))
}

#[pyfunction]
fn py_recover_xor_ema(
    _py: Python<'_>,
    errors: PyReadonlyArray1<'_, i8>,
    n_channels: usize,
    alpha: f64,
    threshold: f64,
) -> PyResult<Py<PyArray1<i8>>> {
    let data = errors
        .as_slice()
        .map_err(|e| PyValueError::new_err(format!("errors must be contiguous: {e}")))?;
    let spikes = predictive_coding::xor_and_recover_ema(data, n_channels, alpha, threshold);
    Ok(PyArray1::from_vec(_py, spikes).into())
}

#[pyfunction]
fn py_predict_xor_lfsr(
    _py: Python<'_>,
    spikes: PyReadonlyArray1<'_, i8>,
    n_channels: usize,
    alpha_q8: i32,
    seed: u16,
) -> PyResult<(Py<PyArray1<i8>>, usize)> {
    let data = spikes
        .as_slice()
        .map_err(|e| PyValueError::new_err(format!("spikes must be contiguous: {e}")))?;
    let (errors, correct) =
        predictive_coding::predict_and_xor_lfsr(data, n_channels, alpha_q8, seed);
    Ok((PyArray1::from_vec(_py, errors).into(), correct))
}

#[pyfunction]
fn py_recover_xor_lfsr(
    _py: Python<'_>,
    errors: PyReadonlyArray1<'_, i8>,
    n_channels: usize,
    alpha_q8: i32,
    seed: u16,
) -> PyResult<Py<PyArray1<i8>>> {
    let data = errors
        .as_slice()
        .map_err(|e| PyValueError::new_err(format!("errors must be contiguous: {e}")))?;
    let spikes = predictive_coding::xor_and_recover_lfsr(data, n_channels, alpha_q8, seed);
    Ok(PyArray1::from_vec(_py, spikes).into())
}

// ── Phi* ─────────────────────────────────────────────────────────────

#[pyfunction]
fn py_phi_star(_py: Python<'_>, data: PyReadonlyArray2<'_, f64>, tau: usize) -> PyResult<f64> {
    if !data.is_c_contiguous() {
        return Err(PyValueError::new_err(
            "py_phi_star requires C-contiguous array input",
        ));
    }
    let shape = data.shape();
    let n_channels = shape[0];
    let n_timesteps = shape[1];
    let flat = data.as_slice().map_err(|e| {
        PyValueError::new_err(format!("py_phi_star requires C-contiguous array: {e}"))
    })?;
    let channels: Vec<Vec<f64>> = (0..n_channels)
        .map(|i| flat[i * n_timesteps..(i + 1) * n_timesteps].to_vec())
        .collect();
    Ok(phi::phi_star(&channels, tau))
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

#[pyfunction]
fn popcount(packed: &Bound<'_, PyAny>) -> PyResult<u64> {
    if let Ok(rows) = packed.extract::<Vec<Vec<u64>>>() {
        return Ok(rows
            .iter()
            .map(|row| simd::popcount_dispatch(row))
            .sum::<u64>());
    }

    let words = packed.extract::<Vec<u64>>().map_err(|_| {
        PyValueError::new_err("Expected packed uint64 words as 1-D or 2-D sequence.")
    })?;
    Ok(simd::popcount_dispatch(&words))
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

fn parse_surrogate(name: &str, k: Option<f32>) -> PyResult<grad::SurrogateType> {
    let normalized = name.to_ascii_lowercase().replace('-', "_");
    match normalized.as_str() {
        "fast_sigmoid" => Ok(grad::SurrogateType::FastSigmoid {
            k: k.unwrap_or(25.0),
        }),
        "superspike" | "super_spike" => Ok(grad::SurrogateType::SuperSpike {
            k: k.unwrap_or(100.0),
        }),
        "arctan" | "arc_tan" => Ok(grad::SurrogateType::ArcTan { k: k.unwrap_or(10.0) }),
        "straightthrough" | "straight_through" | "ste" => Ok(grad::SurrogateType::StraightThrough),
        _ => Err(PyValueError::new_err(format!(
            "Unknown surrogate '{}'. Use one of: fast_sigmoid, superspike, arctan, straight_through.",
            name
        ))),
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
    name = "SurrogateLif",
    module = "sc_neurocore_engine.sc_neurocore_engine"
)]
pub struct PySurrogateLif {
    inner: grad::SurrogateLif,
}

#[pymethods]
impl PySurrogateLif {
    #[new]
    #[pyo3(signature = (
        data_width=16,
        fraction=8,
        v_rest=0,
        v_reset=0,
        v_threshold=256,
        refractory_period=2,
        surrogate="fast_sigmoid",
        k=None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        data_width: u32,
        fraction: u32,
        v_rest: i16,
        v_reset: i16,
        v_threshold: i16,
        refractory_period: i32,
        surrogate: &str,
        k: Option<f32>,
    ) -> PyResult<Self> {
        let surrogate = parse_surrogate(surrogate, k)?;
        Ok(Self {
            inner: grad::SurrogateLif::new(
                data_width,
                fraction,
                v_rest,
                v_reset,
                v_threshold,
                refractory_period,
                surrogate,
            ),
        })
    }

    #[pyo3(signature = (leak_k, gain_k, i_t, noise_in=0))]
    fn forward(&mut self, leak_k: i16, gain_k: i16, i_t: i16, noise_in: i16) -> (i32, i16) {
        self.inner.forward(leak_k, gain_k, i_t, noise_in)
    }

    fn backward(&mut self, grad_output: f32) -> PyResult<f32> {
        self.inner
            .backward(grad_output)
            .map_err(PyValueError::new_err)
    }

    fn clear_trace(&mut self) {
        self.inner.clear_trace();
    }

    fn reset(&mut self) {
        self.inner.reset();
    }

    fn trace_len(&self) -> usize {
        self.inner.trace_len()
    }
}

#[pyclass(
    name = "DifferentiableDenseLayer",
    module = "sc_neurocore_engine.sc_neurocore_engine"
)]
pub struct PyDifferentiableDenseLayer {
    inner: grad::DifferentiableDenseLayer,
}

#[pymethods]
impl PyDifferentiableDenseLayer {
    #[new]
    #[pyo3(signature = (
        n_inputs,
        n_neurons,
        length=1024,
        seed=24301,
        surrogate="fast_sigmoid",
        k=None
    ))]
    fn new(
        n_inputs: usize,
        n_neurons: usize,
        length: usize,
        seed: u64,
        surrogate: &str,
        k: Option<f32>,
    ) -> PyResult<Self> {
        let surrogate = parse_surrogate(surrogate, k)?;
        Ok(Self {
            inner: grad::DifferentiableDenseLayer::new(
                n_inputs, n_neurons, length, seed, surrogate,
            ),
        })
    }

    fn get_weights(&self) -> Vec<Vec<f64>> {
        self.inner.layer.get_weights()
    }

    #[pyo3(signature = (input_values, seed=44257))]
    fn forward(&mut self, input_values: Vec<f64>, seed: u64) -> PyResult<Vec<f64>> {
        self.inner
            .forward(&input_values, seed)
            .map_err(PyValueError::new_err)
    }

    fn backward(&self, grad_output: Vec<f64>) -> PyResult<(Vec<f64>, Vec<Vec<f64>>)> {
        self.inner
            .backward(&grad_output)
            .map_err(PyValueError::new_err)
    }

    fn update_weights(&mut self, weight_grads: Vec<Vec<f64>>, lr: f64) -> PyResult<()> {
        if weight_grads.len() != self.inner.layer.n_neurons {
            return Err(PyValueError::new_err(format!(
                "Expected {} grad rows, got {}.",
                self.inner.layer.n_neurons,
                weight_grads.len()
            )));
        }
        if weight_grads
            .iter()
            .any(|row| row.len() != self.inner.layer.n_inputs)
        {
            return Err(PyValueError::new_err(format!(
                "Expected each grad row to have length {}.",
                self.inner.layer.n_inputs
            )));
        }
        self.inner.update_weights(&weight_grads, lr);
        Ok(())
    }

    fn clear_cache(&mut self) {
        self.inner.clear_cache();
    }
}

#[pyclass(
    name = "StochasticAttention",
    module = "sc_neurocore_engine.sc_neurocore_engine"
)]
pub struct PyStochasticAttention {
    inner: attention::StochasticAttention,
}

#[pymethods]
impl PyStochasticAttention {
    #[new]
    #[pyo3(signature = (dim_k, temperature=None))]
    fn new(dim_k: usize, temperature: Option<f64>) -> Self {
        Self {
            inner: match temperature {
                Some(t) => attention::StochasticAttention::with_temperature(dim_k, t),
                None => attention::StochasticAttention::new(dim_k),
            },
        }
    }

    fn forward_softmax(
        &self,
        q: &Bound<'_, PyAny>,
        k: &Bound<'_, PyAny>,
        v: &Bound<'_, PyAny>,
    ) -> PyResult<Vec<Vec<f64>>> {
        let (q_data, q_rows, q_cols) = extract_matrix_f64(q, "Q")?;
        let (k_data, k_rows, k_cols) = extract_matrix_f64(k, "K")?;
        let (v_data, v_rows, v_cols) = extract_matrix_f64(v, "V")?;

        let out = self
            .inner
            .forward_softmax(
                &q_data, q_rows, q_cols, &k_data, k_rows, k_cols, &v_data, v_rows, v_cols,
            )
            .map_err(PyValueError::new_err)?;

        Ok(reshape_flat_to_rows(out, q_rows, v_cols))
    }

    #[pyo3(signature = (q, k, v, n_heads))]
    fn forward_multihead_softmax(
        &self,
        q: &Bound<'_, PyAny>,
        k: &Bound<'_, PyAny>,
        v: &Bound<'_, PyAny>,
        n_heads: usize,
    ) -> PyResult<Vec<Vec<f64>>> {
        let (q_data, q_rows, q_cols) = extract_matrix_f64(q, "Q")?;
        let (k_data, k_rows, k_cols) = extract_matrix_f64(k, "K")?;
        let (v_data, v_rows, v_cols) = extract_matrix_f64(v, "V")?;

        let out = self
            .inner
            .forward_multihead_softmax(
                &q_data, q_rows, q_cols, &k_data, k_rows, k_cols, &v_data, v_rows, v_cols, n_heads,
            )
            .map_err(PyValueError::new_err)?;

        let out_cols = v_cols;
        Ok(reshape_flat_to_rows(out, q_rows, out_cols))
    }

    fn forward(
        &self,
        q: &Bound<'_, PyAny>,
        k: &Bound<'_, PyAny>,
        v: &Bound<'_, PyAny>,
    ) -> PyResult<Vec<Vec<f64>>> {
        let (q_data, q_rows, q_cols) = extract_matrix_f64(q, "Q")?;
        let (k_data, k_rows, k_cols) = extract_matrix_f64(k, "K")?;
        let (v_data, v_rows, v_cols) = extract_matrix_f64(v, "V")?;

        let out = self
            .inner
            .forward(
                &q_data, q_rows, q_cols, &k_data, k_rows, k_cols, &v_data, v_rows, v_cols,
            )
            .map_err(PyValueError::new_err)?;

        Ok(reshape_flat_to_rows(out, q_rows, v_cols))
    }

    #[pyo3(signature = (q, k, v, length=1024, seed=44257))]
    fn forward_sc(
        &self,
        q: &Bound<'_, PyAny>,
        k: &Bound<'_, PyAny>,
        v: &Bound<'_, PyAny>,
        length: usize,
        seed: u64,
    ) -> PyResult<Vec<Vec<f64>>> {
        let (q_data, q_rows, q_cols) = extract_matrix_f64(q, "Q")?;
        let (k_data, k_rows, k_cols) = extract_matrix_f64(k, "K")?;
        let (v_data, v_rows, v_cols) = extract_matrix_f64(v, "V")?;

        let out = self
            .inner
            .forward_sc(
                &q_data, q_rows, q_cols, &k_data, k_rows, k_cols, &v_data, v_rows, v_cols, length,
                seed,
            )
            .map_err(PyValueError::new_err)?;

        Ok(reshape_flat_to_rows(out, q_rows, v_cols))
    }

    #[pyo3(signature = (q, k, v, n_heads))]
    fn forward_multihead(
        &self,
        q: &Bound<'_, PyAny>,
        k: &Bound<'_, PyAny>,
        v: &Bound<'_, PyAny>,
        n_heads: usize,
    ) -> PyResult<Vec<Vec<f64>>> {
        let (q_data, q_rows, q_cols) = extract_matrix_f64(q, "Q")?;
        let (k_data, k_rows, k_cols) = extract_matrix_f64(k, "K")?;
        let (v_data, v_rows, v_cols) = extract_matrix_f64(v, "V")?;

        let out = self
            .inner
            .forward_multihead(
                &q_data, q_rows, q_cols, &k_data, k_rows, k_cols, &v_data, v_rows, v_cols, n_heads,
            )
            .map_err(PyValueError::new_err)?;

        let out_cols = v_cols;
        Ok(reshape_flat_to_rows(out, q_rows, out_cols))
    }
}

#[pyclass(
    name = "StochasticGraphLayer",
    module = "sc_neurocore_engine.sc_neurocore_engine"
)]
pub struct PyStochasticGraphLayer {
    inner: graph::StochasticGraphLayer,
}

#[pymethods]
impl PyStochasticGraphLayer {
    #[new]
    #[pyo3(signature = (adj_matrix, n_features, seed=42))]
    fn new(adj_matrix: &Bound<'_, PyAny>, n_features: usize, seed: u64) -> PyResult<Self> {
        let (adj_flat, n_rows, n_cols) = extract_matrix_f64(adj_matrix, "adj_matrix")?;
        if n_rows != n_cols {
            return Err(PyValueError::new_err(format!(
                "adj_matrix must be square, got {}x{}.",
                n_rows, n_cols
            )));
        }
        Ok(Self {
            inner: graph::StochasticGraphLayer::new(adj_flat, n_rows, n_features, seed),
        })
    }

    /// Construct from CSR arrays (row_offsets, col_indices, values).
    #[staticmethod]
    #[pyo3(signature = (row_offsets, col_indices, values, n_nodes, n_features, seed=42))]
    fn from_sparse(
        row_offsets: Vec<usize>,
        col_indices: Vec<usize>,
        values: Vec<f64>,
        n_nodes: usize,
        n_features: usize,
        seed: u64,
    ) -> PyResult<Self> {
        let csr = graph::CsrMatrix::new(row_offsets, col_indices, values, n_nodes, n_nodes)
            .map_err(PyValueError::new_err)?;
        let inner = graph::StochasticGraphLayer::new_sparse(csr, n_features, seed)
            .map_err(PyValueError::new_err)?;
        Ok(Self { inner })
    }

    /// Dense adjacency with automatic CSR conversion if density < threshold.
    #[staticmethod]
    #[pyo3(signature = (adj_matrix, n_features, seed=42, density_threshold=0.3))]
    fn from_dense_auto(
        adj_matrix: &Bound<'_, PyAny>,
        n_features: usize,
        seed: u64,
        density_threshold: f64,
    ) -> PyResult<Self> {
        let (adj_flat, n_rows, n_cols) = extract_matrix_f64(adj_matrix, "adj_matrix")?;
        if n_rows != n_cols {
            return Err(PyValueError::new_err(format!(
                "adj_matrix must be square, got {}x{}.",
                n_rows, n_cols
            )));
        }
        Ok(Self {
            inner: graph::StochasticGraphLayer::from_dense_auto(
                adj_flat,
                n_rows,
                n_features,
                seed,
                density_threshold,
            ),
        })
    }

    fn is_sparse(&self) -> bool {
        self.inner.is_sparse()
    }

    fn forward(&self, node_features: &Bound<'_, PyAny>) -> PyResult<Vec<Vec<f64>>> {
        let (x_flat, x_rows, x_cols) = extract_matrix_f64(node_features, "node_features")?;
        if x_rows != self.inner.n_nodes || x_cols != self.inner.n_features {
            return Err(PyValueError::new_err(format!(
                "Expected node_features shape ({}, {}), got ({}, {}).",
                self.inner.n_nodes, self.inner.n_features, x_rows, x_cols
            )));
        }
        let out = self.inner.forward(&x_flat).map_err(PyValueError::new_err)?;
        Ok(reshape_flat_to_rows(
            out,
            self.inner.n_nodes,
            self.inner.n_features,
        ))
    }

    #[pyo3(signature = (node_features, length=1024, seed=44257))]
    fn forward_sc(
        &self,
        node_features: &Bound<'_, PyAny>,
        length: usize,
        seed: u64,
    ) -> PyResult<Vec<Vec<f64>>> {
        let (x_flat, x_rows, x_cols) = extract_matrix_f64(node_features, "node_features")?;
        if x_rows != self.inner.n_nodes || x_cols != self.inner.n_features {
            return Err(PyValueError::new_err(format!(
                "Expected node_features shape ({}, {}), got ({}, {}).",
                self.inner.n_nodes, self.inner.n_features, x_rows, x_cols
            )));
        }
        let out = self
            .inner
            .forward_sc(&x_flat, length, seed)
            .map_err(PyValueError::new_err)?;
        Ok(reshape_flat_to_rows(
            out,
            self.inner.n_nodes,
            self.inner.n_features,
        ))
    }

    fn get_weights(&self) -> Vec<f64> {
        self.inner.get_weights()
    }

    fn set_weights(&mut self, weights: Vec<f64>) -> PyResult<()> {
        self.inner
            .set_weights(weights)
            .map_err(PyValueError::new_err)
    }
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

// IR bridge

#[pyclass(name = "ScGraph", module = "sc_neurocore_engine.sc_neurocore_engine")]
pub struct PyScGraph {
    inner: ir::graph::ScGraph,
}

#[pymethods]
impl PyScGraph {
    /// Number of operations in the graph.
    fn len(&self) -> usize {
        self.inner.len()
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// Whether the graph is empty.
    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Graph name.
    #[getter]
    fn name(&self) -> &str {
        &self.inner.name
    }

    /// Number of input ports.
    fn num_inputs(&self) -> usize {
        self.inner.inputs().len()
    }

    /// Number of output ports.
    fn num_outputs(&self) -> usize {
        self.inner.outputs().len()
    }

    fn __repr__(&self) -> String {
        format!("ScGraph('{}', ops={})", self.inner.name, self.inner.len())
    }
}

#[pyclass(
    name = "ScGraphBuilder",
    module = "sc_neurocore_engine.sc_neurocore_engine"
)]
pub struct PyScGraphBuilder {
    inner: Option<ir::builder::ScGraphBuilder>,
}

impl PyScGraphBuilder {
    fn builder_mut(&mut self) -> PyResult<&mut ir::builder::ScGraphBuilder> {
        self.inner
            .as_mut()
            .ok_or_else(|| PyValueError::new_err("Builder already consumed by build()."))
    }
}

#[pymethods]
impl PyScGraphBuilder {
    #[new]
    fn new(name: String) -> Self {
        Self {
            inner: Some(ir::builder::ScGraphBuilder::new(name)),
        }
    }

    /// Add a typed input port. Returns value ID.
    fn input(&mut self, name: &str, ty: &str) -> PyResult<u32> {
        let sc_type = parse_sc_type(ty)?;
        Ok(self.builder_mut()?.input(name, sc_type).0)
    }

    /// Add an output port forwarding a value.
    fn output(&mut self, name: &str, source_id: u32) -> PyResult<u32> {
        Ok(self
            .builder_mut()?
            .output(name, ir::graph::ValueId(source_id))
            .0)
    }

    /// Add a float constant.
    fn constant_f64(&mut self, value: f64, ty: &str) -> PyResult<u32> {
        let sc_type = parse_sc_type(ty)?;
        Ok(self
            .builder_mut()?
            .constant(ir::graph::ScConst::F64(value), sc_type)
            .0)
    }

    /// Add an integer constant.
    fn constant_i64(&mut self, value: i64, ty: &str) -> PyResult<u32> {
        let sc_type = parse_sc_type(ty)?;
        Ok(self
            .builder_mut()?
            .constant(ir::graph::ScConst::I64(value), sc_type)
            .0)
    }

    /// Add a Bernoulli encode operation.
    fn encode(&mut self, prob_id: u32, length: usize, seed: u64) -> PyResult<u32> {
        let seed = u16::try_from(seed)
            .map_err(|_| PyValueError::new_err(format!("Seed out of range for u16: {seed}")))?;
        Ok(self
            .builder_mut()?
            .encode(ir::graph::ValueId(prob_id), length, seed)
            .0)
    }

    /// Add a bitwise AND (SC multiply).
    fn bitwise_and(&mut self, lhs_id: u32, rhs_id: u32) -> PyResult<u32> {
        Ok(self
            .builder_mut()?
            .bitwise_and(ir::graph::ValueId(lhs_id), ir::graph::ValueId(rhs_id))
            .0)
    }

    /// Add a popcount operation.
    fn popcount(&mut self, input_id: u32) -> PyResult<u32> {
        Ok(self.builder_mut()?.popcount(ir::graph::ValueId(input_id)).0)
    }

    /// Add a LIF neuron step.
    #[pyo3(signature = (
        current_id,
        leak_id,
        gain_id,
        noise_id,
        data_width=16,
        fraction=8,
        v_rest=0,
        v_reset=0,
        v_threshold=256,
        refractory_period=2
    ))]
    #[allow(clippy::too_many_arguments)]
    fn lif_step(
        &mut self,
        current_id: u32,
        leak_id: u32,
        gain_id: u32,
        noise_id: u32,
        data_width: u32,
        fraction: u32,
        v_rest: i64,
        v_reset: i64,
        v_threshold: i64,
        refractory_period: u32,
    ) -> PyResult<u32> {
        let params = ir::graph::LifParams {
            data_width,
            fraction,
            v_rest,
            v_reset,
            v_threshold,
            refractory_period,
        };
        Ok(self
            .builder_mut()?
            .lif_step(
                ir::graph::ValueId(current_id),
                ir::graph::ValueId(leak_id),
                ir::graph::ValueId(gain_id),
                ir::graph::ValueId(noise_id),
                params,
            )
            .0)
    }

    /// Add a dense layer forward pass.
    #[pyo3(signature = (
        inputs_id,
        weights_id,
        leak_id,
        gain_id,
        n_inputs=3,
        n_neurons=7,
        data_width=16,
        stream_length=1024,
        seed_base=0xACE1u64,
        y_min=0,
        y_max=65535
    ))]
    #[allow(clippy::too_many_arguments)]
    fn dense_forward(
        &mut self,
        inputs_id: u32,
        weights_id: u32,
        leak_id: u32,
        gain_id: u32,
        n_inputs: usize,
        n_neurons: usize,
        data_width: u32,
        stream_length: usize,
        seed_base: u64,
        y_min: i64,
        y_max: i64,
    ) -> PyResult<u32> {
        let input_seed_base = u16::try_from(seed_base).map_err(|_| {
            PyValueError::new_err(format!("seed_base out of range for u16: {seed_base}"))
        })?;
        let params = ir::graph::DenseParams {
            n_inputs,
            n_neurons,
            data_width,
            stream_length,
            input_seed_base,
            weight_seed_base: input_seed_base.wrapping_add(1),
            y_min,
            y_max,
        };
        Ok(self
            .builder_mut()?
            .dense_forward(
                ir::graph::ValueId(inputs_id),
                ir::graph::ValueId(weights_id),
                ir::graph::ValueId(leak_id),
                ir::graph::ValueId(gain_id),
                params,
            )
            .0)
    }

    /// Add a scale (multiply by constant factor) operation.
    fn scale(&mut self, input_id: u32, factor: f64) -> PyResult<u32> {
        Ok(self
            .builder_mut()?
            .scale(ir::graph::ValueId(input_id), factor)
            .0)
    }

    /// Add an offset (add constant) operation.
    fn offset(&mut self, input_id: u32, offset_val: f64) -> PyResult<u32> {
        Ok(self
            .builder_mut()?
            .offset(ir::graph::ValueId(input_id), offset_val)
            .0)
    }

    /// Add a divide-by-constant operation.
    fn div_const(&mut self, input_id: u32, divisor: u64) -> PyResult<u32> {
        Ok(self
            .builder_mut()?
            .div_const(ir::graph::ValueId(input_id), divisor)
            .0)
    }

    /// Consume the builder and return a graph.
    fn build(&mut self) -> PyResult<PyScGraph> {
        let builder = self
            .inner
            .take()
            .ok_or_else(|| PyValueError::new_err("Builder already consumed by build()."))?;
        Ok(PyScGraph {
            inner: builder.build(),
        })
    }
}

/// Verify an IR graph. Returns None on success, or a list of error strings.
#[pyfunction]
fn ir_verify(graph: PyRef<'_, PyScGraph>) -> Option<Vec<String>> {
    match ir::verify::verify(&graph.inner) {
        Ok(()) => None,
        Err(errors) => Some(errors.iter().map(|e| e.to_string()).collect()),
    }
}

/// Print an IR graph to its stable text format.
#[pyfunction]
fn ir_print(graph: PyRef<'_, PyScGraph>) -> String {
    ir::printer::print(&graph.inner)
}

/// Parse an IR graph from text format.
#[pyfunction]
fn ir_parse(text: &str) -> PyResult<PyScGraph> {
    ir::parser::parse(text)
        .map(|graph| PyScGraph { inner: graph })
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Emit SystemVerilog from an IR graph.
#[pyfunction]
fn ir_emit_sv(graph: PyRef<'_, PyScGraph>) -> PyResult<String> {
    ir::emit_sv::emit(&graph.inner).map_err(PyValueError::new_err)
}

/// Parse a Python type string into ScType.
///
/// Accepted formats: "bool", "rate", "u32", "u64", "i16", "i32",
/// "bitstream", "bitstream<1024>", "fixed<16,8>", "vec<bool,7>".
fn parse_sc_type(s: &str) -> PyResult<ir::graph::ScType> {
    let s = s.trim();
    let lower = s.to_ascii_lowercase();
    match lower.as_str() {
        "bool" => Ok(ir::graph::ScType::Bool),
        "rate" => Ok(ir::graph::ScType::Rate),
        "u32" => Ok(ir::graph::ScType::UInt { width: 32 }),
        "u64" => Ok(ir::graph::ScType::UInt { width: 64 }),
        "i16" => Ok(ir::graph::ScType::SInt { width: 16 }),
        "i32" => Ok(ir::graph::ScType::SInt { width: 32 }),
        "bitstream" => Ok(ir::graph::ScType::Bitstream { length: 0 }),
        _ => {
            if let Some(width) = lower.strip_prefix('u') {
                if let Ok(width) = width.parse::<u32>() {
                    return Ok(ir::graph::ScType::UInt { width });
                }
            }
            if let Some(width) = lower.strip_prefix('i') {
                if let Ok(width) = width.parse::<u32>() {
                    return Ok(ir::graph::ScType::SInt { width });
                }
            }
            if let Some(inner) = lower
                .strip_prefix("bitstream<")
                .and_then(|r| r.strip_suffix('>'))
            {
                let length = inner.parse::<usize>().map_err(|_| {
                    PyValueError::new_err(format!("Invalid bitstream length: '{inner}'"))
                })?;
                return Ok(ir::graph::ScType::Bitstream { length });
            }
            if let Some(inner) = lower
                .strip_prefix("fixed<")
                .and_then(|r| r.strip_suffix('>'))
            {
                let parts: Vec<&str> = inner.split(',').collect();
                if parts.len() != 2 {
                    return Err(PyValueError::new_err(format!(
                        "fixed type needs 2 params: '{s}'"
                    )));
                }
                let width = parts[0].trim().parse::<u32>().map_err(|_| {
                    PyValueError::new_err(format!("Invalid fixed width: '{}'", parts[0]))
                })?;
                let frac = parts[1].trim().parse::<u32>().map_err(|_| {
                    PyValueError::new_err(format!("Invalid fixed frac: '{}'", parts[1]))
                })?;
                return Ok(ir::graph::ScType::FixedPoint { width, frac });
            }
            if let Some(inner) = lower.strip_prefix("vec<").and_then(|r| r.strip_suffix('>')) {
                if let Some(comma_pos) = inner.rfind(',') {
                    let inner_ty_str = &inner[..comma_pos];
                    let count_str = inner[comma_pos + 1..].trim();
                    let inner_ty = parse_sc_type(inner_ty_str)?;
                    let count = count_str.parse::<usize>().map_err(|_| {
                        PyValueError::new_err(format!("Invalid vec count: '{count_str}'"))
                    })?;
                    return Ok(ir::graph::ScType::Vec {
                        element: Box::new(inner_ty),
                        count,
                    });
                }
            }
            Err(PyValueError::new_err(format!("Unknown IR type: '{s}'")))
        }
    }
}

// ── Analysis PyO3 wrappers (P0-A: spike_stats) ─────────────────────

#[pyfunction]
#[pyo3(signature = (binary_train, dt=0.001))]
fn py_spike_times(
    py: Python<'_>,
    binary_train: PyReadonlyArray1<'_, i32>,
    dt: f64,
) -> Py<PyArray1<f64>> {
    let data = binary_train.as_slice().unwrap();
    analysis::basic::spike_times(data, dt)
        .into_pyarray(py)
        .into()
}

#[pyfunction]
#[pyo3(signature = (binary_train, dt=0.001))]
fn py_isi(py: Python<'_>, binary_train: PyReadonlyArray1<'_, i32>, dt: f64) -> Py<PyArray1<f64>> {
    let data = binary_train.as_slice().unwrap();
    analysis::basic::isi(data, dt).into_pyarray(py).into()
}

#[pyfunction]
#[pyo3(signature = (binary_train, dt=0.001))]
fn py_firing_rate(binary_train: PyReadonlyArray1<'_, i32>, dt: f64) -> f64 {
    let data = binary_train.as_slice().unwrap();
    analysis::basic::firing_rate(data, dt)
}

#[pyfunction]
fn py_spike_count(binary_train: PyReadonlyArray1<'_, i32>) -> i64 {
    let data = binary_train.as_slice().unwrap();
    analysis::basic::spike_count(data)
}

#[pyfunction]
#[pyo3(signature = (binary_train, bin_size=10))]
fn py_bin_spike_train(
    py: Python<'_>,
    binary_train: PyReadonlyArray1<'_, i32>,
    bin_size: usize,
) -> Py<PyArray1<i64>> {
    let data = binary_train.as_slice().unwrap();
    analysis::basic::bin_spike_train(data, bin_size)
        .into_pyarray(py)
        .into()
}

#[pyfunction]
#[pyo3(signature = (binary_train, dt=0.001, kernel="gaussian", sigma_ms=10.0))]
fn py_instantaneous_rate(
    py: Python<'_>,
    binary_train: PyReadonlyArray1<'_, f64>,
    dt: f64,
    kernel: &str,
    sigma_ms: f64,
) -> Py<PyArray1<f64>> {
    let data = binary_train.as_slice().unwrap();
    analysis::rate::instantaneous_rate(data, dt, kernel, sigma_ms)
        .into_pyarray(py)
        .into()
}

#[pyfunction]
#[pyo3(signature = (trials, bin_ms=10.0, dt=0.001))]
fn py_psth(
    py: Python<'_>,
    trials: Vec<PyReadonlyArray1<'_, f64>>,
    bin_ms: f64,
    dt: f64,
) -> (Py<PyArray1<f64>>, Py<PyArray1<f64>>) {
    let vecs: Vec<Vec<f64>> = trials
        .iter()
        .map(|t| t.as_slice().unwrap().to_vec())
        .collect();
    let (rates, centers) = analysis::rate::psth(&vecs, bin_ms, dt);
    (
        rates.into_pyarray(py).into(),
        centers.into_pyarray(py).into(),
    )
}

#[pyfunction]
#[pyo3(signature = (binary_train, dt=0.001))]
fn py_cv_isi(binary_train: PyReadonlyArray1<'_, i32>, dt: f64) -> f64 {
    let data = binary_train.as_slice().unwrap();
    analysis::variability::cv_isi(data, dt)
}

#[pyfunction]
#[pyo3(signature = (binary_train, dt=0.001))]
fn py_cv2(binary_train: PyReadonlyArray1<'_, i32>, dt: f64) -> f64 {
    let data = binary_train.as_slice().unwrap();
    analysis::variability::cv2(data, dt)
}

#[pyfunction]
#[pyo3(signature = (binary_train, dt=0.001))]
fn py_local_variation(binary_train: PyReadonlyArray1<'_, i32>, dt: f64) -> f64 {
    let data = binary_train.as_slice().unwrap();
    analysis::variability::local_variation(data, dt)
}

#[pyfunction]
#[pyo3(signature = (binary_train, window_ms=50.0, dt=0.001))]
fn py_fano_factor(binary_train: PyReadonlyArray1<'_, i32>, window_ms: f64, dt: f64) -> f64 {
    let data = binary_train.as_slice().unwrap();
    analysis::variability::fano_factor(data, window_ms, dt)
}

#[pyfunction]
fn py_lempel_ziv_complexity(binary_train: PyReadonlyArray1<'_, i32>) -> f64 {
    let data = binary_train.as_slice().unwrap();
    analysis::variability::lempel_ziv_complexity(data)
}

#[pyfunction]
#[pyo3(signature = (binary_train, order=3, delay=1))]
fn py_permutation_entropy(
    binary_train: PyReadonlyArray1<'_, i32>,
    order: usize,
    delay: usize,
) -> f64 {
    let data = binary_train.as_slice().unwrap();
    analysis::variability::permutation_entropy(data, order, delay)
}

#[pyfunction]
#[pyo3(signature = (binary_train, min_window=10))]
fn py_hurst_exponent(binary_train: PyReadonlyArray1<'_, i32>, min_window: usize) -> f64 {
    let data = binary_train.as_slice().unwrap();
    analysis::variability::hurst_exponent(data, min_window)
}

#[pyfunction]
#[pyo3(signature = (binary_train, m=2, r_factor=0.2))]
fn py_approximate_entropy(binary_train: PyReadonlyArray1<'_, i32>, m: usize, r_factor: f64) -> f64 {
    let data = binary_train.as_slice().unwrap();
    analysis::variability::approximate_entropy(data, m, r_factor)
}

#[pyfunction]
#[pyo3(signature = (binary_train, m=2, r_factor=0.2))]
fn py_sample_entropy(binary_train: PyReadonlyArray1<'_, i32>, m: usize, r_factor: f64) -> f64 {
    let data = binary_train.as_slice().unwrap();
    analysis::variability::sample_entropy(data, m, r_factor)
}

// ── Correlation PyO3 wrappers (P0-A: spike_stats/correlation) ────

#[pyfunction]
#[pyo3(signature = (train_a, train_b, max_lag_ms=50.0, dt=0.001))]
fn py_cross_correlation(
    py: Python<'_>,
    train_a: PyReadonlyArray1<'_, i32>,
    train_b: PyReadonlyArray1<'_, i32>,
    max_lag_ms: f64,
    dt: f64,
) -> (Py<PyArray1<f64>>, Py<PyArray1<f64>>) {
    let a = train_a.as_slice().unwrap();
    let b = train_b.as_slice().unwrap();
    let (cc, lags) = analysis::correlation::cross_correlation(a, b, max_lag_ms, dt);
    (cc.into_pyarray(py).into(), lags.into_pyarray(py).into())
}

#[pyfunction]
#[pyo3(signature = (trains, dt=0.001))]
fn py_pairwise_correlation(
    py: Python<'_>,
    trains: Vec<PyReadonlyArray1<'_, i32>>,
    dt: f64,
) -> Py<PyArray2<f64>> {
    let vecs: Vec<Vec<i32>> = trains
        .iter()
        .map(|t| t.as_slice().unwrap().to_vec())
        .collect();
    let refs: Vec<&[i32]> = vecs.iter().map(|v| v.as_slice()).collect();
    let mat = analysis::correlation::pairwise_correlation(&refs, dt);
    let n = mat.len();
    let flat: Vec<f64> = mat.into_iter().flatten().collect();
    numpy::PyArray2::from_vec2(py, &flat.chunks(n).map(|c| c.to_vec()).collect::<Vec<_>>())
        .unwrap()
        .into()
}

#[pyfunction]
#[pyo3(signature = (train_a, train_b, dt=0.001, tau_ms=5.0))]
fn py_event_synchronization(
    train_a: PyReadonlyArray1<'_, i32>,
    train_b: PyReadonlyArray1<'_, i32>,
    dt: f64,
    tau_ms: f64,
) -> f64 {
    let a = train_a.as_slice().unwrap();
    let b = train_b.as_slice().unwrap();
    analysis::correlation::event_synchronization(a, b, dt, tau_ms)
}

#[pyfunction]
#[pyo3(signature = (train_a, train_b, dt=0.001))]
fn py_spike_train_coherence(
    py: Python<'_>,
    train_a: PyReadonlyArray1<'_, i32>,
    train_b: PyReadonlyArray1<'_, i32>,
    dt: f64,
) -> (Py<PyArray1<f64>>, Py<PyArray1<f64>>) {
    let a = train_a.as_slice().unwrap();
    let b = train_b.as_slice().unwrap();
    let (coh, freqs) = analysis::correlation::spike_train_coherence(a, b, dt);
    (coh.into_pyarray(py).into(), freqs.into_pyarray(py).into())
}

#[pyfunction]
#[pyo3(signature = (train_a, train_b, dt=0.001, delta_ms=5.0))]
fn py_spike_time_tiling_coefficient(
    train_a: PyReadonlyArray1<'_, i32>,
    train_b: PyReadonlyArray1<'_, i32>,
    dt: f64,
    delta_ms: f64,
) -> f64 {
    let a = train_a.as_slice().unwrap();
    let b = train_b.as_slice().unwrap();
    analysis::correlation::spike_time_tiling_coefficient(a, b, dt, delta_ms)
}

#[pyfunction]
#[pyo3(signature = (trains, bin_size=10))]
fn py_covariance_matrix(
    py: Python<'_>,
    trains: Vec<PyReadonlyArray1<'_, i32>>,
    bin_size: usize,
) -> Py<PyArray2<f64>> {
    let vecs: Vec<Vec<i32>> = trains
        .iter()
        .map(|t| t.as_slice().unwrap().to_vec())
        .collect();
    let refs: Vec<&[i32]> = vecs.iter().map(|v| v.as_slice()).collect();
    let mat = analysis::correlation::covariance_matrix(&refs, bin_size);
    let n = mat.len();
    let rows: Vec<Vec<f64>> = mat.into_iter().collect();
    numpy::PyArray2::from_vec2(py, &rows)
        .unwrap_or_else(|_| numpy::PyArray2::zeros(py, [n, n], false))
        .into()
}

#[pyfunction]
#[pyo3(signature = (binary_train, dt=0.001, max_lag_ms=100.0))]
fn py_autocorrelation_time(
    binary_train: PyReadonlyArray1<'_, i32>,
    dt: f64,
    max_lag_ms: f64,
) -> f64 {
    let data = binary_train.as_slice().unwrap();
    analysis::correlation::autocorrelation_time(data, dt, max_lag_ms)
}

#[pyfunction]
#[pyo3(signature = (trains, bin_size=50))]
fn py_noise_correlation(
    py: Python<'_>,
    trains: Vec<PyReadonlyArray1<'_, i32>>,
    bin_size: usize,
) -> Py<PyArray2<f64>> {
    let vecs: Vec<Vec<i32>> = trains
        .iter()
        .map(|t| t.as_slice().unwrap().to_vec())
        .collect();
    let refs: Vec<&[i32]> = vecs.iter().map(|v| v.as_slice()).collect();
    let mat = analysis::correlation::noise_correlation(&refs, bin_size);
    let rows: Vec<Vec<f64>> = mat.into_iter().collect();
    let n = rows.len();
    numpy::PyArray2::from_vec2(py, &rows)
        .unwrap_or_else(|_| numpy::PyArray2::zeros(py, [n, n], false))
        .into()
}

#[pyfunction]
#[pyo3(signature = (trains, bin_size=50))]
fn py_signal_correlation(
    py: Python<'_>,
    trains: Vec<PyReadonlyArray1<'_, i32>>,
    bin_size: usize,
) -> Py<PyArray2<f64>> {
    let vecs: Vec<Vec<i32>> = trains
        .iter()
        .map(|t| t.as_slice().unwrap().to_vec())
        .collect();
    let refs: Vec<&[i32]> = vecs.iter().map(|v| v.as_slice()).collect();
    let mat = analysis::correlation::signal_correlation(&refs, bin_size);
    let rows: Vec<Vec<f64>> = mat.into_iter().collect();
    let n = rows.len();
    numpy::PyArray2::from_vec2(py, &rows)
        .unwrap_or_else(|_| numpy::PyArray2::zeros(py, [n, n], false))
        .into()
}

#[pyfunction]
#[pyo3(signature = (trains, window=50))]
fn py_spike_count_covariance(
    py: Python<'_>,
    trains: Vec<PyReadonlyArray1<'_, i32>>,
    window: usize,
) -> Py<PyArray2<f64>> {
    let vecs: Vec<Vec<i32>> = trains
        .iter()
        .map(|t| t.as_slice().unwrap().to_vec())
        .collect();
    let refs: Vec<&[i32]> = vecs.iter().map(|v| v.as_slice()).collect();
    let mat = analysis::correlation::spike_count_covariance(&refs, window);
    let rows: Vec<Vec<f64>> = mat.into_iter().collect();
    let n = rows.len();
    numpy::PyArray2::from_vec2(py, &rows)
        .unwrap_or_else(|_| numpy::PyArray2::zeros(py, [n, n], false))
        .into()
}

#[pyfunction]
#[pyo3(signature = (train_a, train_b, bin_size=10))]
fn py_joint_psth(
    py: Python<'_>,
    train_a: PyReadonlyArray1<'_, i32>,
    train_b: PyReadonlyArray1<'_, i32>,
    bin_size: usize,
) -> Py<PyArray2<f64>> {
    let a = train_a.as_slice().unwrap();
    let b = train_b.as_slice().unwrap();
    let (flat, n) = analysis::correlation::joint_psth(a, b, bin_size);
    if n == 0 {
        return numpy::PyArray2::zeros(py, [0, 0], false).into();
    }
    let rows: Vec<Vec<f64>> = flat.chunks(n).map(|c| c.to_vec()).collect();
    numpy::PyArray2::from_vec2(py, &rows).unwrap().into()
}

#[pyfunction]
#[pyo3(signature = (train_a, train_b, dt=0.001, delta_ms=2.0))]
fn py_coincidence_index(
    train_a: PyReadonlyArray1<'_, i32>,
    train_b: PyReadonlyArray1<'_, i32>,
    dt: f64,
    delta_ms: f64,
) -> f64 {
    let a = train_a.as_slice().unwrap();
    let b = train_b.as_slice().unwrap();
    analysis::correlation::coincidence_index(a, b, dt, delta_ms)
}

// ── Distance PyO3 wrappers (P0-A: spike_stats/distance) ─────────

#[pyfunction]
#[pyo3(signature = (train_a, train_b, dt=0.001, tau_ms=10.0))]
fn py_van_rossum_distance(
    train_a: PyReadonlyArray1<'_, i32>,
    train_b: PyReadonlyArray1<'_, i32>,
    dt: f64,
    tau_ms: f64,
) -> f64 {
    let a = train_a.as_slice().unwrap();
    let b = train_b.as_slice().unwrap();
    analysis::distance::van_rossum_distance(a, b, dt, tau_ms)
}

#[pyfunction]
#[pyo3(signature = (times_a, times_b, cost_per_s=1000.0))]
fn py_victor_purpura_distance(
    times_a: PyReadonlyArray1<'_, f64>,
    times_b: PyReadonlyArray1<'_, f64>,
    cost_per_s: f64,
) -> f64 {
    let a = times_a.as_slice().unwrap();
    let b = times_b.as_slice().unwrap();
    analysis::distance::victor_purpura_distance(a, b, cost_per_s)
}

#[pyfunction]
#[pyo3(signature = (train_a, train_b, dt=0.001))]
fn py_isi_distance(
    train_a: PyReadonlyArray1<'_, i32>,
    train_b: PyReadonlyArray1<'_, i32>,
    dt: f64,
) -> f64 {
    let a = train_a.as_slice().unwrap();
    let b = train_b.as_slice().unwrap();
    analysis::distance::isi_distance(a, b, dt)
}

#[pyfunction]
#[pyo3(signature = (times_a, times_b, t_start=0.0, t_end=1.0))]
fn py_spike_distance(
    times_a: PyReadonlyArray1<'_, f64>,
    times_b: PyReadonlyArray1<'_, f64>,
    t_start: f64,
    t_end: f64,
) -> f64 {
    let a = times_a.as_slice().unwrap();
    let b = times_b.as_slice().unwrap();
    analysis::distance::spike_distance(a, b, t_start, t_end)
}

#[pyfunction]
#[pyo3(signature = (times_a, times_b, t_start=0.0, t_end=1.0))]
fn py_spike_sync(
    times_a: PyReadonlyArray1<'_, f64>,
    times_b: PyReadonlyArray1<'_, f64>,
    t_start: f64,
    t_end: f64,
) -> f64 {
    let a = times_a.as_slice().unwrap();
    let b = times_b.as_slice().unwrap();
    analysis::distance::spike_sync(a, b, t_start, t_end)
}

#[pyfunction]
#[pyo3(signature = (times_a, times_b, n_bins=50, t_start=0.0, t_end=1.0))]
fn py_spike_sync_profile(
    py: Python<'_>,
    times_a: PyReadonlyArray1<'_, f64>,
    times_b: PyReadonlyArray1<'_, f64>,
    n_bins: usize,
    t_start: f64,
    t_end: f64,
) -> Py<PyArray1<f64>> {
    let a = times_a.as_slice().unwrap();
    let b = times_b.as_slice().unwrap();
    analysis::distance::spike_sync_profile(a, b, n_bins, t_start, t_end)
        .into_pyarray(py)
        .into()
}

#[pyfunction]
#[pyo3(signature = (times_a, times_b, n_bins=50, t_start=0.0, t_end=1.0))]
fn py_spike_profile(
    py: Python<'_>,
    times_a: PyReadonlyArray1<'_, f64>,
    times_b: PyReadonlyArray1<'_, f64>,
    n_bins: usize,
    t_start: f64,
    t_end: f64,
) -> Py<PyArray1<f64>> {
    let a = times_a.as_slice().unwrap();
    let b = times_b.as_slice().unwrap();
    analysis::distance::spike_profile(a, b, n_bins, t_start, t_end)
        .into_pyarray(py)
        .into()
}

#[pyfunction]
#[pyo3(signature = (binary_train_a, binary_train_b, dt=0.001, n_bins=50))]
fn py_isi_profile(
    py: Python<'_>,
    binary_train_a: PyReadonlyArray1<'_, i32>,
    binary_train_b: PyReadonlyArray1<'_, i32>,
    dt: f64,
    n_bins: usize,
) -> Py<PyArray1<f64>> {
    let a = binary_train_a.as_slice().unwrap();
    let b = binary_train_b.as_slice().unwrap();
    analysis::distance::isi_profile(a, b, dt, n_bins)
        .into_pyarray(py)
        .into()
}

#[pyfunction]
#[pyo3(signature = (times_a, times_b, t_start=0.0, t_end=1.0, cost=0.0))]
fn py_adaptive_spike_distance(
    times_a: PyReadonlyArray1<'_, f64>,
    times_b: PyReadonlyArray1<'_, f64>,
    t_start: f64,
    t_end: f64,
    cost: f64,
) -> f64 {
    let a = times_a.as_slice().unwrap();
    let b = times_b.as_slice().unwrap();
    analysis::distance::adaptive_spike_distance(a, b, t_start, t_end, cost)
}

#[pyfunction]
#[pyo3(signature = (train_a, train_b, dt=0.001, sigma_ms=5.0))]
fn py_schreiber_similarity(
    train_a: PyReadonlyArray1<'_, i32>,
    train_b: PyReadonlyArray1<'_, i32>,
    dt: f64,
    sigma_ms: f64,
) -> f64 {
    let a = train_a.as_slice().unwrap();
    let b = train_b.as_slice().unwrap();
    analysis::distance::schreiber_similarity(a, b, dt, sigma_ms)
}

#[pyfunction]
#[pyo3(signature = (times_a, times_b, dt_max=0.01))]
fn py_hunter_milton_similarity(
    times_a: PyReadonlyArray1<'_, f64>,
    times_b: PyReadonlyArray1<'_, f64>,
    dt_max: f64,
) -> f64 {
    let a = times_a.as_slice().unwrap();
    let b = times_b.as_slice().unwrap();
    analysis::distance::hunter_milton_similarity(a, b, dt_max)
}

#[pyfunction]
#[pyo3(signature = (times_a, times_b, t_start=0.0, t_end=1.0, n_bins=100))]
fn py_earth_movers_distance(
    times_a: PyReadonlyArray1<'_, f64>,
    times_b: PyReadonlyArray1<'_, f64>,
    t_start: f64,
    t_end: f64,
    n_bins: usize,
) -> f64 {
    let a = times_a.as_slice().unwrap();
    let b = times_b.as_slice().unwrap();
    analysis::distance::earth_movers_distance(a, b, t_start, t_end, n_bins)
}

#[pyfunction]
#[pyo3(signature = (spike_times_list, cost_per_s=1000.0))]
fn py_multi_neuron_victor_purpura(
    py: Python<'_>,
    spike_times_list: Vec<PyReadonlyArray1<'_, f64>>,
    cost_per_s: f64,
) -> Py<PyArray2<f64>> {
    let vecs: Vec<Vec<f64>> = spike_times_list
        .iter()
        .map(|t| t.as_slice().unwrap().to_vec())
        .collect();
    let refs: Vec<&[f64]> = vecs.iter().map(|v| v.as_slice()).collect();
    let mat = analysis::distance::multi_neuron_victor_purpura(&refs, cost_per_s);
    let rows: Vec<Vec<f64>> = mat.into_iter().collect();
    let n = rows.len();
    numpy::PyArray2::from_vec2(py, &rows)
        .unwrap_or_else(|_| numpy::PyArray2::zeros(py, [n, n], false))
        .into()
}

#[pyfunction]
#[pyo3(signature = (spike_times_list, metric="spike_distance", t_start=0.0, t_end=1.0))]
fn py_spike_distance_matrix(
    py: Python<'_>,
    spike_times_list: Vec<PyReadonlyArray1<'_, f64>>,
    metric: &str,
    t_start: f64,
    t_end: f64,
) -> Py<PyArray2<f64>> {
    let vecs: Vec<Vec<f64>> = spike_times_list
        .iter()
        .map(|t| t.as_slice().unwrap().to_vec())
        .collect();
    let refs: Vec<&[f64]> = vecs.iter().map(|v| v.as_slice()).collect();
    let mat = analysis::distance::spike_distance_matrix(&refs, metric, t_start, t_end);
    let rows: Vec<Vec<f64>> = mat.into_iter().collect();
    let n = rows.len();
    numpy::PyArray2::from_vec2(py, &rows)
        .unwrap_or_else(|_| numpy::PyArray2::zeros(py, [n, n], false))
        .into()
}

// ── Information PyO3 wrappers (P0-A: spike_stats/information) ────

#[pyfunction]
#[pyo3(signature = (train_a, train_b, bin_size=10))]
fn py_mutual_information(
    train_a: PyReadonlyArray1<'_, i32>,
    train_b: PyReadonlyArray1<'_, i32>,
    bin_size: usize,
) -> f64 {
    let a = train_a.as_slice().unwrap();
    let b = train_b.as_slice().unwrap();
    analysis::information::mutual_information(a, b, bin_size)
}

#[pyfunction]
#[pyo3(signature = (source, target, bin_size=10, lag=1))]
fn py_transfer_entropy(
    source: PyReadonlyArray1<'_, i32>,
    target: PyReadonlyArray1<'_, i32>,
    bin_size: usize,
    lag: usize,
) -> f64 {
    let s = source.as_slice().unwrap();
    let t = target.as_slice().unwrap();
    analysis::information::transfer_entropy(s, t, bin_size, lag)
}

#[pyfunction]
#[pyo3(signature = (binary_train, bin_size=10, word_length=4))]
fn py_spike_train_entropy(
    binary_train: PyReadonlyArray1<'_, i32>,
    bin_size: usize,
    word_length: usize,
) -> f64 {
    let data = binary_train.as_slice().unwrap();
    analysis::information::spike_train_entropy(data, bin_size, word_length)
}

#[pyfunction]
#[pyo3(signature = (binary_train, n_trials=10, bin_size=10, word_length=4))]
fn py_noise_entropy(
    binary_train: PyReadonlyArray1<'_, i32>,
    n_trials: usize,
    bin_size: usize,
    word_length: usize,
) -> f64 {
    let data = binary_train.as_slice().unwrap();
    analysis::information::noise_entropy(data, n_trials, bin_size, word_length)
}

#[pyfunction]
fn py_stimulus_specific_information(
    spike_counts: PyReadonlyArray1<'_, f64>,
    stimulus_ids: PyReadonlyArray1<'_, i64>,
) -> f64 {
    let counts = spike_counts.as_slice().unwrap();
    let ids = stimulus_ids.as_slice().unwrap();
    analysis::information::stimulus_specific_information(counts, ids)
}

#[pyfunction]
#[pyo3(signature = (x, y, k=3))]
fn py_kozachenko_leonenko_mi(
    x: PyReadonlyArray1<'_, f64>,
    y: PyReadonlyArray1<'_, f64>,
    k: usize,
) -> f64 {
    let xd = x.as_slice().unwrap();
    let yd = y.as_slice().unwrap();
    analysis::information::kozachenko_leonenko_mi(xd, yd, k)
}

// ── Causality PyO3 wrappers (P0-A: spike_stats/causality) ───────

#[pyfunction]
#[pyo3(signature = (source, target, bin_size=10, order=5))]
fn py_pairwise_granger_causality(
    source: PyReadonlyArray1<'_, i32>,
    target: PyReadonlyArray1<'_, i32>,
    bin_size: usize,
    order: usize,
) -> f64 {
    let s = source.as_slice().unwrap();
    let t = target.as_slice().unwrap();
    analysis::causality::pairwise_granger_causality(s, t, bin_size, order)
}

#[pyfunction]
#[pyo3(signature = (source, target, condition, bin_size=10, order=5))]
fn py_conditional_granger_causality(
    source: PyReadonlyArray1<'_, i32>,
    target: PyReadonlyArray1<'_, i32>,
    condition: PyReadonlyArray1<'_, i32>,
    bin_size: usize,
    order: usize,
) -> f64 {
    let s = source.as_slice().unwrap();
    let t = target.as_slice().unwrap();
    let c = condition.as_slice().unwrap();
    analysis::causality::conditional_granger_causality(s, t, c, bin_size, order)
}

#[pyfunction]
#[pyo3(signature = (trains, bin_size=10, order=5, n_freqs=64))]
fn py_spectral_granger_causality(
    py: Python<'_>,
    trains: Vec<PyReadonlyArray1<'_, i32>>,
    bin_size: usize,
    order: usize,
    n_freqs: usize,
) -> (Py<PyArray1<f64>>, usize, usize) {
    let vecs: Vec<Vec<i32>> = trains
        .iter()
        .map(|t| t.as_slice().unwrap().to_vec())
        .collect();
    let refs: Vec<&[i32]> = vecs.iter().map(|v| v.as_slice()).collect();
    let (gc, d) = analysis::causality::spectral_granger_causality(&refs, bin_size, order, n_freqs);
    (gc.into_pyarray(py).into(), d, n_freqs)
}

#[pyfunction]
#[pyo3(signature = (trains, bin_size=10, order=5, n_freqs=64))]
fn py_partial_directed_coherence(
    py: Python<'_>,
    trains: Vec<PyReadonlyArray1<'_, i32>>,
    bin_size: usize,
    order: usize,
    n_freqs: usize,
) -> (Py<PyArray1<f64>>, usize, usize) {
    let vecs: Vec<Vec<i32>> = trains
        .iter()
        .map(|t| t.as_slice().unwrap().to_vec())
        .collect();
    let refs: Vec<&[i32]> = vecs.iter().map(|v| v.as_slice()).collect();
    let (pdc, d) = analysis::causality::partial_directed_coherence(&refs, bin_size, order, n_freqs);
    (pdc.into_pyarray(py).into(), d, n_freqs)
}

#[pyfunction]
#[pyo3(signature = (trains, bin_size=10, order=5, n_freqs=64))]
fn py_directed_transfer_function(
    py: Python<'_>,
    trains: Vec<PyReadonlyArray1<'_, i32>>,
    bin_size: usize,
    order: usize,
    n_freqs: usize,
) -> (Py<PyArray1<f64>>, usize, usize) {
    let vecs: Vec<Vec<i32>> = trains
        .iter()
        .map(|t| t.as_slice().unwrap().to_vec())
        .collect();
    let refs: Vec<&[i32]> = vecs.iter().map(|v| v.as_slice()).collect();
    let (dtf, d) = analysis::causality::directed_transfer_function(&refs, bin_size, order, n_freqs);
    (dtf.into_pyarray(py).into(), d, n_freqs)
}

// ── Decoding PyO3 wrappers (P0-A: spike_stats/decoding) ─────────

#[pyfunction]
#[pyo3(signature = (trains, preferred_directions, window=50))]
fn py_population_vector_decode(
    py: Python<'_>,
    trains: Vec<PyReadonlyArray1<'_, i32>>,
    preferred_directions: PyReadonlyArray1<'_, f64>,
    window: usize,
) -> Py<PyArray1<f64>> {
    let vecs: Vec<Vec<i32>> = trains
        .iter()
        .map(|t| t.as_slice().unwrap().to_vec())
        .collect();
    let refs: Vec<&[i32]> = vecs.iter().map(|v| v.as_slice()).collect();
    let dirs = preferred_directions.as_slice().unwrap();
    analysis::decoding::population_vector_decode(&refs, dirs, window)
        .into_pyarray(py)
        .into()
}

#[pyfunction]
#[pyo3(signature = (spike_counts, tuning_rates, n_stimuli, n_neurons, prior=None))]
fn py_bayesian_decode(
    spike_counts: PyReadonlyArray1<'_, f64>,
    tuning_rates: PyReadonlyArray1<'_, f64>,
    n_stimuli: usize,
    n_neurons: usize,
    prior: Option<PyReadonlyArray1<'_, f64>>,
) -> usize {
    let counts = spike_counts.as_slice().unwrap();
    let rates = tuning_rates.as_slice().unwrap();
    let p: Vec<f64> = prior
        .map(|p| p.as_slice().unwrap().to_vec())
        .unwrap_or_default();
    analysis::decoding::bayesian_decode(counts, rates, n_stimuli, n_neurons, &p)
}

#[pyfunction]
fn py_maximum_likelihood_decode(
    spike_counts: PyReadonlyArray1<'_, f64>,
    tuning_rates: PyReadonlyArray1<'_, f64>,
    n_stimuli: usize,
    n_neurons: usize,
) -> usize {
    let counts = spike_counts.as_slice().unwrap();
    let rates = tuning_rates.as_slice().unwrap();
    analysis::decoding::maximum_likelihood_decode(counts, rates, n_stimuli, n_neurons)
}

#[pyfunction]
fn py_linear_discriminant_decode(
    train_data: PyReadonlyArray1<'_, f64>,
    n_samples: usize,
    n_features: usize,
    labels: PyReadonlyArray1<'_, i64>,
    test_point: PyReadonlyArray1<'_, f64>,
) -> i64 {
    let data = train_data.as_slice().unwrap();
    let lbl = labels.as_slice().unwrap();
    let tp = test_point.as_slice().unwrap();
    analysis::decoding::linear_discriminant_decode(data, n_samples, n_features, lbl, tp)
}

#[pyfunction]
fn py_naive_bayes_decode(
    train_data: PyReadonlyArray1<'_, f64>,
    n_samples: usize,
    n_features: usize,
    labels: PyReadonlyArray1<'_, i64>,
    test_point: PyReadonlyArray1<'_, f64>,
) -> i64 {
    let data = train_data.as_slice().unwrap();
    let lbl = labels.as_slice().unwrap();
    let tp = test_point.as_slice().unwrap();
    analysis::decoding::naive_bayes_decode(data, n_samples, n_features, lbl, tp)
}

// ── Neural decoder PyO3 wrappers (P1: neural_decoders) ─────────

#[pyfunction]
#[pyo3(signature = (trains, dt=1.0))]
fn py_tokenise_spikes(
    py: Python<'_>,
    trains: Vec<PyReadonlyArray1<'_, i32>>,
    dt: f64,
) -> (Py<PyArray1<i64>>, Py<PyArray1<f64>>) {
    let vecs: Vec<Vec<i32>> = trains
        .iter()
        .map(|t| t.as_slice().unwrap().to_vec())
        .collect();
    let refs: Vec<&[i32]> = vecs.iter().map(|v| v.as_slice()).collect();
    let tokens = analysis::neural_decoders::tokenise_spikes(&refs, dt);
    let uids: Vec<i64> = tokens.iter().map(|t| t.0 as i64).collect();
    let times: Vec<f64> = tokens.iter().map(|t| t.1).collect();
    (uids.into_pyarray(py).into(), times.into_pyarray(py).into())
}

#[pyfunction]
fn py_sinusoidal_position_encode(
    py: Python<'_>,
    timestamps: PyReadonlyArray1<'_, f64>,
    d_model: usize,
) -> Py<PyArray1<f64>> {
    let ts = timestamps.as_slice().unwrap();
    analysis::neural_decoders::sinusoidal_position_encode(ts, d_model)
        .into_pyarray(py)
        .into()
}

#[pyfunction]
fn py_scaled_dot_product_attention(
    py: Python<'_>,
    queries: PyReadonlyArray1<'_, f64>,
    keys: PyReadonlyArray1<'_, f64>,
    values: PyReadonlyArray1<'_, f64>,
    nq: usize,
    nk: usize,
    d: usize,
) -> Py<PyArray1<f64>> {
    let q = queries.as_slice().unwrap();
    let k = keys.as_slice().unwrap();
    let v = values.as_slice().unwrap();
    analysis::neural_decoders::scaled_dot_product_attention(q, k, v, nq, nk, d)
        .into_pyarray(py)
        .into()
}

#[pyfunction]
fn py_gaussian_attention(
    py: Python<'_>,
    queries: PyReadonlyArray1<'_, f64>,
    keys: PyReadonlyArray1<'_, f64>,
    values: PyReadonlyArray1<'_, f64>,
    nq: usize,
    nk: usize,
    d: usize,
    sigma: f64,
) -> Py<PyArray1<f64>> {
    let q = queries.as_slice().unwrap();
    let k = keys.as_slice().unwrap();
    let v = values.as_slice().unwrap();
    analysis::neural_decoders::gaussian_attention(q, k, v, nq, nk, d, sigma)
        .into_pyarray(py)
        .into()
}

#[pyfunction]
fn py_infonce_loss(
    anchors: PyReadonlyArray1<'_, f64>,
    positives: PyReadonlyArray1<'_, f64>,
    n: usize,
    d: usize,
    temperature: f64,
) -> f64 {
    let a = anchors.as_slice().unwrap();
    let p = positives.as_slice().unwrap();
    analysis::neural_decoders::infonce_loss(a, p, n, d, temperature)
}

// ── Network PyO3 wrappers (P0-A: spike_stats/network) ───────────

#[pyfunction]
#[pyo3(signature = (trains, max_lag_ms=20.0, dt=0.001))]
fn py_functional_connectivity(
    py: Python<'_>,
    trains: Vec<PyReadonlyArray1<'_, i32>>,
    max_lag_ms: f64,
    dt: f64,
) -> Py<PyArray2<f64>> {
    let vecs: Vec<Vec<i32>> = trains
        .iter()
        .map(|t| t.as_slice().unwrap().to_vec())
        .collect();
    let refs: Vec<&[i32]> = vecs.iter().map(|v| v.as_slice()).collect();
    let mat = analysis::network::functional_connectivity(&refs, max_lag_ms, dt);
    let n = refs.len();
    let rows: Vec<Vec<f64>> = mat.chunks(n).map(|c| c.to_vec()).collect();
    numpy::PyArray2::from_vec2(py, &rows)
        .unwrap_or_else(|_| numpy::PyArray2::zeros(py, [n, n], false))
        .into()
}

#[pyfunction]
#[pyo3(signature = (trains, bin_size=5, alpha=0.05))]
fn py_unitary_events(
    py: Python<'_>,
    trains: Vec<PyReadonlyArray1<'_, i32>>,
    bin_size: usize,
    alpha: f64,
) -> Py<PyArray1<i64>> {
    let vecs: Vec<Vec<i32>> = trains
        .iter()
        .map(|t| t.as_slice().unwrap().to_vec())
        .collect();
    let refs: Vec<&[i32]> = vecs.iter().map(|v| v.as_slice()).collect();
    let result = analysis::network::unitary_events(&refs, bin_size, alpha);
    let as_i64: Vec<i64> = result.into_iter().map(|v| v as i64).collect();
    as_i64.into_pyarray(py).into()
}

#[pyfunction]
#[pyo3(signature = (trains, bin_size=5, threshold=2.0))]
fn py_cell_assembly_detection(
    trains: Vec<PyReadonlyArray1<'_, i32>>,
    bin_size: usize,
    threshold: f64,
) -> Vec<Vec<usize>> {
    let vecs: Vec<Vec<i32>> = trains
        .iter()
        .map(|t| t.as_slice().unwrap().to_vec())
        .collect();
    let refs: Vec<&[i32]> = vecs.iter().map(|v| v.as_slice()).collect();
    analysis::network::cell_assembly_detection(&refs, bin_size, threshold)
}

#[pyfunction]
#[pyo3(signature = (trains, dt=0.001, max_delay_ms=20.0, min_chain_length=3))]
fn py_synfire_chain_detection(
    trains: Vec<PyReadonlyArray1<'_, i32>>,
    dt: f64,
    max_delay_ms: f64,
    min_chain_length: usize,
) -> Vec<Vec<usize>> {
    let vecs: Vec<Vec<i32>> = trains
        .iter()
        .map(|t| t.as_slice().unwrap().to_vec())
        .collect();
    let refs: Vec<&[i32]> = vecs.iter().map(|v| v.as_slice()).collect();
    analysis::network::synfire_chain_detection(&refs, dt, max_delay_ms, min_chain_length)
}

// ── Surrogates PyO3 wrappers (P0-A: spike_stats/surrogates) ─────

#[pyfunction]
#[pyo3(signature = (binary_train, seed=0))]
fn py_surrogate_isi_shuffle(
    py: Python<'_>,
    binary_train: PyReadonlyArray1<'_, i32>,
    seed: u64,
) -> Py<PyArray1<i32>> {
    let data = binary_train.as_slice().unwrap();
    analysis::surrogates::surrogate_isi_shuffle(data, seed)
        .into_pyarray(py)
        .into()
}

#[pyfunction]
#[pyo3(signature = (binary_train, dither_ms=5.0, dt=0.001, seed=0))]
fn py_surrogate_dither(
    py: Python<'_>,
    binary_train: PyReadonlyArray1<'_, i32>,
    dither_ms: f64,
    dt: f64,
    seed: u64,
) -> Py<PyArray1<i32>> {
    let data = binary_train.as_slice().unwrap();
    analysis::surrogates::surrogate_dither(data, dither_ms, dt, seed)
        .into_pyarray(py)
        .into()
}

#[pyfunction]
#[pyo3(signature = (rate_hz, duration_s, dt=0.001, seed=0))]
fn py_homogeneous_poisson(
    py: Python<'_>,
    rate_hz: f64,
    duration_s: f64,
    dt: f64,
    seed: u64,
) -> Py<PyArray1<f64>> {
    analysis::surrogates::homogeneous_poisson(rate_hz, duration_s, dt, seed)
        .into_pyarray(py)
        .into()
}

#[pyfunction]
#[pyo3(signature = (rate_hz, shape, duration_s, dt=0.001, seed=0))]
fn py_gamma_process(
    py: Python<'_>,
    rate_hz: f64,
    shape: f64,
    duration_s: f64,
    dt: f64,
    seed: u64,
) -> Py<PyArray1<f64>> {
    analysis::surrogates::gamma_process(rate_hz, shape, duration_s, dt, seed)
        .into_pyarray(py)
        .into()
}

#[pyfunction]
#[pyo3(signature = (rate_hz, burst_mean, duration_s, dt=0.001, seed=0))]
fn py_compound_poisson_process(
    py: Python<'_>,
    rate_hz: f64,
    burst_mean: f64,
    duration_s: f64,
    dt: f64,
    seed: u64,
) -> Py<PyArray1<f64>> {
    analysis::surrogates::compound_poisson_process(rate_hz, burst_mean, duration_s, dt, seed)
        .into_pyarray(py)
        .into()
}

#[pyfunction]
#[pyo3(signature = (binary_train, seed=0))]
fn py_surrogate_joint_isi(
    py: Python<'_>,
    binary_train: PyReadonlyArray1<'_, i32>,
    seed: u64,
) -> Py<PyArray1<i32>> {
    let data = binary_train.as_slice().unwrap();
    analysis::surrogates::surrogate_joint_isi(data, seed)
        .into_pyarray(py)
        .into()
}

#[pyfunction]
#[pyo3(signature = (binary_train, bin_size=10, seed=0))]
fn py_surrogate_bin_shuffling(
    py: Python<'_>,
    binary_train: PyReadonlyArray1<'_, i32>,
    bin_size: usize,
    seed: u64,
) -> Py<PyArray1<i32>> {
    let data = binary_train.as_slice().unwrap();
    analysis::surrogates::surrogate_bin_shuffling(data, bin_size, seed)
        .into_pyarray(py)
        .into()
}

#[pyfunction]
#[pyo3(signature = (binary_train, max_shift=50, seed=0))]
fn py_surrogate_spike_train_shifting(
    py: Python<'_>,
    binary_train: PyReadonlyArray1<'_, i32>,
    max_shift: usize,
    seed: u64,
) -> Py<PyArray1<i32>> {
    let data = binary_train.as_slice().unwrap();
    analysis::surrogates::surrogate_spike_train_shifting(data, max_shift, seed)
        .into_pyarray(py)
        .into()
}

// ── Temporal PyO3 wrappers (P0-A: spike_stats/temporal) ─────────

#[pyfunction]
#[pyo3(signature = (binary_train, dt=0.001, max_isi_ms=10.0, min_spikes=3))]
fn py_burst_detection(
    binary_train: PyReadonlyArray1<'_, i32>,
    dt: f64,
    max_isi_ms: f64,
    min_spikes: usize,
) -> Vec<(f64, f64, usize)> {
    let data = binary_train.as_slice().unwrap();
    analysis::temporal::burst_detection(data, dt, max_isi_ms, min_spikes)
}

#[pyfunction]
#[pyo3(signature = (binary_train, dt=0.001))]
fn py_first_spike_latency(binary_train: PyReadonlyArray1<'_, i32>, dt: f64) -> f64 {
    let data = binary_train.as_slice().unwrap();
    analysis::temporal::first_spike_latency(data, dt)
}

#[pyfunction]
#[pyo3(signature = (binary_train, baseline_steps=100, dt=0.001, threshold_sigma=3.0))]
fn py_response_onset(
    binary_train: PyReadonlyArray1<'_, i32>,
    baseline_steps: usize,
    dt: f64,
    threshold_sigma: f64,
) -> f64 {
    let data = binary_train.as_slice().unwrap();
    analysis::temporal::response_onset(data, baseline_steps, dt, threshold_sigma)
}

#[pyfunction]
#[pyo3(signature = (binary_train, bin_size=50, threshold=3.0))]
fn py_change_point_detection(
    py: Python<'_>,
    binary_train: PyReadonlyArray1<'_, i32>,
    bin_size: usize,
    threshold: f64,
) -> Py<PyArray1<i64>> {
    let data = binary_train.as_slice().unwrap();
    let cps = analysis::temporal::change_point_detection(data, bin_size, threshold);
    let as_i64: Vec<i64> = cps.into_iter().map(|v| v as i64).collect();
    as_i64.into_pyarray(py).into()
}

// ── Patterns PyO3 wrappers (P0-A: spike_stats/patterns) ─────────

#[pyfunction]
#[pyo3(signature = (times_a, times_b, t_start=0.0, t_end=1.0))]
fn py_spike_directionality(
    times_a: PyReadonlyArray1<'_, f64>,
    times_b: PyReadonlyArray1<'_, f64>,
    t_start: f64,
    t_end: f64,
) -> f64 {
    let a = times_a.as_slice().unwrap();
    let b = times_b.as_slice().unwrap();
    analysis::patterns::spike_directionality(a, b, t_start, t_end)
}

#[pyfunction]
#[pyo3(signature = (times_list, t_start=0.0, t_end=1.0))]
fn py_spike_train_order(
    py: Python<'_>,
    times_list: Vec<PyReadonlyArray1<'_, f64>>,
    t_start: f64,
    t_end: f64,
) -> Py<PyArray2<f64>> {
    let vecs: Vec<Vec<f64>> = times_list
        .iter()
        .map(|t| t.as_slice().unwrap().to_vec())
        .collect();
    let refs: Vec<&[f64]> = vecs.iter().map(|v| v.as_slice()).collect();
    let mat = analysis::patterns::spike_train_order(&refs, t_start, t_end);
    let n = refs.len();
    let rows: Vec<Vec<f64>> = mat.chunks(n).map(|c| c.to_vec()).collect();
    numpy::PyArray2::from_vec2(py, &rows)
        .unwrap_or_else(|_| numpy::PyArray2::zeros(py, [n, n], false))
        .into()
}

#[pyfunction]
#[pyo3(signature = (binary_train, dt=0.001, max_lag=20))]
fn py_cubic_higher_order(
    py: Python<'_>,
    binary_train: PyReadonlyArray1<'_, i32>,
    dt: f64,
    max_lag: usize,
) -> Py<PyArray2<f64>> {
    let data = binary_train.as_slice().unwrap();
    let c3 = analysis::patterns::cubic_higher_order(data, dt, max_lag);
    let rows: Vec<Vec<f64>> = c3.chunks(max_lag).map(|c| c.to_vec()).collect();
    numpy::PyArray2::from_vec2(py, &rows)
        .unwrap_or_else(|_| numpy::PyArray2::zeros(py, [max_lag, max_lag], false))
        .into()
}

// ── Spectral PyO3 wrappers (P0-A: spike_stats/spectral) ─────────

#[pyfunction]
#[pyo3(signature = (binary_train, dt=0.001))]
fn py_power_spectrum(
    py: Python<'_>,
    binary_train: PyReadonlyArray1<'_, i32>,
    dt: f64,
) -> (Py<PyArray1<f64>>, Py<PyArray1<f64>>) {
    let data = binary_train.as_slice().unwrap();
    let (psd, freqs) = analysis::spectral::power_spectrum(data, dt);
    (psd.into_pyarray(py).into(), freqs.into_pyarray(py).into())
}

// ── Waveform PyO3 wrappers (P0-A: spike_stats/waveform) ─────────

#[pyfunction]
#[pyo3(signature = (waveform, dt=3.3333333333333335e-05))]
fn py_waveform_width(waveform: PyReadonlyArray1<'_, f64>, dt: f64) -> f64 {
    analysis::waveform::waveform_width(waveform.as_slice().unwrap(), dt)
}

#[pyfunction]
fn py_waveform_amplitude(waveform: PyReadonlyArray1<'_, f64>) -> f64 {
    analysis::waveform::waveform_amplitude(waveform.as_slice().unwrap())
}

#[pyfunction]
#[pyo3(signature = (waveform, dt=3.3333333333333335e-05))]
fn py_waveform_repolarization_slope(waveform: PyReadonlyArray1<'_, f64>, dt: f64) -> f64 {
    analysis::waveform::waveform_repolarization_slope(waveform.as_slice().unwrap(), dt)
}

#[pyfunction]
#[pyo3(signature = (waveform, dt=3.3333333333333335e-05))]
fn py_waveform_recovery_slope(waveform: PyReadonlyArray1<'_, f64>, dt: f64) -> f64 {
    analysis::waveform::waveform_recovery_slope(waveform.as_slice().unwrap(), dt)
}

#[pyfunction]
#[pyo3(signature = (waveform, dt=3.3333333333333335e-05))]
fn py_waveform_halfwidth(waveform: PyReadonlyArray1<'_, f64>, dt: f64) -> f64 {
    analysis::waveform::waveform_halfwidth(waveform.as_slice().unwrap(), dt)
}

#[pyfunction]
fn py_waveform_pt_ratio(waveform: PyReadonlyArray1<'_, f64>) -> f64 {
    analysis::waveform::waveform_pt_ratio(waveform.as_slice().unwrap())
}

// ── Point process PyO3 wrappers (P0-A: spike_stats/point_process) ──

#[pyfunction]
#[pyo3(signature = (binary_train, dt=0.001, window_ms=50.0))]
fn py_conditional_intensity(
    py: Python<'_>,
    binary_train: PyReadonlyArray1<'_, i32>,
    dt: f64,
    window_ms: f64,
) -> Py<PyArray1<f64>> {
    let data = binary_train.as_slice().unwrap();
    analysis::point_process::conditional_intensity(data, dt, window_ms)
        .into_pyarray(py)
        .into()
}

#[pyfunction]
#[pyo3(signature = (binary_train, dt=0.001, bins=30))]
fn py_isi_hazard_function(
    py: Python<'_>,
    binary_train: PyReadonlyArray1<'_, i32>,
    dt: f64,
    bins: usize,
) -> (Py<PyArray1<f64>>, Py<PyArray1<f64>>) {
    let data = binary_train.as_slice().unwrap();
    let (hazard, centres) = analysis::point_process::isi_hazard_function(data, dt, bins);
    (
        hazard.into_pyarray(py).into(),
        centres.into_pyarray(py).into(),
    )
}

#[pyfunction]
#[pyo3(signature = (binary_train, dt=0.001, bins=30))]
fn py_isi_survivor_function(
    py: Python<'_>,
    binary_train: PyReadonlyArray1<'_, i32>,
    dt: f64,
    bins: usize,
) -> (Py<PyArray1<f64>>, Py<PyArray1<f64>>) {
    let data = binary_train.as_slice().unwrap();
    let (surv, centres) = analysis::point_process::isi_survivor_function(data, dt, bins);
    (
        surv.into_pyarray(py).into(),
        centres.into_pyarray(py).into(),
    )
}

#[pyfunction]
#[pyo3(signature = (binary_train, dt=0.001, bins=30))]
fn py_renewal_density(
    py: Python<'_>,
    binary_train: PyReadonlyArray1<'_, i32>,
    dt: f64,
    bins: usize,
) -> (Py<PyArray1<f64>>, Py<PyArray1<f64>>) {
    let data = binary_train.as_slice().unwrap();
    let (dens, centres) = analysis::point_process::renewal_density(data, dt, bins);
    (
        dens.into_pyarray(py).into(),
        centres.into_pyarray(py).into(),
    )
}

// ── Stimulus PyO3 wrappers (P0-A: spike_stats/stimulus) ─────────

#[pyfunction]
#[pyo3(signature = (stimulus, binary_train, window_steps=50))]
fn py_spike_triggered_average(
    py: Python<'_>,
    stimulus: PyReadonlyArray1<'_, f64>,
    binary_train: PyReadonlyArray1<'_, i32>,
    window_steps: usize,
) -> Py<PyArray1<f64>> {
    analysis::stimulus::spike_triggered_average(
        stimulus.as_slice().unwrap(),
        binary_train.as_slice().unwrap(),
        window_steps,
    )
    .into_pyarray(py)
    .into()
}

#[pyfunction]
#[pyo3(signature = (stimulus, binary_train, window_steps=50))]
fn py_spike_triggered_covariance(
    py: Python<'_>,
    stimulus: PyReadonlyArray1<'_, f64>,
    binary_train: PyReadonlyArray1<'_, i32>,
    window_steps: usize,
) -> Py<PyArray2<f64>> {
    let cov = analysis::stimulus::spike_triggered_covariance(
        stimulus.as_slice().unwrap(),
        binary_train.as_slice().unwrap(),
        window_steps,
    );
    let rows: Vec<Vec<f64>> = cov.chunks(window_steps).map(|c| c.to_vec()).collect();
    numpy::PyArray2::from_vec2(py, &rows)
        .unwrap_or_else(|_| numpy::PyArray2::zeros(py, [window_steps, window_steps], false))
        .into()
}

#[pyfunction]
#[pyo3(signature = (binary_train, positions, n_bins=20, dt=0.001))]
fn py_spatial_information(
    binary_train: PyReadonlyArray1<'_, i32>,
    positions: PyReadonlyArray1<'_, f64>,
    n_bins: usize,
    dt: f64,
) -> f64 {
    analysis::stimulus::spatial_information(
        binary_train.as_slice().unwrap(),
        positions.as_slice().unwrap(),
        n_bins,
        dt,
    )
}

#[pyfunction]
#[pyo3(signature = (binary_train, positions, n_bins=50, threshold_std=2.0, dt=0.001))]
fn py_place_field_detection(
    binary_train: PyReadonlyArray1<'_, i32>,
    positions: PyReadonlyArray1<'_, f64>,
    n_bins: usize,
    threshold_std: f64,
    dt: f64,
) -> Vec<(f64, f64)> {
    analysis::stimulus::place_field_detection(
        binary_train.as_slice().unwrap(),
        positions.as_slice().unwrap(),
        n_bins,
        threshold_std,
        dt,
    )
}

#[pyfunction]
#[pyo3(signature = (binary_train, stimulus_values, n_bins=20, dt=0.001))]
fn py_tuning_curve(
    py: Python<'_>,
    binary_train: PyReadonlyArray1<'_, i32>,
    stimulus_values: PyReadonlyArray1<'_, f64>,
    n_bins: usize,
    dt: f64,
) -> (Py<PyArray1<f64>>, Py<PyArray1<f64>>) {
    let (rates, centres) = analysis::stimulus::tuning_curve(
        binary_train.as_slice().unwrap(),
        stimulus_values.as_slice().unwrap(),
        n_bins,
        dt,
    );
    (
        rates.into_pyarray(py).into(),
        centres.into_pyarray(py).into(),
    )
}

// ── LFP PyO3 wrappers (P0-A: spike_stats/lfp) ─────────────────

#[pyfunction]
fn py_phase_locking_value(
    binary_train: PyReadonlyArray1<'_, i32>,
    lfp_signal: PyReadonlyArray1<'_, f64>,
) -> f64 {
    analysis::lfp::phase_locking_value(
        binary_train.as_slice().unwrap(),
        lfp_signal.as_slice().unwrap(),
    )
}

#[pyfunction]
#[pyo3(signature = (binary_train, lfp_signal, dt=0.001))]
fn py_spike_field_coherence(
    py: Python<'_>,
    binary_train: PyReadonlyArray1<'_, i32>,
    lfp_signal: PyReadonlyArray1<'_, f64>,
    dt: f64,
) -> (Py<PyArray1<f64>>, Py<PyArray1<f64>>) {
    let (sfc, freqs) = analysis::lfp::spike_field_coherence(
        binary_train.as_slice().unwrap(),
        lfp_signal.as_slice().unwrap(),
        dt,
    );
    (sfc.into_pyarray(py).into(), freqs.into_pyarray(py).into())
}

#[pyfunction]
#[pyo3(signature = (binary_train, lfp_signal, n_bins=36))]
fn py_spike_phase_histogram(
    py: Python<'_>,
    binary_train: PyReadonlyArray1<'_, i32>,
    lfp_signal: PyReadonlyArray1<'_, f64>,
    n_bins: usize,
) -> (Py<PyArray1<i64>>, Py<PyArray1<f64>>) {
    let (hist, centres) = analysis::lfp::spike_phase_histogram(
        binary_train.as_slice().unwrap(),
        lfp_signal.as_slice().unwrap(),
        n_bins,
    );
    (
        hist.into_pyarray(py).into(),
        centres.into_pyarray(py).into(),
    )
}

// ── Sorting quality PyO3 wrappers (P0-A: spike_stats/sorting_quality)

#[pyfunction]
fn py_isolation_distance(
    cluster: PyReadonlyArray2<'_, f64>,
    noise: PyReadonlyArray2<'_, f64>,
) -> f64 {
    let c_shape = cluster.shape();
    let n_shape = noise.shape();
    let d = c_shape[1];
    let c_data: Vec<f64> = cluster.as_slice().unwrap().to_vec();
    let n_data: Vec<f64> = noise.as_slice().unwrap().to_vec();
    analysis::sorting_quality::isolation_distance(&c_data, c_shape[0], &n_data, n_shape[0], d)
}

#[pyfunction]
fn py_l_ratio(cluster: PyReadonlyArray2<'_, f64>, noise: PyReadonlyArray2<'_, f64>) -> f64 {
    let c_shape = cluster.shape();
    let n_shape = noise.shape();
    let d = c_shape[1];
    let c_data: Vec<f64> = cluster.as_slice().unwrap().to_vec();
    let n_data: Vec<f64> = noise.as_slice().unwrap().to_vec();
    analysis::sorting_quality::l_ratio(&c_data, c_shape[0], &n_data, n_shape[0], d)
}

#[pyfunction]
fn py_silhouette_score(
    features: PyReadonlyArray2<'_, f64>,
    labels: PyReadonlyArray1<'_, i64>,
) -> f64 {
    let shape = features.shape();
    let f_data: Vec<f64> = features.as_slice().unwrap().to_vec();
    let l_data: Vec<i64> = labels.as_slice().unwrap().to_vec();
    analysis::sorting_quality::silhouette_score(&f_data, shape[0], shape[1], &l_data)
}

#[pyfunction]
fn py_d_prime(cluster_a: PyReadonlyArray2<'_, f64>, cluster_b: PyReadonlyArray2<'_, f64>) -> f64 {
    let a_shape = cluster_a.shape();
    let b_shape = cluster_b.shape();
    let d = a_shape[1];
    let a_data: Vec<f64> = cluster_a.as_slice().unwrap().to_vec();
    let b_data: Vec<f64> = cluster_b.as_slice().unwrap().to_vec();
    analysis::sorting_quality::d_prime(&a_data, a_shape[0], &b_data, b_shape[0], d)
}

#[pyfunction]
#[pyo3(signature = (binary_train, dt=0.001, refractory_ms=1.5))]
fn py_isi_violation_rate(
    binary_train: PyReadonlyArray1<'_, i32>,
    dt: f64,
    refractory_ms: f64,
) -> f64 {
    analysis::sorting_quality::isi_violation_rate(
        binary_train.as_slice().unwrap(),
        dt,
        refractory_ms,
    )
}

#[pyfunction]
#[pyo3(signature = (binary_train, n_bins=100))]
fn py_presence_ratio(binary_train: PyReadonlyArray1<'_, i32>, n_bins: usize) -> f64 {
    analysis::sorting_quality::presence_ratio(binary_train.as_slice().unwrap(), n_bins)
}

#[pyfunction]
#[pyo3(signature = (amplitudes, bins=100))]
fn py_amplitude_cutoff(amplitudes: PyReadonlyArray1<'_, f64>, bins: usize) -> f64 {
    analysis::sorting_quality::amplitude_cutoff(amplitudes.as_slice().unwrap(), bins)
}

#[pyfunction]
fn py_snr(waveforms: PyReadonlyArray2<'_, f64>) -> f64 {
    let shape = waveforms.shape();
    let data: Vec<f64> = waveforms.as_slice().unwrap().to_vec();
    analysis::sorting_quality::snr(&data, shape[0], shape[1])
}

#[pyfunction]
#[pyo3(signature = (cluster, noise, k=4))]
fn py_nn_hit_rate(
    cluster: PyReadonlyArray2<'_, f64>,
    noise: PyReadonlyArray2<'_, f64>,
    k: usize,
) -> f64 {
    let c_shape = cluster.shape();
    let n_shape = noise.shape();
    let d = c_shape[1];
    let c_data: Vec<f64> = cluster.as_slice().unwrap().to_vec();
    let n_data: Vec<f64> = noise.as_slice().unwrap().to_vec();
    analysis::sorting_quality::nn_hit_rate(&c_data, c_shape[0], &n_data, n_shape[0], d, k)
}

#[pyfunction]
#[pyo3(signature = (waveforms, timestamps, n_bins=10))]
fn py_drift_metric(
    waveforms: PyReadonlyArray2<'_, f64>,
    timestamps: PyReadonlyArray1<'_, f64>,
    n_bins: usize,
) -> f64 {
    let shape = waveforms.shape();
    let data: Vec<f64> = waveforms.as_slice().unwrap().to_vec();
    let ts: Vec<f64> = timestamps.as_slice().unwrap().to_vec();
    analysis::sorting_quality::drift_metric(&data, shape[0], shape[1], &ts, n_bins)
}

// ── Dimensionality PyO3 wrappers (P0-A: spike_stats/dimensionality)

#[pyfunction]
#[pyo3(signature = (trains, n_components=3, bin_size=10))]
fn py_spike_train_pca(
    py: Python<'_>,
    trains: Vec<PyReadonlyArray1<'_, i32>>,
    n_components: usize,
    bin_size: usize,
) -> (Py<PyArray1<f64>>, Py<PyArray1<f64>>) {
    let vecs: Vec<Vec<i32>> = trains
        .iter()
        .map(|t| t.as_slice().unwrap().to_vec())
        .collect();
    let refs: Vec<&[i32]> = vecs.iter().map(|v| v.as_slice()).collect();
    let (proj, expl) = analysis::dimensionality::spike_train_pca(&refs, n_components, bin_size);
    (proj.into_pyarray(py).into(), expl.into_pyarray(py).into())
}

#[pyfunction]
#[pyo3(signature = (conditions, n_components=3, bin_size=10))]
fn py_demixed_pca(
    py: Python<'_>,
    conditions: Vec<Vec<PyReadonlyArray1<'_, i32>>>,
    n_components: usize,
    bin_size: usize,
) -> (Py<PyArray1<f64>>, Py<PyArray1<f64>>) {
    let vecs: Vec<Vec<Vec<i32>>> = conditions
        .iter()
        .map(|cond| {
            cond.iter()
                .map(|t| t.as_slice().unwrap().to_vec())
                .collect()
        })
        .collect();
    let refs: Vec<Vec<&[i32]>> = vecs
        .iter()
        .map(|cond| cond.iter().map(|v| v.as_slice()).collect())
        .collect();
    let (proj, expl) = analysis::dimensionality::demixed_pca(&refs, n_components, bin_size);
    (proj.into_pyarray(py).into(), expl.into_pyarray(py).into())
}

#[pyfunction]
#[pyo3(signature = (trains, n_factors=3, bin_size=10, n_iter=50))]
fn py_factor_analysis(
    py: Python<'_>,
    trains: Vec<PyReadonlyArray1<'_, i32>>,
    n_factors: usize,
    bin_size: usize,
    n_iter: usize,
) -> (Py<PyArray1<f64>>, Py<PyArray1<f64>>) {
    let vecs: Vec<Vec<i32>> = trains
        .iter()
        .map(|t| t.as_slice().unwrap().to_vec())
        .collect();
    let refs: Vec<&[i32]> = vecs.iter().map(|v| v.as_slice()).collect();
    let (loadings, psi) =
        analysis::dimensionality::factor_analysis(&refs, n_factors, bin_size, n_iter);
    (
        loadings.into_pyarray(py).into(),
        psi.into_pyarray(py).into(),
    )
}

// Matrix-input wrappers: the caller bins and mean-centres once, so every backend
// (NumPy / Rust / Julia / Go / Mojo) shares an identical input matrix and the
// outputs agree to floating-point round-off.

#[pyfunction]
#[pyo3(signature = (mat, n_components=3))]
fn py_pca_components(
    py: Python<'_>,
    mat: PyReadonlyArray2<'_, f64>,
    n_components: usize,
) -> (Py<PyArray1<f64>>, Py<PyArray1<f64>>) {
    let shape = mat.shape();
    let data: Vec<f64> = mat.as_slice().unwrap().to_vec();
    let (proj, expl) =
        analysis::dimensionality::pca_from_centered(&data, shape[0], shape[1], n_components);
    (proj.into_pyarray(py).into(), expl.into_pyarray(py).into())
}

#[pyfunction]
#[pyo3(signature = (mean_mat, n_components=3))]
fn py_demixed_components(
    py: Python<'_>,
    mean_mat: PyReadonlyArray2<'_, f64>,
    n_components: usize,
) -> (Py<PyArray1<f64>>, Py<PyArray1<f64>>) {
    let shape = mean_mat.shape();
    let data: Vec<f64> = mean_mat.as_slice().unwrap().to_vec();
    let (proj, expl) =
        analysis::dimensionality::demixed_from_centered(&data, shape[0], shape[1], n_components);
    (proj.into_pyarray(py).into(), expl.into_pyarray(py).into())
}

#[pyfunction]
#[pyo3(signature = (mat, n_factors=3, n_iter=50))]
fn py_factor_loadings(
    py: Python<'_>,
    mat: PyReadonlyArray2<'_, f64>,
    n_factors: usize,
    n_iter: usize,
) -> (Py<PyArray1<f64>>, Py<PyArray1<f64>>) {
    let shape = mat.shape();
    let data: Vec<f64> = mat.as_slice().unwrap().to_vec();
    let (loadings, psi) =
        analysis::dimensionality::fa_from_centered(&data, shape[0], shape[1], n_factors, n_iter);
    (
        loadings.into_pyarray(py).into(),
        psi.into_pyarray(py).into(),
    )
}

// ── GPFA PyO3 wrappers (P0-A: spike_stats/gpfa) ─────────────────

#[pyfunction]
#[pyo3(signature = (trains, n_latents=3, bin_ms=20.0, dt=0.001, max_iter=50, tol=1e-4, seed=42))]
fn py_gpfa<'py>(
    py: Python<'py>,
    trains: Vec<PyReadonlyArray1<'py, i32>>,
    n_latents: usize,
    bin_ms: f64,
    dt: f64,
    max_iter: usize,
    tol: f64,
    seed: u64,
) -> PyResult<Bound<'py, PyDict>> {
    let vecs: Vec<Vec<i32>> = trains
        .iter()
        .map(|t| t.as_slice().unwrap().to_vec())
        .collect();
    let refs: Vec<&[i32]> = vecs.iter().map(|v| v.as_slice()).collect();
    let result = analysis::gpfa::gpfa(&refs, n_latents, bin_ms, dt, max_iter, tol, seed);

    let dict = PyDict::new(py);
    dict.set_item("trajectories", result.trajectories.into_pyarray(py))?;
    dict.set_item("C", result.c.into_pyarray(py))?;
    dict.set_item("d", result.d.into_pyarray(py))?;
    dict.set_item("R", result.r.into_pyarray(py))?;
    dict.set_item("tau", result.tau.into_pyarray(py))?;
    dict.set_item("log_likelihoods", result.log_likelihoods.into_pyarray(py))?;
    dict.set_item("n_latents", result.n_latents)?;
    dict.set_item("n_bins", result.n_bins)?;
    dict.set_item("n_neurons", result.n_neurons)?;
    Ok(dict)
}

/// Run the GPFA EM loop from a caller-supplied deterministic initialisation.
///
/// Parity contract with `sc_neurocore.analysis.spike_stats.gpfa.gpfa_em`: identical
/// inputs (the PCA init computed once in Python) produce the same trajectories,
/// parameters and exact-marginal log-likelihoods up to floating-point round-off.
#[pyfunction]
#[pyo3(signature = (y, n_neurons, n_bins, c0, d0, r0_diag, tau, n_latents, max_iter, tol))]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn py_gpfa_em<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f64>,
    n_neurons: usize,
    n_bins: usize,
    c0: PyReadonlyArray1<'py, f64>,
    d0: PyReadonlyArray1<'py, f64>,
    r0_diag: PyReadonlyArray1<'py, f64>,
    tau: PyReadonlyArray1<'py, f64>,
    n_latents: usize,
    max_iter: usize,
    tol: f64,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Vec<f64>,
)> {
    let (x_post, c, d, r, log_liks) = analysis::gpfa::gpfa_em_from_init(
        y.as_slice()?,
        c0.as_slice()?,
        d0.as_slice()?,
        r0_diag.as_slice()?,
        tau.as_slice()?,
        n_neurons,
        n_bins,
        n_latents,
        max_iter,
        tol,
    );
    Ok((
        x_post.into_pyarray(py),
        c.into_pyarray(py),
        d.into_pyarray(py),
        r.into_pyarray(py),
        log_liks,
    ))
}

#[pyfunction]
#[pyo3(signature = (new_trains, c, d, r, tau, n_latents, bin_ms=20.0, dt=0.001))]
fn py_gpfa_transform(
    py: Python<'_>,
    new_trains: Vec<PyReadonlyArray1<'_, i32>>,
    c: PyReadonlyArray1<'_, f64>,
    d: PyReadonlyArray1<'_, f64>,
    r: PyReadonlyArray1<'_, f64>,
    tau: PyReadonlyArray1<'_, f64>,
    n_latents: usize,
    bin_ms: f64,
    dt: f64,
) -> Py<PyArray1<f64>> {
    let vecs: Vec<Vec<i32>> = new_trains
        .iter()
        .map(|t| t.as_slice().unwrap().to_vec())
        .collect();
    let refs: Vec<&[i32]> = vecs.iter().map(|v| v.as_slice()).collect();
    let proj = analysis::gpfa::gpfa_transform(
        &refs,
        c.as_slice().unwrap(),
        d.as_slice().unwrap(),
        r.as_slice().unwrap(),
        tau.as_slice().unwrap(),
        n_latents,
        bin_ms,
        dt,
    );
    proj.into_pyarray(py).into()
}

// ── SPADE PyO3 wrappers (P0-A: spike_stats/spade) ─────────────

#[pyfunction]
#[pyo3(signature = (trains, bin_ms=5.0, dt=0.001, min_support=3, max_pattern_size=5, n_surrogates=100, alpha=0.05, seed=42))]
fn py_spade_detect<'py>(
    py: Python<'py>,
    trains: Vec<PyReadonlyArray1<'py, i32>>,
    bin_ms: f64,
    dt: f64,
    min_support: usize,
    max_pattern_size: usize,
    n_surrogates: usize,
    alpha: f64,
    seed: u64,
) -> PyResult<Vec<Bound<'py, PyDict>>> {
    let vecs: Vec<Vec<i32>> = trains
        .iter()
        .map(|t| t.as_slice().unwrap().to_vec())
        .collect();
    let refs: Vec<&[i32]> = vecs.iter().map(|v| v.as_slice()).collect();
    let results = analysis::spade::spade_detect(
        &refs,
        bin_ms,
        dt,
        min_support,
        max_pattern_size,
        n_surrogates,
        alpha,
        seed,
    );
    let mut dicts = Vec::new();
    for pat in results {
        let dict = PyDict::new(py);
        dict.set_item(
            "neurons",
            pat.neurons.iter().map(|&n| n as i64).collect::<Vec<_>>(),
        )?;
        dict.set_item("lags", pat.lags.clone())?;
        dict.set_item("count", pat.count as i64)?;
        dict.set_item("p_value", pat.p_value)?;
        dicts.push(dict);
    }
    Ok(dicts)
}

// ── DNA acceleration PyO3 wrappers ───────────────────────────────────

#[pyfunction]
fn py_dna_design_sequence(_py: Python<'_>, length: usize, seed: u64) -> String {
    String::from_utf8(dna::design_sequence(length, seed)).unwrap_or_default()
}

#[pyfunction]
fn py_dna_design_orthogonal_set(
    _py: Python<'_>,
    count: usize,
    length: usize,
    seed: u64,
) -> Vec<String> {
    dna::design_orthogonal_set(count, length, seed)
        .into_iter()
        .map(|s| String::from_utf8(s).unwrap_or_default())
        .collect()
}

#[pyfunction]
fn py_dna_check_cross_hybridization<'py>(
    py: Python<'py>,
    sequences: Vec<String>,
    threshold: usize,
) -> PyResult<Py<PyAny>> {
    let seqs: Vec<Vec<u8>> = sequences.into_iter().map(|s| s.into_bytes()).collect();
    let flags = dna::check_cross_hybridization(&seqs, threshold);
    let result: Vec<Py<PyAny>> = flags
        .into_iter()
        .map(|(i, j, score)| {
            let d = PyDict::new(py);
            d.set_item("strand_a", i).unwrap();
            d.set_item("strand_b", j).unwrap();
            d.set_item("score", score).unwrap();
            d.into_any().unbind()
        })
        .collect();
    Ok(result.into_pyobject(py)?.into())
}

#[pyfunction]
#[pyo3(signature = (
    gate_types, gate_inputs, gate_outputs, gate_thresholds, gate_leaks,
    input_names, input_concs, duration_s=1800.0, dt=1.0,
    k_hyb=3e5, k_disp=1.0, temperature_c=37.0, use_rk4=true
))]
#[allow(clippy::too_many_arguments)]
fn py_dna_simulate_kinetics<'py>(
    py: Python<'py>,
    gate_types: Vec<String>,
    gate_inputs: Vec<Vec<String>>,
    gate_outputs: Vec<String>,
    gate_thresholds: Vec<f64>,
    gate_leaks: Vec<f64>,
    input_names: Vec<String>,
    input_concs: Vec<f64>,
    duration_s: f64,
    dt: f64,
    k_hyb: f64,
    k_disp: f64,
    temperature_c: f64,
    use_rk4: bool,
) -> PyResult<Py<PyAny>> {
    let gates: Vec<dna::DnaGateSpec> = gate_types
        .iter()
        .zip(gate_inputs.iter())
        .zip(gate_outputs.iter())
        .zip(gate_thresholds.iter())
        .zip(gate_leaks.iter())
        .map(|((((gt, gi), go), th), lk)| {
            let gate_type = match gt.to_uppercase().as_str() {
                "AND" => dna::DnaGateType::And,
                "OR" => dna::DnaGateType::Or,
                "NOT" => dna::DnaGateType::Not,
                "THRESHOLD" => dna::DnaGateType::Threshold,
                "MUX" => dna::DnaGateType::Mux,
                "AMPLIFIER" => dna::DnaGateType::Amplifier,
                "BUFFER" => dna::DnaGateType::Buffer,
                "NAND" => dna::DnaGateType::Nand,
                "XOR" => dna::DnaGateType::Xor,
                _ => dna::DnaGateType::And,
            };
            dna::DnaGateSpec {
                gate_type,
                input_names: gi.clone(),
                output_name: go.clone(),
                threshold: *th,
                leak_rate: *lk,
            }
        })
        .collect();

    let mut inputs = std::collections::HashMap::new();
    for (name, conc) in input_names.into_iter().zip(input_concs) {
        inputs.insert(name, conc);
    }

    let config = dna::KineticConfig {
        k_hyb,
        k_disp,
        temperature_c,
        max_conc: 200.0,
        use_rk4,
    };

    let result = dna::simulate_kinetics(&gates, &inputs, duration_s, dt, &config);

    let dict = PyDict::new(py);
    for (key, trace) in result {
        dict.set_item(key, trace.into_pyarray(py))?;
    }
    Ok(dict.into_any().unbind())
}

#[pyfunction]
#[pyo3(signature = (sequence, min_stem=4, min_loop=3))]
fn py_dna_detect_hairpins(
    _py: Python<'_>,
    sequence: &str,
    min_stem: usize,
    min_loop: usize,
) -> Vec<(usize, usize, usize)> {
    dna::detect_hairpins(sequence.as_bytes(), min_stem, min_loop)
}

// ── Quantum Annealing PyO3 Wrappers ──────────────────────────────────

/// Compute Ising energy for a spin configuration.
///
/// Args:
///     h_indices, h_values: linear biases (parallel arrays)
///     j_i, j_j, j_values: quadratic couplings (parallel arrays)
///     spins: spin configuration (+1/-1)
///     offset: constant energy offset
#[pyfunction]
#[pyo3(signature = (h_indices, h_values, j_i, j_j, j_values, spins, offset=0.0))]
fn py_qa_ising_energy(
    _py: Python<'_>,
    h_indices: Vec<usize>,
    h_values: Vec<f64>,
    j_i: Vec<usize>,
    j_j: Vec<usize>,
    j_values: Vec<f64>,
    spins: Vec<i8>,
    offset: f64,
) -> f64 {
    let h: Vec<(usize, f64)> = h_indices.into_iter().zip(h_values).collect();
    let j: Vec<((usize, usize), f64)> = j_i.into_iter().zip(j_j).zip(j_values).collect();
    quantum::ising_energy(&h, &j, &spins, offset)
}

/// Batch compute Ising energies for many configurations.
#[pyfunction]
#[pyo3(signature = (h_indices, h_values, j_i, j_j, j_values, configs, offset=0.0))]
fn py_qa_batch_ising_energy(
    _py: Python<'_>,
    h_indices: Vec<usize>,
    h_values: Vec<f64>,
    j_i: Vec<usize>,
    j_j: Vec<usize>,
    j_values: Vec<f64>,
    configs: Vec<Vec<i8>>,
    offset: f64,
) -> Vec<f64> {
    let h: Vec<(usize, f64)> = h_indices.into_iter().zip(h_values).collect();
    let j: Vec<((usize, usize), f64)> = j_i.into_iter().zip(j_j).zip(j_values).collect();
    quantum::batch_ising_energy(&h, &j, &configs, offset)
}

/// Run simulated annealing on an Ising model (Rust-accelerated).
///
/// Returns dict with best_spins, best_energy, energies, samples.
#[pyfunction]
#[pyo3(signature = (h_indices, h_values, j_i, j_j, j_values, n_qubits, offset=0.0, n_sweeps=1000, num_reads=10, beta_start=0.1, beta_end=10.0, seed=42))]
fn py_qa_simulated_annealing<'py>(
    py: Python<'py>,
    h_indices: Vec<usize>,
    h_values: Vec<f64>,
    j_i: Vec<usize>,
    j_j: Vec<usize>,
    j_values: Vec<f64>,
    n_qubits: usize,
    offset: f64,
    n_sweeps: usize,
    num_reads: usize,
    beta_start: f64,
    beta_end: f64,
    seed: u64,
) -> PyResult<Py<PyAny>> {
    let h: Vec<(usize, f64)> = h_indices.into_iter().zip(h_values).collect();
    let j: Vec<((usize, usize), f64)> = j_i.into_iter().zip(j_j).zip(j_values).collect();

    let (best_spins, best_energy, all_energies, all_samples) = quantum::simulated_annealing(
        &h, &j, n_qubits, offset, n_sweeps, num_reads, beta_start, beta_end, seed,
    );

    let dict = PyDict::new(py);
    dict.set_item("best_spins", best_spins)?;
    dict.set_item("best_energy", best_energy)?;
    dict.set_item("energies", all_energies)?;
    dict.set_item("samples", all_samples)?;
    dict.set_item("n_sweeps", n_sweeps)?;
    dict.set_item("num_reads", num_reads)?;
    dict.set_item("backend", "rust")?;
    Ok(dict.into_any().unbind())
}

/// Apply gauge transform to Ising biases and couplings.
#[pyfunction]
#[allow(clippy::type_complexity)] // tuple shape mirrors Python's QUBO format
fn py_qa_gauge_transform(
    _py: Python<'_>,
    h_indices: Vec<usize>,
    h_values: Vec<f64>,
    j_i: Vec<usize>,
    j_j: Vec<usize>,
    j_values: Vec<f64>,
    gauge: Vec<i8>,
) -> (Vec<(usize, f64)>, Vec<((usize, usize), f64)>) {
    let h: Vec<(usize, f64)> = h_indices.into_iter().zip(h_values).collect();
    let j: Vec<((usize, usize), f64)> = j_i.into_iter().zip(j_j).zip(j_values).collect();
    quantum::gauge_transform(&h, &j, &gauge)
}

/// Generate random gauge vectors.
#[pyfunction]
#[pyo3(signature = (n_qubits, n_gauges=10, seed=42))]
fn py_qa_generate_gauges(
    _py: Python<'_>,
    n_qubits: usize,
    n_gauges: usize,
    seed: u64,
) -> Vec<Vec<i8>> {
    quantum::generate_gauges(n_qubits, n_gauges, seed)
}

/// Greedy graph partitioning for problem decomposition.
#[pyfunction]
#[pyo3(signature = (n_qubits, j_i, j_j, j_values, max_partition_size=64))]
fn py_qa_greedy_partition(
    _py: Python<'_>,
    n_qubits: usize,
    j_i: Vec<usize>,
    j_j: Vec<usize>,
    j_values: Vec<f64>,
    max_partition_size: usize,
) -> Vec<Vec<usize>> {
    let j: Vec<((usize, usize), f64)> = j_i.into_iter().zip(j_j).zip(j_values).collect();
    quantum::greedy_partition(n_qubits, &j, max_partition_size)
}

// ── Photonic NoC PyO3 Wrappers ───────────────────────────────────────

/// Route waveguides on a mesh topology (Rust-accelerated).
#[pyfunction]
#[pyo3(signature = (adjacency_flat, n, pitch_um=250.0, loss_db_per_cm=2.0))]
fn py_ph_route_waveguides<'py>(
    py: Python<'py>,
    adjacency_flat: Vec<f64>,
    n: usize,
    pitch_um: f64,
    loss_db_per_cm: f64,
) -> PyResult<Py<PyAny>> {
    let result = photonic::route_waveguides(&adjacency_flat, n, pitch_um, loss_db_per_cm);

    let dict = PyDict::new(py);
    let sources: Vec<usize> = result.iter().map(|r| r.source).collect();
    let targets: Vec<usize> = result.iter().map(|r| r.target).collect();
    let lengths: Vec<f64> = result.iter().map(|r| r.length_um).collect();
    let losses: Vec<f64> = result.iter().map(|r| r.loss_db).collect();
    let crossings: Vec<usize> = result.iter().map(|r| r.n_crossings).collect();

    dict.set_item("sources", sources)?;
    dict.set_item("targets", targets)?;
    dict.set_item("lengths_um", lengths)?;
    dict.set_item("losses_db", losses)?;
    dict.set_item("crossings", crossings)?;
    dict.set_item("n_segments", result.len())?;
    dict.set_item("backend", "rust")?;
    Ok(dict.into_any().unbind())
}

/// Compute MZI 2×2 transfer matrix for a given phase.
#[pyfunction]
fn py_ph_mzi_transfer_matrix(_py: Python<'_>, phase_rad: f64) -> Vec<f64> {
    photonic::mzi_transfer_matrix(phase_rad).to_vec()
}

/// Cascade multiple MZI stages via matrix multiplication.
#[pyfunction]
fn py_ph_cascade_mzi(_py: Python<'_>, phases: Vec<f64>) -> Vec<f64> {
    photonic::cascade_mzi(&phases).to_vec()
}

/// Analyze WDM crosstalk (Rust-accelerated).
#[pyfunction]
#[pyo3(signature = (channel_ids, wavelengths, bandwidths, powers, adjacent_xt_db=-25.0))]
fn py_ph_analyze_crosstalk<'py>(
    py: Python<'py>,
    channel_ids: Vec<usize>,
    wavelengths: Vec<f64>,
    bandwidths: Vec<f64>,
    powers: Vec<f64>,
    adjacent_xt_db: f64,
) -> PyResult<Py<PyAny>> {
    let channels: Vec<(usize, f64, f64, f64)> = channel_ids
        .into_iter()
        .zip(wavelengths)
        .zip(bandwidths)
        .zip(powers)
        .map(|(((id, wl), bw), p)| (id, wl, bw, p))
        .collect();

    let result = photonic::analyze_crosstalk(&channels, adjacent_xt_db);

    let dict = PyDict::new(py);
    let ids: Vec<usize> = result.iter().map(|r| r.channel_id).collect();
    let xts: Vec<f64> = result.iter().map(|r| r.crosstalk_db).collect();
    let osnrs: Vec<f64> = result.iter().map(|r| r.osnr_db).collect();
    let adjs: Vec<usize> = result.iter().map(|r| r.n_adjacent).collect();

    dict.set_item("channel_ids", ids)?;
    dict.set_item("crosstalk_db", xts)?;
    dict.set_item("osnr_db", osnrs)?;
    dict.set_item("n_adjacent", adjs)?;
    dict.set_item("backend", "rust")?;
    Ok(dict.into_any().unbind())
}

/// Analyze optical power budget (Rust-accelerated).
#[pyfunction]
#[pyo3(signature = (wg_sources, wg_targets, wg_losses, laser_power_dbm=0.0, detector_sensitivity_dbm=-20.0))]
fn py_ph_analyze_power_budget<'py>(
    py: Python<'py>,
    wg_sources: Vec<usize>,
    wg_targets: Vec<usize>,
    wg_losses: Vec<f64>,
    laser_power_dbm: f64,
    detector_sensitivity_dbm: f64,
) -> PyResult<Py<PyAny>> {
    let wgs: Vec<(usize, usize, f64)> = wg_sources
        .into_iter()
        .zip(wg_targets)
        .zip(wg_losses)
        .map(|((s, t), l)| (s, t, l))
        .collect();

    let result =
        photonic::analyze_power_budget(&wgs, &[], laser_power_dbm, detector_sensitivity_dbm);

    let dict = PyDict::new(py);
    let margins: Vec<f64> = result.iter().map(|r| r.margin_db).collect();
    let passed: Vec<bool> = result.iter().map(|r| r.passed).collect();
    let total_losses: Vec<f64> = result.iter().map(|r| r.total_loss_db).collect();

    dict.set_item("margins_db", margins)?;
    dict.set_item("passed", passed)?;
    dict.set_item("total_losses_db", total_losses)?;
    dict.set_item("n_paths", result.len())?;
    dict.set_item("backend", "rust")?;
    Ok(dict.into_any().unbind())
}

/// Geometric crosstalk analysis for a uniform bank of parallel waveguides.
#[pyfunction]
#[pyo3(signature = (num_waveguides, gap_nm, coupling_length_um, wavelength_nm=1550.0, core_index=3.48, cladding_index=1.45))]
fn py_ph_analyze_crosstalk_bank<'py>(
    py: Python<'py>,
    num_waveguides: usize,
    gap_nm: f64,
    coupling_length_um: f64,
    wavelength_nm: f64,
    core_index: f64,
    cladding_index: f64,
) -> PyResult<Py<PyAny>> {
    let r = photonic::analyze_crosstalk_bank(
        num_waveguides,
        gap_nm,
        coupling_length_um,
        wavelength_nm,
        core_index,
        cladding_index,
    );
    let dict = PyDict::new(py);
    dict.set_item("num_waveguides", r.num_waveguides)?;
    dict.set_item("num_pairs", r.num_near_pairs + r.num_far_pairs)?;
    dict.set_item("num_near_pairs", r.num_near_pairs)?;
    dict.set_item("num_far_pairs", r.num_far_pairs)?;
    dict.set_item("gap_nm", r.gap_nm)?;
    dict.set_item("coupling_length_um", r.coupling_length_um)?;
    dict.set_item("adjacent_coupling_ratio", r.adjacent_coupling_ratio)?;
    dict.set_item("adjacent_isolation_db", r.adjacent_isolation_db)?;
    dict.set_item("next_nearest_coupling_ratio", r.next_nearest_coupling_ratio)?;
    dict.set_item("next_nearest_isolation_db", r.next_nearest_isolation_db)?;
    dict.set_item("worst_isolation_db", r.worst_isolation_db)?;
    dict.set_item("mean_coupling_ratio", r.mean_coupling_ratio)?;
    dict.set_item("max_coupling_ratio", r.max_coupling_ratio)?;
    dict.set_item("crosstalk_safe", r.crosstalk_safe)?;
    dict.set_item("backend", "rust")?;
    Ok(dict.into_any().unbind())
}

/// Per-pair geometric crosstalk for arbitrary waveguide geometry.
/// `pairs_a[i]`, `pairs_b[i]`, `gaps_nm[i]`, `lengths_um[i]` describe pair i.
/// Evaluated in parallel via Rayon — the O(N²) analysis path.
#[pyfunction]
#[pyo3(signature = (pairs_a, pairs_b, gaps_nm, lengths_um, wavelength_nm=1550.0, core_index=3.48, cladding_index=1.45))]
fn py_ph_analyze_crosstalk_pairs<'py>(
    py: Python<'py>,
    pairs_a: Vec<usize>,
    pairs_b: Vec<usize>,
    gaps_nm: Vec<f64>,
    lengths_um: Vec<f64>,
    wavelength_nm: f64,
    core_index: f64,
    cladding_index: f64,
) -> PyResult<Py<PyAny>> {
    let n = pairs_a.len();
    if pairs_b.len() != n || gaps_nm.len() != n || lengths_um.len() != n {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "pairs_a, pairs_b, gaps_nm, lengths_um must be equal length",
        ));
    }
    let pairs: Vec<(usize, usize, f64, f64)> = pairs_a
        .into_iter()
        .zip(pairs_b)
        .zip(gaps_nm)
        .zip(lengths_um)
        .map(|(((a, b), g), l)| (a, b, g, l))
        .collect();
    let results =
        photonic::analyze_crosstalk_pairs(&pairs, wavelength_nm, core_index, cladding_index);

    let dict = PyDict::new(py);
    let idx_a: Vec<usize> = results.iter().map(|r| r.index_a).collect();
    let idx_b: Vec<usize> = results.iter().map(|r| r.index_b).collect();
    let gaps: Vec<f64> = results.iter().map(|r| r.gap_nm).collect();
    let lens: Vec<f64> = results.iter().map(|r| r.coupling_length_um).collect();
    let kappas: Vec<f64> = results
        .iter()
        .map(|r| r.coupling_coefficient_per_um)
        .collect();
    let ratios: Vec<f64> = results.iter().map(|r| r.coupling_ratio).collect();
    let isos: Vec<f64> = results.iter().map(|r| r.isolation_db).collect();

    dict.set_item("pair_a", idx_a)?;
    dict.set_item("pair_b", idx_b)?;
    dict.set_item("gap_nm", gaps)?;
    dict.set_item("coupling_length_um", lens)?;
    dict.set_item("coupling_coefficient_per_um", kappas)?;
    dict.set_item("coupling_ratio", ratios)?;
    dict.set_item("isolation_db", isos)?;
    dict.set_item("num_pairs", n)?;
    dict.set_item("backend", "rust")?;
    Ok(dict.into_any().unbind())
}

// ── SC-Optimizer PyO3 Wrappers ───────────────────────────────────────

/// Run SA design-space search (Rust-accelerated).
///
/// mac_counts: per-layer MAC counts
/// weights: per-layer scoring weights
/// Returns dict with best config indices, score, pareto data
#[pyfunction]
#[pyo3(signature = (mac_counts, weights, max_luts, max_power, max_latency=0, t_init=1.0, t_min=0.001, alpha=0.95, max_iter=2000, seed=42))]
fn py_opt_sa_search<'py>(
    py: Python<'py>,
    mac_counts: Vec<i64>,
    weights: Vec<f64>,
    max_luts: i64,
    max_power: f64,
    max_latency: i64,
    t_init: f64,
    t_min: f64,
    alpha: f64,
    max_iter: usize,
    seed: u64,
) -> PyResult<Py<PyAny>> {
    let candidates: Vec<Vec<optimizer::Candidate>> = mac_counts
        .iter()
        .map(|&mc| optimizer::generate_candidates(mc))
        .collect();

    let result = optimizer::simulated_annealing(
        &candidates,
        &weights,
        max_luts,
        max_power,
        max_latency,
        t_init,
        t_min,
        alpha,
        max_iter,
        seed,
    );

    let dict = PyDict::new(py);
    match result {
        Some(r) => {
            // Extract details before moving best_config
            let mut luts_list = Vec::new();
            let mut power_list = Vec::new();
            let mut acc_list = Vec::new();
            for (i, &idx) in r.best_config.iter().enumerate() {
                let c = &candidates[i][idx];
                luts_list.push(c.luts);
                power_list.push(c.power);
                acc_list.push(c.accuracy);
            }

            dict.set_item("best_config", r.best_config)?;
            dict.set_item("best_score", r.best_score)?;
            dict.set_item("pareto_luts", r.pareto_luts)?;
            dict.set_item("pareto_power", r.pareto_power)?;
            dict.set_item("pareto_score", r.pareto_score)?;
            dict.set_item("feasible", true)?;
            dict.set_item("layer_luts", luts_list)?;
            dict.set_item("layer_power", power_list)?;
            dict.set_item("layer_accuracy", acc_list)?;
        }
        None => {
            dict.set_item("feasible", false)?;
        }
    }
    dict.set_item("backend", "rust")?;
    Ok(dict.into_any().unbind())
}

/// Extract Pareto frontier from (luts, power, score) arrays.
#[pyfunction]
fn py_opt_extract_pareto<'py>(
    py: Python<'py>,
    luts: Vec<i64>,
    power: Vec<f64>,
    score: Vec<f64>,
) -> PyResult<Py<PyAny>> {
    let indices = optimizer::extract_pareto(&luts, &power, &score);
    let dict = PyDict::new(py);
    let p_luts: Vec<i64> = indices.iter().map(|&i| luts[i]).collect();
    let p_power: Vec<f64> = indices.iter().map(|&i| power[i]).collect();
    let p_score: Vec<f64> = indices.iter().map(|&i| score[i]).collect();
    dict.set_item("indices", indices)?;
    dict.set_item("luts", p_luts)?;
    dict.set_item("power", p_power)?;
    dict.set_item("score", p_score)?;
    dict.set_item("backend", "rust")?;
    Ok(dict.into_any().unbind())
}

// ── Evolutionary Substrate PyO3 Wrappers ─────────────────────────────

/// Batch-mutate population weights (Rust-accelerated).
#[pyfunction]
#[pyo3(signature = (genomes, mutation_rate=0.1, mutation_scale=0.1, seed=42))]
fn py_evo_batch_mutate(
    _py: Python<'_>,
    mut genomes: Vec<Vec<f64>>,
    mutation_rate: f64,
    mutation_scale: f64,
    seed: u64,
) -> Vec<Vec<f64>> {
    evo::batch_mutate_weights(&mut genomes, mutation_rate, mutation_scale, seed);
    genomes
}

/// Batch fitness evaluation (Rust-accelerated).
#[pyfunction]
fn py_evo_batch_fitness(
    _py: Python<'_>,
    genomes: Vec<Vec<f64>>,
    inputs: Vec<f64>,
    target: f64,
) -> Vec<f64> {
    evo::batch_evaluate_fitness(&genomes, &inputs, target)
}

/// Batch uniform crossover (Rust-accelerated).
#[pyfunction]
#[pyo3(signature = (parents_a, parents_b, seed=42))]
fn py_evo_batch_crossover(
    _py: Python<'_>,
    parents_a: Vec<Vec<f64>>,
    parents_b: Vec<Vec<f64>>,
    seed: u64,
) -> Vec<Vec<f64>> {
    evo::batch_crossover(&parents_a, &parents_b, seed)
}

/// Population diversity (mean pairwise L2 distance).
#[pyfunction]
fn py_evo_diversity(_py: Python<'_>, genomes: Vec<Vec<f64>>) -> f64 {
    evo::population_diversity(&genomes)
}

/// Novelty scores against archive.
#[pyfunction]
#[pyo3(signature = (genomes, archive, k_nearest=5))]
fn py_evo_novelty(
    _py: Python<'_>,
    genomes: Vec<Vec<f64>>,
    archive: Vec<Vec<f64>>,
    k_nearest: usize,
) -> Vec<f64> {
    evo::novelty_scores(&genomes, &archive, k_nearest)
}

/// Tournament selection.
#[pyfunction]
#[pyo3(signature = (fitness, n_select, tournament_size=3, seed=42))]
fn py_evo_tournament(
    _py: Python<'_>,
    fitness: Vec<f64>,
    n_select: usize,
    tournament_size: usize,
    seed: u64,
) -> Vec<usize> {
    evo::tournament_select(&fitness, n_select, tournament_size, seed)
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

/// Discrete Ollivier-Ricci curvature between two nodes of a coupling graph.
///
/// Parity contract with `sc_neurocore.math.topology.ollivier_ricci_curvature`:
/// for the same `knm` and `(i, j)`, the Rust value agrees with the Python
/// value to within float64 round-off.
///
/// `knm_flat` is the row-major `n x n` coupling matrix. Raises `ValueError`
/// on a malformed shape, a non-finite or negative entry, or an out-of-range
/// index, mirroring the Python validation.
#[pyfunction]
#[pyo3(signature = (knm_flat, n, i, j))]
fn py_ollivier_ricci_curvature(knm_flat: Vec<f64>, n: usize, i: usize, j: usize) -> PyResult<f64> {
    use topology::CurvatureError;
    match topology::ollivier_ricci_curvature(&knm_flat, n, i, j) {
        Ok(value) => Ok(value),
        Err(CurvatureError::BadShape) => Err(PyValueError::new_err(
            "knm must be a square coupling matrix with at least one node",
        )),
        Err(CurvatureError::BadValue) => Err(PyValueError::new_err(
            "knm must contain only finite, non-negative values",
        )),
        Err(CurvatureError::BadIndex) => Err(PyValueError::new_err(
            "node index out of range for coupling graph",
        )),
        Err(CurvatureError::Infeasible) => {
            Err(PyValueError::new_err("transport problem is infeasible"))
        }
    }
}

/// N-step Cazelles-Courbage-Rabinovich (2001) bursting-map simulation.
///
/// Parity contract with `sc_neurocore.neurons.models.cazelles_map.CazellesMapNeuron.simulate`:
/// for the same parameters and constant input the returned `x` trace, spike
/// count, and final `(x, y)` state are bit-identical to the Python reference
/// (the map is exact floating-point arithmetic, no transcendental functions).
#[pyfunction]
#[pyo3(signature = (x0, y0, a, epsilon, sigma, x_threshold, n_steps, current))]
#[allow(clippy::too_many_arguments)]
fn py_cazelles_map_simulate<'py>(
    py: Python<'py>,
    x0: f64,
    y0: f64,
    a: f64,
    epsilon: f64,
    sigma: f64,
    x_threshold: f64,
    n_steps: usize,
    current: f64,
) -> (Bound<'py, PyArray1<f64>>, i64, f64, f64) {
    let mut neuron = crate::neurons::CazellesMapNeuron {
        x: x0,
        y: y0,
        a,
        epsilon,
        sigma,
        x_threshold,
    };
    let (trace, spikes) = neuron.simulate(n_steps, current);
    // Return the trace as a NumPy array directly; marshalling a multi-million
    // element Vec<f64> into a Python list would dominate the wall-clock.
    (trace.into_pyarray(py), spikes, neuron.x, neuron.y)
}

/// N-step Courbage-Nekorkin-Vdovin (2007) discontinuous spiking-map simulation.
///
/// Parity contract with
/// `sc_neurocore.neurons.models.courage_nekorkin_map.CourageNekorkinMapNeuron.simulate`:
/// for the same parameters and constant input the returned `x` trace,
/// upward-crossing spike count, and final `(x, y)` state are bit-identical to
/// the Python reference (the map is exact floating-point arithmetic — additions,
/// multiplications, one division for the breakpoints, and a piecewise/Heaviside
/// branch, no transcendental functions).
#[pyfunction]
#[pyo3(signature = (x0, y0, m0, m1, a, d, j, beta, eps, x_threshold, n_steps, current))]
#[allow(clippy::too_many_arguments)]
fn py_courage_nekorkin_map_simulate<'py>(
    py: Python<'py>,
    x0: f64,
    y0: f64,
    m0: f64,
    m1: f64,
    a: f64,
    d: f64,
    j: f64,
    beta: f64,
    eps: f64,
    x_threshold: f64,
    n_steps: usize,
    current: f64,
) -> (Bound<'py, PyArray1<f64>>, i64, f64, f64) {
    let mut neuron = crate::neurons::CourageNekorkinMapNeuron {
        x: x0,
        y: y0,
        m0,
        m1,
        a,
        d,
        j,
        beta,
        eps,
        x_threshold,
    };
    let (trace, spikes) = neuron.simulate(n_steps, current);
    (trace.into_pyarray(py), spikes, neuron.x, neuron.y)
}

/// Parity contract with `sc_neurocore.neurons.models.rulkov_map.RulkovMapNeuron.simulate`:
/// for the same parameters and constant input the returned `x` trace, upward-
/// crossing spike count, and final `(x, y)` state are bit-identical to the
/// Python reference (the map is exact floating-point arithmetic — one division,
/// additions and multiplications, no transcendental functions).
#[pyfunction]
#[pyo3(signature = (x0, y0, alpha, sigma, mu, x_threshold, n_steps, current))]
#[allow(clippy::too_many_arguments)]
fn py_rulkov_map_simulate<'py>(
    py: Python<'py>,
    x0: f64,
    y0: f64,
    alpha: f64,
    sigma: f64,
    mu: f64,
    x_threshold: f64,
    n_steps: usize,
    current: f64,
) -> (Bound<'py, PyArray1<f64>>, i64, f64, f64) {
    let mut neuron = crate::neurons::RulkovMapNeuron {
        x: x0,
        y: y0,
        alpha,
        sigma,
        mu,
        x_threshold,
    };
    let (trace, spikes) = neuron.simulate(n_steps, current);
    (trace.into_pyarray(py), spikes, neuron.x, neuron.y)
}

/// Parity contract with
/// `sc_neurocore.neurons.models.ibarz_tanaka_map.IbarzTanakaMapNeuron.simulate`:
/// for the same parameters and constant input the returned `x` trace (already
/// reset on spiking steps), spike count, and final `(x, y)` state are
/// bit-identical to the Python reference (the map is exact floating-point
/// arithmetic — one division, additions and multiplications, no transcendental
/// functions).
#[pyfunction]
#[pyo3(signature = (x0, y0, alpha, beta, mu, sigma, x_threshold, x_reset, n_steps, current))]
#[allow(clippy::too_many_arguments)]
fn py_ibarz_tanaka_map_simulate<'py>(
    py: Python<'py>,
    x0: f64,
    y0: f64,
    alpha: f64,
    beta: f64,
    mu: f64,
    sigma: f64,
    x_threshold: f64,
    x_reset: f64,
    n_steps: usize,
    current: f64,
) -> (Bound<'py, PyArray1<f64>>, i64, f64, f64) {
    let mut neuron = crate::neurons::IbarzTanakaMapNeuron {
        x: x0,
        y: y0,
        alpha,
        beta,
        mu,
        sigma,
        x_threshold,
        x_reset,
    };
    let (trace, spikes) = neuron.simulate(n_steps, current);
    (trace.into_pyarray(py), spikes, neuron.x, neuron.y)
}

/// Parity contract with
/// `sc_neurocore.neurons.models.medvedev_map.MedvedevMapNeuron.simulate`: for
/// the same parameters and constant input the returned `x` trace, upward-
/// crossing spike count, and final `x` state are bit-identical to the Python
/// reference (the map is exact floating-point arithmetic — a multiply, an add,
/// and a fold into `[0, 1)`; `f64::rem_euclid(1.0)` equals Python's `x % 1.0`
/// bit-for-bit). This is a one-dimensional map, so there is no `y` state.
#[pyfunction]
#[pyo3(signature = (x0, alpha, beta, x_threshold, n_steps, current))]
fn py_medvedev_map_simulate<'py>(
    py: Python<'py>,
    x0: f64,
    alpha: f64,
    beta: f64,
    x_threshold: f64,
    n_steps: usize,
    current: f64,
) -> (Bound<'py, PyArray1<f64>>, i64, f64) {
    let mut neuron = crate::neurons::MedvedevMapNeuron {
        x: x0,
        alpha,
        beta,
        x_threshold,
    };
    let (trace, spikes) = neuron.simulate(n_steps, current);
    (trace.into_pyarray(py), spikes, neuron.x)
}

/// Parity contract with
/// `sc_neurocore.neurons.models.ermentrout_kopell_map_neuron.ErmentroutKopellMapNeuron.simulate`:
/// for the same parameters and constant input the returned `theta` trace,
/// upward-crossing spike count, and final `theta` state match the Python
/// reference bit-for-bit on a shared libm (the only transcendental is `cos`,
/// and the non-chaotic phase flow does not amplify ULP differences). This is a
/// one-dimensional phase map, so there is no second state.
#[pyfunction]
#[pyo3(signature = (theta0, dt, gain, theta_threshold, n_steps, current))]
fn py_ermentrout_kopell_map_simulate<'py>(
    py: Python<'py>,
    theta0: f64,
    dt: f64,
    gain: f64,
    theta_threshold: f64,
    n_steps: usize,
    current: f64,
) -> (Bound<'py, PyArray1<f64>>, i64, f64) {
    let mut neuron = crate::neurons::ErmentroutKopellMapNeuron {
        theta: theta0,
        dt,
        gain,
        theta_threshold,
    };
    let (trace, spikes) = neuron.simulate(n_steps, current);
    (trace.into_pyarray(py), spikes, neuron.theta)
}

/// N-step McKean (1970) piecewise-linear FitzHugh-Nagumo caricature simulation.
///
/// Parity contract with
/// `sc_neurocore.neurons.models.mckean.McKeanNeuron.simulate`: for the same
/// parameters and constant input the returned `v` trace, upward-`v_peak`-crossing
/// spike count, and final `(v, w)` state are bit-identical to the Python RK4
/// reference (the piecewise-linear right-hand side is exact arithmetic —
/// additions, multiplications and branch selection, no transcendental functions —
/// and a two-dimensional autonomous flow cannot be chaotic).
#[pyfunction]
#[pyo3(signature = (v0, w0, a, epsilon, gamma, dt, v_peak, n_steps, current))]
#[allow(clippy::too_many_arguments)]
fn py_mckean_simulate<'py>(
    py: Python<'py>,
    v0: f64,
    w0: f64,
    a: f64,
    epsilon: f64,
    gamma: f64,
    dt: f64,
    v_peak: f64,
    n_steps: usize,
    current: f64,
) -> (Bound<'py, PyArray1<f64>>, i64, f64, f64) {
    let mut neuron = crate::neurons::McKeanNeuron {
        v: v0,
        w: w0,
        a,
        epsilon,
        gamma,
        dt,
        v_peak,
    };
    let (trace, spikes) = neuron.simulate(n_steps, current);
    (trace.into_pyarray(py), spikes, neuron.v, neuron.w)
}

/// N-step Wilson (1999) polynomial cortical-neuron simulation.
///
/// Parity contract with
/// `sc_neurocore.neurons.models.wilson_hr.WilsonHRNeuron.simulate`: for the same
/// parameters and constant input the returned `v` trace (already hard-reset to
/// `-0.7` on spiking steps), spike count, and final `(v, r)` state are
/// bit-identical to the Python RK4 reference (the right-hand side is exact
/// polynomial arithmetic — no transcendental functions).
#[pyfunction]
#[pyo3(signature = (v0, r0, tau_r, v_peak, dt, n_steps, current))]
#[allow(clippy::too_many_arguments)]
fn py_wilson_hr_simulate<'py>(
    py: Python<'py>,
    v0: f64,
    r0: f64,
    tau_r: f64,
    v_peak: f64,
    dt: f64,
    n_steps: usize,
    current: f64,
) -> (Bound<'py, PyArray1<f64>>, i64, f64, f64) {
    let mut neuron = crate::neurons::WilsonHRNeuron {
        v: v0,
        r: r0,
        tau_r,
        v_peak,
        dt,
    };
    let (trace, spikes) = neuron.simulate(n_steps, current);
    (trace.into_pyarray(py), spikes, neuron.v, neuron.r)
}

/// N-step Pernarowski (1994) pancreatic beta-cell burster simulation.
///
/// Parity contract with
/// `sc_neurocore.neurons.models.pernarowski.PernarowskiNeuron.simulate`: for the
/// same parameters and constant input the returned `v` trace, upward-crossing
/// spike count, and final `(v, w, z)` state are bit-identical to the Python RK4
/// reference (the cubic uses `v.powi(3)` = `v*v*v`, matching the Python `v*v*v`;
/// no transcendental functions).
#[pyfunction]
#[pyo3(signature = (v0, w0, z0, alpha, beta, eps1, eps2, gamma, dt, v_threshold, n_steps, current))]
#[allow(clippy::too_many_arguments)]
fn py_pernarowski_simulate<'py>(
    py: Python<'py>,
    v0: f64,
    w0: f64,
    z0: f64,
    alpha: f64,
    beta: f64,
    eps1: f64,
    eps2: f64,
    gamma: f64,
    dt: f64,
    v_threshold: f64,
    n_steps: usize,
    current: f64,
) -> (Bound<'py, PyArray1<f64>>, i64, f64, f64, f64) {
    let mut neuron = crate::neurons::PernarowskiNeuron {
        v: v0,
        w: w0,
        z: z0,
        alpha,
        beta,
        eps1,
        eps2,
        gamma,
        dt,
        v_threshold,
    };
    let (trace, spikes) = neuron.simulate(n_steps, current);
    (trace.into_pyarray(py), spikes, neuron.v, neuron.w, neuron.z)
}

/// N-step Terman-Wang (LEGION) relaxation-oscillator simulation.
///
/// Parity contract with
/// `sc_neurocore.neurons.models.terman_wang.TermanWangOscillator.simulate`: for
/// the same parameters and constant input the returned `v` trace, upward-crossing
/// spike count, and final `(v, w)` state match the Python RK4 reference. The cubic
/// uses `v.powi(3)` = `v*v*v` (matching the Python `v*v*v`); the `tanh` gating
/// resolves to the same glibc symbol as Python on Linux, so this backend is
/// bit-identical there (Julia/Go/Mojo use their own libm `tanh`, ULP-bounded, and
/// the two-dimensional relaxation oscillator is non-chaotic so it does not amplify).
#[pyfunction]
#[pyo3(signature = (v0, w0, alpha, beta, epsilon, rho, dt, v_peak, n_steps, current))]
#[allow(clippy::too_many_arguments)]
fn py_terman_wang_simulate<'py>(
    py: Python<'py>,
    v0: f64,
    w0: f64,
    alpha: f64,
    beta: f64,
    epsilon: f64,
    rho: f64,
    dt: f64,
    v_peak: f64,
    n_steps: usize,
    current: f64,
) -> (Bound<'py, PyArray1<f64>>, i64, f64, f64) {
    let mut neuron = crate::neurons::TermanWangOscillator {
        v: v0,
        w: w0,
        alpha,
        beta,
        epsilon,
        rho,
        dt,
        v_peak,
    };
    let (trace, spikes) = neuron.simulate(n_steps, current);
    (trace.into_pyarray(py), spikes, neuron.v, neuron.w)
}

/// Parity contract with
/// `sc_neurocore.neurons.models.mihalas_niebur.MihalasNieburNeuron.simulate`:
/// for the same parameters and constant input the returned `v` trace, total
/// spike count, and final `(v, theta, i1, i2)` state are bit-identical to the
/// Python RK4 reference. The Mihalas-Niebur 2009 generalised integrate-and-fire
/// model has a purely linear right-hand side (no transcendental functions), so
/// every RK4 stage is exact arithmetic and the trace reproduces to the last bit
/// across Rust/Julia/Go (Mojo FMA-fuses, validated non-amplifying).
#[pyfunction]
#[pyo3(signature = (
    v0, theta0, i1_0, i2_0, v_rest, v_reset, theta_reset, theta_inf,
    tau_v, tau_theta, tau_1, tau_2, a, b, r1, r2, dt, n_steps, current
))]
#[allow(clippy::too_many_arguments)]
fn py_mihalas_niebur_simulate<'py>(
    py: Python<'py>,
    v0: f64,
    theta0: f64,
    i1_0: f64,
    i2_0: f64,
    v_rest: f64,
    v_reset: f64,
    theta_reset: f64,
    theta_inf: f64,
    tau_v: f64,
    tau_theta: f64,
    tau_1: f64,
    tau_2: f64,
    a: f64,
    b: f64,
    r1: f64,
    r2: f64,
    dt: f64,
    n_steps: usize,
    current: f64,
) -> (Bound<'py, PyArray1<f64>>, i64, f64, f64, f64, f64) {
    let mut neuron = crate::neurons::MihalasNieburNeuron {
        v: v0,
        theta: theta0,
        i1: i1_0,
        i2: i2_0,
        v_rest,
        v_reset,
        theta_reset,
        theta_inf,
        tau_v,
        tau_theta,
        tau_1,
        tau_2,
        a,
        b,
        r1,
        r2,
        dt,
    };
    let (trace, spikes) = neuron.simulate(n_steps, current);
    (
        trace.into_pyarray(py),
        spikes,
        neuron.v,
        neuron.theta,
        neuron.i1,
        neuron.i2,
    )
}

/// Parity contract with
/// `sc_neurocore.neurons.models.glif.GLIFNeuron.simulate`: for the same
/// parameters and constant input the returned `v` trace, total spike count, and
/// final `(v, theta, i_asc1, i_asc2)` state are bit-identical to the Python RK4
/// reference. The Allen GLIF5 model has a purely linear right-hand side (no
/// transcendental functions), so every RK4 stage is exact arithmetic and the
/// trace reproduces to the last bit across Rust/Julia/Go (Mojo FMA-fuses,
/// validated non-amplifying).
#[pyfunction]
#[pyo3(signature = (
    v0, theta0, theta_inf, i_asc1_0, i_asc2_0, v_rest, v_reset, tau_m, tau_theta,
    tau_asc1, tau_asc2, a_theta, delta_theta, r_asc1, r_asc2, resistance, dt,
    n_steps, current
))]
#[allow(clippy::too_many_arguments)]
fn py_glif_simulate<'py>(
    py: Python<'py>,
    v0: f64,
    theta0: f64,
    theta_inf: f64,
    i_asc1_0: f64,
    i_asc2_0: f64,
    v_rest: f64,
    v_reset: f64,
    tau_m: f64,
    tau_theta: f64,
    tau_asc1: f64,
    tau_asc2: f64,
    a_theta: f64,
    delta_theta: f64,
    r_asc1: f64,
    r_asc2: f64,
    resistance: f64,
    dt: f64,
    n_steps: usize,
    current: f64,
) -> (Bound<'py, PyArray1<f64>>, i64, f64, f64, f64, f64) {
    let mut neuron = crate::neurons::GLIFNeuron {
        v: v0,
        theta: theta0,
        theta_inf,
        i_asc1: i_asc1_0,
        i_asc2: i_asc2_0,
        v_rest,
        v_reset,
        tau_m,
        tau_theta,
        tau_asc1,
        tau_asc2,
        a_theta,
        delta_theta,
        r_asc1,
        r_asc2,
        resistance,
        dt,
    };
    let (trace, spikes) = neuron.simulate(n_steps, current);
    (
        trace.into_pyarray(py),
        spikes,
        neuron.v,
        neuron.theta,
        neuron.i_asc1,
        neuron.i_asc2,
    )
}

/// Parity contract with
/// `sc_neurocore.neurons.models.fitzhugh_nagumo.FitzHughNagumoNeuron.simulate`:
/// for the same parameters and constant input the returned `v` trace, upward-
/// crossing spike count, and final `(v, w)` state are bit-identical to the
/// Python RK4 reference (the right-hand side is exact arithmetic — a cube
/// `v.powi(3)` = `v*v*v`, additions and multiplications, no transcendental
/// functions — and a two-dimensional flow cannot be chaotic).
#[pyfunction]
#[pyo3(signature = (v0, w0, a, b, epsilon, dt, v_threshold, n_steps, current))]
#[allow(clippy::too_many_arguments)]
fn py_fitzhugh_nagumo_simulate<'py>(
    py: Python<'py>,
    v0: f64,
    w0: f64,
    a: f64,
    b: f64,
    epsilon: f64,
    dt: f64,
    v_threshold: f64,
    n_steps: usize,
    current: f64,
) -> (Bound<'py, PyArray1<f64>>, i64, f64, f64) {
    let mut neuron = crate::neurons::FitzHughNagumoNeuron {
        v: v0,
        w: w0,
        a,
        b,
        epsilon,
        dt,
        v_threshold,
    };
    let (trace, spikes) = neuron.simulate(n_steps, current);
    (trace.into_pyarray(py), spikes, neuron.v, neuron.w)
}

/// Parity contract with
/// `sc_neurocore.neurons.models.hindmarsh_rose.HindmarshRoseNeuron.simulate`:
/// for the same parameters and constant input the returned `x` trace, upward-
/// crossing spike count, and final `(x, y, z)` state are bit-identical to the
/// Python RK4 reference (the right-hand side is exact arithmetic — `x.powi(3)`
/// = `x*x*x`, `x.powi(2)` = `x*x`, no transcendental functions — so even the
/// chaotic bursting trace reproduces exactly).
#[pyfunction]
#[pyo3(signature = (x0, y0, z0, b, r, s, x_rest, dt, x_threshold, n_steps, current))]
#[allow(clippy::too_many_arguments)]
fn py_hindmarsh_rose_simulate<'py>(
    py: Python<'py>,
    x0: f64,
    y0: f64,
    z0: f64,
    b: f64,
    r: f64,
    s: f64,
    x_rest: f64,
    dt: f64,
    x_threshold: f64,
    n_steps: usize,
    current: f64,
) -> (Bound<'py, PyArray1<f64>>, i64, f64, f64, f64) {
    let mut neuron = crate::neurons::HindmarshRoseNeuron {
        x: x0,
        y: y0,
        z: z0,
        b,
        r,
        s,
        x_rest,
        dt,
        x_threshold,
    };
    let (trace, spikes) = neuron.simulate(n_steps, current);
    (trace.into_pyarray(py), spikes, neuron.x, neuron.y, neuron.z)
}

/// Parity contract with
/// `sc_neurocore.neurons.models.fitzhugh_rinzel.FitzHughRinzelNeuron.simulate`:
/// for the same parameters and constant input the returned `v` trace, upward-
/// crossing spike count, and final `(v, w, y)` state are bit-identical to the
/// Python RK4 reference (the right-hand side is exact arithmetic — `v.powi(3)`
/// = `v*v*v`, additions and multiplications, no transcendental functions).
#[pyfunction]
#[pyo3(signature = (v0, w0, y0, a, b, c, d, delta, mu, dt, v_threshold, n_steps, current))]
#[allow(clippy::too_many_arguments)]
fn py_fitzhugh_rinzel_simulate<'py>(
    py: Python<'py>,
    v0: f64,
    w0: f64,
    y0: f64,
    a: f64,
    b: f64,
    c: f64,
    d: f64,
    delta: f64,
    mu: f64,
    dt: f64,
    v_threshold: f64,
    n_steps: usize,
    current: f64,
) -> (Bound<'py, PyArray1<f64>>, i64, f64, f64, f64) {
    let mut neuron = crate::neurons::FitzHughRinzelNeuron {
        v: v0,
        w: w0,
        y: y0,
        a,
        b,
        c,
        d,
        delta,
        mu,
        dt,
        v_threshold,
    };
    let (trace, spikes) = neuron.simulate(n_steps, current);
    (trace.into_pyarray(py), spikes, neuron.v, neuron.w, neuron.y)
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

// ── Byte-level fault injection (PyO3) ──
//
// Each function consumes a numpy bool/uint8 array (read-only), copies
// into a Vec<u8>, applies the inject kernel in pure Rust, and returns
// (corrupted_array, num_affected). RNG seed is per-call so back-to-back
// invocations are reproducible at the API level.

#[pyfunction]
fn py_inject_bitflip_u8<'py>(
    py: Python<'py>,
    bitstream: PyReadonlyArray1<'_, u8>,
    ber: f64,
    seed: u64,
) -> PyResult<(Py<PyArray1<u8>>, u64)> {
    let mut buf = bitstream.as_slice()?.to_vec();
    let n = fault::inject_bitflip_u8(&mut buf, ber, seed);
    Ok((buf.into_pyarray(py).into(), n))
}

#[pyfunction]
fn py_inject_stuck_at_0_u8<'py>(
    py: Python<'py>,
    bitstream: PyReadonlyArray1<'_, u8>,
    ber: f64,
    seed: u64,
) -> PyResult<(Py<PyArray1<u8>>, u64)> {
    let mut buf = bitstream.as_slice()?.to_vec();
    let n = fault::inject_stuck_at_0_u8(&mut buf, ber, seed);
    Ok((buf.into_pyarray(py).into(), n))
}

#[pyfunction]
fn py_inject_stuck_at_1_u8<'py>(
    py: Python<'py>,
    bitstream: PyReadonlyArray1<'_, u8>,
    ber: f64,
    seed: u64,
) -> PyResult<(Py<PyArray1<u8>>, u64)> {
    let mut buf = bitstream.as_slice()?.to_vec();
    let n = fault::inject_stuck_at_1_u8(&mut buf, ber, seed);
    Ok((buf.into_pyarray(py).into(), n))
}

#[pyfunction]
fn py_inject_dropout_u8<'py>(
    py: Python<'py>,
    bitstream: PyReadonlyArray1<'_, u8>,
    ber: f64,
    seed: u64,
) -> PyResult<(Py<PyArray1<u8>>, u64)> {
    let mut buf = bitstream.as_slice()?.to_vec();
    let n = fault::inject_dropout_u8(&mut buf, ber, seed);
    Ok((buf.into_pyarray(py).into(), n))
}

#[pyfunction]
fn py_inject_gaussian_u8<'py>(
    py: Python<'py>,
    bitstream: PyReadonlyArray1<'_, u8>,
    ber: f64,
    seed: u64,
) -> PyResult<(Py<PyArray1<u8>>, u64)> {
    let mut buf = bitstream.as_slice()?.to_vec();
    let n = fault::inject_gaussian_u8(&mut buf, ber, seed);
    Ok((buf.into_pyarray(py).into(), n))
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
