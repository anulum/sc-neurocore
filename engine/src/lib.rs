// CopyRight: (c) 1998-2026 Miroslav Sotek. All rights reserved.
// Contact us: www.anulum.li  protoscience@anulum.li
// ORCID: https://orcid.org/0009-0009-3560-0851
// License: GNU AFFERO GENERAL PUBLIC LICENSE v3
// Commercial Licensing: Available

#![allow(clippy::useless_conversion)]

use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2,
    PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};

pub mod attention;
pub mod bitstream;
pub mod encoder;
pub mod grad;
pub mod graph;
pub mod ir;
pub mod layer;
pub mod neuron;
pub mod scpn;
pub mod simd;

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
    fn from_packed(data: Vec<u64>, length: usize) -> Self {
        Self {
            inner: bitstream::BitStreamTensor::from_words(data, length),
        }
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

/// SC-NeuroCore v3.7 — High-Performance Rust Engine
#[pymodule]
fn sc_neurocore_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", "3.7.0")?;
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
    m.add_class::<Lfsr16>()?;
    m.add_class::<BitstreamEncoder>()?;
    m.add_class::<FixedPointLif>()?;
    m.add_class::<DenseLayer>()?;
    m.add_class::<PySurrogateLif>()?;
    m.add_class::<PyDifferentiableDenseLayer>()?;
    m.add_class::<PyStochasticAttention>()?;
    m.add_class::<PyStochasticGraphLayer>()?;
    m.add_class::<PyKuramotoSolver>()?;
    m.add_class::<PySCPNMetrics>()?;
    m.add_class::<PyBitStreamTensor>()?;
    m.add_class::<PyScGraph>()?;
    m.add_class::<PyScGraphBuilder>()?;
    m.add_function(wrap_pyfunction!(ir_verify, m)?)?;
    m.add_function(wrap_pyfunction!(ir_print, m)?)?;
    m.add_function(wrap_pyfunction!(ir_parse, m)?)?;
    m.add_function(wrap_pyfunction!(ir_emit_sv, m)?)?;
    Ok(())
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
        return Ok(packed_rows.into_py(py));
    }

    let flat = bits
        .extract::<Vec<u8>>()
        .map_err(|_| PyValueError::new_err("Expected a 1-D or 2-D array of uint8 bits."))?;
    Ok(bitstream::pack(&flat).data.into_py(py))
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
        } else if batch == 0 {
            0
        } else {
            original_length / batch
        };

        let unpacked_rows: Vec<Vec<u8>> = rows
            .into_iter()
            .map(|row| {
                bitstream::unpack(&bitstream::BitStreamTensor::from_words(row, per_batch_len))
            })
            .collect();
        return Ok(unpacked_rows.into_py(py));
    }

    let words = packed.extract::<Vec<u64>>().map_err(|_| {
        PyValueError::new_err("Expected packed uint64 words as 1-D or 2-D sequence.")
    })?;
    let tensor = bitstream::BitStreamTensor::from_words(words, original_length);
    Ok(bitstream::unpack(&tensor).into_py(py))
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
    Ok(tensor.data.into_pyarray_bound(py))
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
    Ok(bits.into_pyarray_bound(py))
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
    let spikes_arr = PyArray1::<i32>::zeros_bound(py, n_steps, false);
    let voltages_arr = PyArray1::<i16>::zeros_bound(py, n_steps, false);

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

    let spikes_arr = PyArray2::<i32>::zeros_bound(py, [n_neurons, n_steps], false);
    let voltages_arr = PyArray2::<i16>::zeros_bound(py, [n_neurons, n_steps], false);

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
    let spikes_arr = PyArray1::<i32>::zeros_bound(py, n_steps, false);
    let voltages_arr = PyArray1::<i16>::zeros_bound(py, n_steps, false);

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
    Ok(arr.into_pyarray_bound(py))
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
        let dict = PyDict::new_bound(py);
        dict.set_item("v", self.inner.v)?;
        dict.set_item("refractory_counter", self.inner.refractory_counter)?;
        Ok(dict.into_any().unbind())
    }
}

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
            .forward_fused(&input_values, seed)
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
        Ok(out.into_pyarray_bound(py))
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
        let out = PyArray2::<f64>::zeros_bound(py, [n_samples, self.inner.n_neurons], false);
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
        Ok(out.into_pyarray_bound(py))
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

    fn backward(&mut self, grad_output: f32) -> f32 {
        self.inner.backward(grad_output)
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
    fn new(dim_k: usize) -> Self {
        Self {
            inner: attention::StochasticAttention::new(dim_k),
        }
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

#[pymethods]
impl PyKuramotoSolver {
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

        Ok(Self {
            inner: scpn::KuramotoSolver::new(omega, coupling_flat, phases, noise_amp),
        })
    }

    #[pyo3(signature = (dt, seed=0))]
    fn step(&mut self, dt: f64, seed: u64) -> f64 {
        self.inner.step(dt, seed)
    }

    #[pyo3(signature = (n_steps, dt, seed=0))]
    fn run(&mut self, n_steps: usize, dt: f64, seed: u64) -> Vec<f64> {
        self.inner.run(n_steps, dt, seed)
    }

    fn set_field_pressure(&mut self, f: f64) {
        self.inner.set_field_pressure(f);
    }

    #[pyo3(signature = (
        dt,
        seed=0,
        w_flat=vec![],
        sigma_g=0.0,
        h_flat=vec![],
        pgbo_weight=0.0,
    ))]
    fn step_ssgf(
        &mut self,
        dt: f64,
        seed: u64,
        w_flat: Vec<f64>,
        sigma_g: f64,
        h_flat: Vec<f64>,
        pgbo_weight: f64,
    ) -> f64 {
        self.inner
            .step_ssgf(dt, seed, &w_flat, sigma_g, &h_flat, pgbo_weight)
    }

    #[pyo3(signature = (
        n_steps,
        dt,
        seed=0,
        w_flat=vec![],
        sigma_g=0.0,
        h_flat=vec![],
        pgbo_weight=0.0,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn run_ssgf(
        &mut self,
        n_steps: usize,
        dt: f64,
        seed: u64,
        w_flat: Vec<f64>,
        sigma_g: f64,
        h_flat: Vec<f64>,
        pgbo_weight: f64,
    ) -> Vec<f64> {
        self.inner
            .run_ssgf(n_steps, dt, seed, &w_flat, sigma_g, &h_flat, pgbo_weight)
    }

    fn order_parameter(&self) -> f64 {
        self.inner.order_parameter()
    }

    fn get_phases(&self) -> Vec<f64> {
        self.inner.get_phases().to_vec()
    }

    fn set_phases(&mut self, phases: Vec<f64>) -> PyResult<()> {
        if phases.len() != self.inner.n {
            return Err(PyValueError::new_err(format!(
                "phases length mismatch: got {}, expected {}.",
                phases.len(),
                self.inner.n
            )));
        }
        self.inner.set_phases(phases);
        Ok(())
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
fn ir_emit_sv(graph: PyRef<'_, PyScGraph>) -> String {
    ir::emit_sv::emit(&graph.inner)
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

