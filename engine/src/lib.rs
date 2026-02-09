#![allow(clippy::useless_conversion)]

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};

pub mod attention;
pub mod bitstream;
pub mod encoder;
pub mod grad;
pub mod graph;
pub mod layer;
pub mod neuron;
pub mod scpn;
pub mod simd;

/// SC-NeuroCore v3.0 — High-Performance Rust Engine
#[pymodule]
fn sc_neurocore_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", "3.0.0-alpha.1")?;
    m.add_function(wrap_pyfunction!(simd_tier, m)?)?;
    m.add_function(wrap_pyfunction!(pack_bitstream, m)?)?;
    m.add_function(wrap_pyfunction!(unpack_bitstream, m)?)?;
    m.add_function(wrap_pyfunction!(popcount, m)?)?;
    m.add_class::<Lfsr16>()?;
    m.add_class::<BitstreamEncoder>()?;
    m.add_class::<FixedPointLif>()?;
    m.add_class::<DenseLayer>()?;
    m.add_class::<PySurrogateLif>()?;
    m.add_class::<PyDifferentiableDenseLayer>()?;
    m.add_class::<PyStochasticAttention>()?;
    m.add_class::<PyStochasticGraphLayer>()?;
    m.add_class::<PyKuramotoSolver>()?;
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
