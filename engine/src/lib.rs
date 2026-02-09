use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};

pub mod bitstream;
pub mod encoder;
pub mod layer;
pub mod neuron;
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
        let packed_rows: Vec<Vec<u64>> = rows
            .iter()
            .map(|row| bitstream::pack(row).data)
            .collect();
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
            .map(|row| bitstream::unpack(&bitstream::BitStreamTensor::from_words(row, per_batch_len)))
            .collect();
        return Ok(unpacked_rows.into_py(py));
    }

    let words = packed
        .extract::<Vec<u64>>()
        .map_err(|_| PyValueError::new_err("Expected packed uint64 words as 1-D or 2-D sequence."))?;
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

    let words = packed
        .extract::<Vec<u64>>()
        .map_err(|_| PyValueError::new_err("Expected packed uint64 words as 1-D or 2-D sequence."))?;
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
        self.inner.set_weights(weights).map_err(PyValueError::new_err)
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
