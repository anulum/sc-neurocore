// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Compute-graph IR PyO3 bindings

//! Python bindings for constructing, verifying, serialising, and emitting SC IR graphs.

use crate::ir;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Register the compute-graph IR classes and functions on the Python extension module.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyScGraph>()?;
    m.add_class::<PyScGraphBuilder>()?;
    m.add_function(wrap_pyfunction!(ir_verify, m)?)?;
    m.add_function(wrap_pyfunction!(ir_print, m)?)?;
    m.add_function(wrap_pyfunction!(ir_parse, m)?)?;
    m.add_function(wrap_pyfunction!(ir_emit_sv, m)?)?;
    Ok(())
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

    /// Add a float-vector constant.
    fn constant_f64_vec(&mut self, values: Vec<f64>, ty: &str) -> PyResult<u32> {
        let sc_type = parse_sc_type(ty)?;
        Ok(self
            .builder_mut()?
            .constant(ir::graph::ScConst::F64Vec(values), sc_type)
            .0)
    }

    /// Add a single Kuramoto integration step over an explicit coupling matrix.
    ///
    /// `phases_id` and `omega_id` are length-`N` vector constants; `coupling_id`
    /// is the row-major `N×N` matrix `K_nm`. `dt` is the Euler step.
    fn kuramoto_step(
        &mut self,
        phases_id: u32,
        omega_id: u32,
        coupling_id: u32,
        dt: f64,
    ) -> PyResult<u32> {
        Ok(self
            .builder_mut()?
            .kuramoto_step(
                ir::graph::ValueId(phases_id),
                ir::graph::ValueId(omega_id),
                ir::graph::ValueId(coupling_id),
                dt,
            )
            .0)
    }

    /// Add a degree-normalised graph aggregation over an explicit adjacency matrix.
    ///
    /// `features_id` is the `n_nodes × n_features` (node-major) constant vector and
    /// `adjacency_id` the row-major `n_nodes × n_nodes` matrix.
    fn graph_forward(
        &mut self,
        features_id: u32,
        adjacency_id: u32,
        n_nodes: usize,
        n_features: usize,
    ) -> PyResult<u32> {
        Ok(self
            .builder_mut()?
            .graph_forward(
                ir::graph::ValueId(features_id),
                ir::graph::ValueId(adjacency_id),
                n_nodes,
                n_features,
            )
            .0)
    }

    /// Add a single-head scaled-dot-product softmax attention op.
    ///
    /// `q_id` is `q_rows × dim_k`, `k_id` is `k_rows × dim_k` and `v_id` is
    /// `k_rows × v_cols` (row-major constant vectors); shapes are inferred from
    /// their lengths and `dim_k` at emit time.
    fn softmax_attention(
        &mut self,
        q_id: u32,
        k_id: u32,
        v_id: u32,
        dim_k: usize,
    ) -> PyResult<u32> {
        Ok(self
            .builder_mut()?
            .softmax_attention(
                ir::graph::ValueId(q_id),
                ir::graph::ValueId(k_id),
                ir::graph::ValueId(v_id),
                dim_k,
            )
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
