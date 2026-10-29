// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — SC Compute Graph data structures

//! SC Compute Graph data structures.

use std::fmt;

// Types

/// Type system for SC IR values.
#[derive(Debug, Clone, PartialEq)]
pub enum ScType {
    /// One bit per clock, observed over a positive window of this many cycles.
    Bitstream { length: usize },
    /// Q-format signed fixed-point. E.g. `FixedPoint { width: 16, frac: 8 }` = Q8.8.
    FixedPoint { width: u32, frac: u32 },
    /// Floating-point probability in [0, 1].
    Rate,
    /// Unsigned integer of a given bit width.
    UInt { width: u32 },
    /// Signed integer of a given bit width.
    SInt { width: u32 },
    /// Boolean (1-bit).
    Bool,
    /// Vector of a base type.
    Vec { element: Box<ScType>, count: usize },
}

impl ScType {
    /// Return the bit width of this type for HDL emission.
    pub fn bit_width(&self) -> usize {
        match self {
            Self::Bool => 1,
            Self::Rate => 16, // mapped to Q8.8
            Self::UInt { width } | Self::SInt { width } => *width as usize,
            Self::FixedPoint { width, .. } => *width as usize,
            Self::Bitstream { .. } => 1, // streaming 1-bit per cycle in current emitter
            Self::Vec { element, count } => element.bit_width() * count,
        }
    }
}

impl fmt::Display for ScType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Bitstream { length } => write!(f, "bitstream<{length}>"),
            Self::FixedPoint { width, frac } => write!(f, "fixed<{width},{frac}>"),
            Self::Rate => write!(f, "rate"),
            Self::UInt { width } => write!(f, "u{width}"),
            Self::SInt { width } => write!(f, "i{width}"),
            Self::Bool => write!(f, "bool"),
            Self::Vec { element, count } => write!(f, "vec<{element},{count}>"),
        }
    }
}

// Value references (SSA-style)

/// Unique identifier for a value produced by an operation.
#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq, Ord, PartialOrd)]
pub struct ValueId(pub u32);

impl fmt::Display for ValueId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "%{}", self.0)
    }
}

/// No further SSA identifier can be allocated without wrapping the graph counter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ValueIdExhausted;

impl fmt::Display for ValueIdExhausted {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("SC IR value ID space exhausted")
    }
}

impl std::error::Error for ValueIdExhausted {}

// Constants

/// Compile-time constant values embedded in the IR.
#[derive(Debug, Clone, PartialEq)]
pub enum ScConst {
    /// Floating-point scalar.
    F64(f64),
    /// Signed integer scalar.
    I64(i64),
    /// Unsigned integer scalar.
    U64(u64),
    /// Flat vector of f64 (for weight matrices).
    F64Vec(Vec<f64>),
    /// Flat vector of i64 (for fixed-point arrays).
    I64Vec(Vec<i64>),
}

// LIF neuron parameters (matches hdl/sc_lif_neuron.v)

/// Parameters for the fixed-point LIF neuron.
/// Maps 1:1 to `sc_lif_neuron` Verilog parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct LifParams {
    pub data_width: u32,
    pub fraction: u32,
    pub v_rest: i64,
    pub v_reset: i64,
    pub v_threshold: i64,
    pub refractory_period: u32,
}

impl Default for LifParams {
    fn default() -> Self {
        Self {
            data_width: 16,
            fraction: 8,
            v_rest: 0,
            v_reset: 0,
            v_threshold: 256, // 1.0 in Q8.8
            refractory_period: 2,
        }
    }
}

// Dense layer parameters

/// Parameters for a dense SC layer.
/// Maps to the row-major `sc_dense_matrix_layer` Verilog module.
/// State and current words have eight fractional bits; probability operands retain their own format.
#[derive(Debug, Clone, PartialEq)]
pub struct DenseParams {
    pub n_inputs: usize,
    pub n_neurons: usize,
    pub data_width: u32,
    /// Bitstream length for SC encoding.
    pub stream_length: usize,
    /// Base LFSR seed for input encoders (per-input stride applied automatically).
    pub input_seed_base: u16,
    /// Base LFSR seed for weight encoders.
    pub weight_seed_base: u16,
    /// Input-to-current mapping: y_min in Q-format.
    pub y_min: i64,
    /// Input-to-current mapping: y_max in Q-format.
    pub y_max: i64,
}

impl Default for DenseParams {
    fn default() -> Self {
        Self {
            n_inputs: 3,
            n_neurons: 7,
            data_width: 16,
            stream_length: 1024,
            input_seed_base: 0xACE1,
            weight_seed_base: 0xBEEF,
            y_min: 0,
            y_max: 256, // 1.0 in Q8.8
        }
    }
}

// DCLS layer parameters

/// Parameters for a delay-coded learnable-spike layer.
/// Maps to `sc_dcls_layer_core` and its parameterized signed fixed-point tent kernel.
#[derive(Debug, Clone, PartialEq)]
pub struct DclsParams {
    /// Number of delayed taps sampled from the input spike stream.
    pub n_taps: usize,
    /// Q-format data width.
    pub data_width: u32,
    /// Q-format fractional bits.
    pub fraction: u32,
    /// Delay-line depth in spike samples.
    pub delay_depth: usize,
    /// Bit width used to address the delay line.
    pub ptr_width: u32,
    /// Per-tap delay offsets, ordered low-to-high in the emitted bus.
    pub tap_offsets: Vec<u32>,
}

impl Default for DclsParams {
    fn default() -> Self {
        Self {
            n_taps: 3,
            data_width: 16,
            fraction: 8,
            delay_depth: 31,
            ptr_width: 5,
            tap_offsets: vec![0, 1, 2],
        }
    }
}

/// Reduce operation mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReduceMode {
    Sum,
    Max,
}

impl fmt::Display for ReduceMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Sum => write!(f, "sum"),
            Self::Max => write!(f, "max"),
        }
    }
}

// Operations

/// A single operation in the SC compute graph.
///
/// Each variant owns a unique `id`; Output, Store and Check are non-value terminators.
/// Input operands reference values produced by earlier operations.
#[derive(Debug, Clone, PartialEq)]
pub enum ScOp {
    /// Effectful combinational simulation check, excluded when SYNTHESIS is defined.
    Check {
        /// Unique terminator identifier; no data word is produced.
        id: ValueId,
        /// Boolean invariant; a false value fails when enabled.
        condition: ValueId,
        /// Optional Boolean enable; None checks unconditionally, including during reset.
        enable: Option<ValueId>,
        /// Closed failure category retained in text and simulation diagnostics.
        kind: super::runtime_check::RuntimeCheckKind,
    },
    /// Current scalar register value, asynchronously reset to an earlier exact-type constant.
    State {
        /// Identifier of the current register word.
        id: ValueId,
        /// Earlier constant whose exact type and raw bits define reset.
        initial: ValueId,
        /// Scalar storage type, including signedness and binary point.
        ty: ScType,
    },
    /// Atomic next-state binding. Its identifier is a terminator, never a numeric value.
    Store {
        /// Unique terminator identifier; it cannot be read as a value.
        id: ValueId,
        /// Register declared by an earlier State operation.
        state: ValueId,
        /// Exact-type value committed simultaneously with every other register.
        next: ValueId,
        /// Optional Boolean control; false holds storage, reset takes priority.
        enable: Option<ValueId>,
    },
    /// Explicitly typed scalar arithmetic and conversion.
    Word {
        /// Unique SSA result identifier.
        id: ValueId,
        /// Scalar instruction and its declared result type.
        op: super::word::WordOp,
    },
    // Data flow
    /// Module input port. No operands; value comes from external I/O.
    Input {
        id: ValueId,
        name: String,
        ty: ScType,
    },

    /// Module output port. Consumes one value; no new value produced.
    /// `id` is a dummy (not referenced by other ops).
    Output {
        id: ValueId,
        name: String,
        source: ValueId,
    },

    /// Compile-time constant.
    Constant {
        id: ValueId,
        value: ScConst,
        ty: ScType,
    },

    // Bitstream primitives
    /// Encode a probability (Rate or FixedPoint) into a Bitstream.
    /// Maps to `sc_bitstream_encoder` in HDL.
    Encode {
        id: ValueId,
        /// Input probability value.
        prob: ValueId,
        /// Bitstream length.
        length: usize,
        /// LFSR seed parameter name (resolved from graph params).
        seed: u16,
    },

    /// Bitwise AND of two bitstreams (stochastic multiply).
    /// Maps to `sc_bitstream_synapse` in HDL.
    BitwiseAnd {
        id: ValueId,
        lhs: ValueId,
        rhs: ValueId,
    },

    /// Count bits in a packed word or a completed temporal stream window.
    /// Temporal results hold between windows and asynchronously reset to zero.
    Popcount { id: ValueId, input: ValueId },

    /// Decode binary observations to a Q8.8 Rate, flooring count * 256 / observations.
    /// Uses the same counting windows as Popcount; division remains integer-valued.
    Decode {
        /// Identifier of the decoded Q8.8 probability result.
        id: ValueId,
        /// Packed word or temporal bitstream whose binary observations are counted.
        input: ValueId,
    },

    // Neuron
    /// Single LIF neuron step.
    /// Maps to `sc_lif_neuron` in HDL.
    LifStep {
        id: ValueId,
        /// Input current (FixedPoint).
        current: ValueId,
        /// Leak coefficient (FixedPoint).
        leak: ValueId,
        /// Input gain coefficient (FixedPoint).
        gain: ValueId,
        /// External noise (FixedPoint, can be zero constant).
        noise: ValueId,
        /// Neuron parameters.
        params: LifParams,
    },

    // Compound operations
    /// Dense SC layer: N_INPUTS → N_NEURONS with full SC pipeline.
    /// Maps to `sc_dense_matrix_layer` in HDL.
    DenseForward {
        id: ValueId,
        /// Input values (`Vec<Rate>` or `Vec<FixedPoint>`).
        inputs: ValueId,
        /// Weight matrix (`Vec<Rate>` or `Vec<FixedPoint>`), row-major [n_neurons × n_inputs].
        weights: ValueId,
        /// Leak coefficient for all neurons.
        leak: ValueId,
        /// Gain coefficient for all neurons.
        gain: ValueId,
        /// Layer parameters.
        params: DenseParams,
    },

    /// Delay-coded learnable-spike layer with a Q8.8 tent kernel.
    /// Maps to `sc_dcls_layer_core` in HDL.
    DclsLayer {
        id: ValueId,
        /// One-bit input spike stream.
        spike: ValueId,
        /// Per-tap Q8.8 weights, packed low-to-high.
        weights: ValueId,
        /// Q8.8 centre of the learnable tent in delay-index units.
        centre: ValueId,
        /// Q8.8 positive tent radius.
        sigma: ValueId,
        /// Layer parameters.
        params: DclsParams,
    },

    // L2: XOR encoding for hyperdimensional computing
    /// Bitwise XOR of two bitstreams (HDC binding).
    BitwiseXor {
        id: ValueId,
        lhs: ValueId,
        rhs: ValueId,
    },

    // L3: Aggregation
    /// Reduce a vector to a scalar (Sum or Max).
    Reduce {
        id: ValueId,
        input: ValueId,
        mode: ReduceMode,
    },

    // L8-L10: Graph message-passing
    /// Graph forward: input features × adjacency → aggregated output.
    GraphForward {
        id: ValueId,
        features: ValueId,
        adjacency: ValueId,
        n_nodes: usize,
        n_features: usize,
    },

    // L7: Softmax attention
    /// Softmax attention: Q·K^T/sqrt(d) → softmax → ·V.
    SoftmaxAttention {
        id: ValueId,
        q: ValueId,
        k: ValueId,
        v: ValueId,
        dim_k: usize,
    },

    // L4: Phase dynamics
    /// Single Kuramoto integration step: dθ/dt = ω + ΣK sin(θ_m - θ_n).
    KuramotoStep {
        id: ValueId,
        phases: ValueId,
        omega: ValueId,
        coupling: ValueId,
        dt: f64,
    },

    // Arithmetic (post-processing)
    /// Scale a value by a constant: output = input * factor.
    Scale {
        id: ValueId,
        input: ValueId,
        factor: f64,
    },

    /// Offset a value by a constant: output = input + offset.
    Offset {
        id: ValueId,
        input: ValueId,
        offset: f64,
    },

    /// Unsigned packed-word division by a positive constant, with a u64 result.
    DivConst {
        id: ValueId,
        input: ValueId,
        divisor: u64,
    },
}

impl ScOp {
    /// Return the unique operation identifier, including non-value terminators.
    pub fn result_id(&self) -> ValueId {
        match self {
            Self::Check { id, .. }
            | Self::State { id, .. }
            | Self::Store { id, .. }
            | Self::Word { id, .. }
            | Self::Input { id, .. }
            | Self::Output { id, .. }
            | Self::Constant { id, .. }
            | Self::Encode { id, .. }
            | Self::BitwiseAnd { id, .. }
            | Self::BitwiseXor { id, .. }
            | Self::Popcount { id, .. }
            | Self::Decode { id, .. }
            | Self::Reduce { id, .. }
            | Self::LifStep { id, .. }
            | Self::DenseForward { id, .. }
            | Self::DclsLayer { id, .. }
            | Self::GraphForward { id, .. }
            | Self::SoftmaxAttention { id, .. }
            | Self::KuramotoStep { id, .. }
            | Self::Scale { id, .. }
            | Self::Offset { id, .. }
            | Self::DivConst { id, .. } => *id,
        }
    }

    /// Return all ValueIds consumed by this operation.
    pub fn operands(&self) -> Vec<ValueId> {
        match self {
            Self::State { initial, .. } => vec![*initial],
            Self::Store {
                state,
                next,
                enable,
                ..
            } => {
                let mut operands = vec![*state, *next];
                operands.extend(*enable);
                operands
            }
            Self::Check {
                condition, enable, ..
            } => {
                let mut operands = vec![*condition];
                operands.extend(*enable);
                operands
            }
            Self::Word { op, .. } => op.operands(),
            Self::Input { .. } | Self::Constant { .. } => vec![],
            Self::Output { source, .. } => vec![*source],
            Self::Encode { prob, .. } => vec![*prob],
            Self::BitwiseAnd { lhs, rhs, .. } | Self::BitwiseXor { lhs, rhs, .. } => {
                vec![*lhs, *rhs]
            }
            Self::Popcount { input, .. }
            | Self::Decode { input, .. }
            | Self::Reduce { input, .. } => vec![*input],
            Self::LifStep {
                current,
                leak,
                gain,
                noise,
                ..
            } => vec![*current, *leak, *gain, *noise],
            Self::DenseForward {
                inputs,
                weights,
                leak,
                gain,
                ..
            } => vec![*inputs, *weights, *leak, *gain],
            Self::DclsLayer {
                spike,
                weights,
                centre,
                sigma,
                ..
            } => vec![*spike, *weights, *centre, *sigma],
            Self::GraphForward {
                features,
                adjacency,
                ..
            } => vec![*features, *adjacency],
            Self::SoftmaxAttention { q, k, v, .. } => vec![*q, *k, *v],
            Self::KuramotoStep {
                phases,
                omega,
                coupling,
                ..
            } => vec![*phases, *omega, *coupling],
            Self::Scale { input, .. }
            | Self::Offset { input, .. }
            | Self::DivConst { input, .. } => {
                vec![*input]
            }
        }
    }

    /// Human-readable operation name for the text format.
    pub fn op_name(&self) -> &'static str {
        match self {
            Self::State { .. } => "sc.state",
            Self::Store { .. } => "sc.store",
            Self::Check { .. } => "sc.check",
            Self::Word { .. } => "sc.word",
            Self::Input { .. } => "sc.input",
            Self::Output { .. } => "sc.output",
            Self::Constant { .. } => "sc.constant",
            Self::Encode { .. } => "sc.encode",
            Self::BitwiseAnd { .. } => "sc.and",
            Self::BitwiseXor { .. } => "sc.xor",
            Self::Popcount { .. } => "sc.popcount",
            Self::Decode { .. } => "sc.decode",
            Self::Reduce { .. } => "sc.reduce",
            Self::LifStep { .. } => "sc.lif_step",
            Self::DenseForward { .. } => "sc.dense_forward",
            Self::DclsLayer { .. } => "sc.dcls_layer",
            Self::GraphForward { .. } => "sc.graph_forward",
            Self::SoftmaxAttention { .. } => "sc.softmax_attention",
            Self::KuramotoStep { .. } => "sc.kuramoto_step",
            Self::Scale { .. } => "sc.scale",
            Self::Offset { .. } => "sc.offset",
            Self::DivConst { .. } => "sc.div_const",
        }
    }
}

// Graph

/// A complete SC compute graph.
///
/// Operations are stored in topological order: every operand
/// referenced by an operation must be defined by an earlier operation.
#[derive(Debug, Clone, PartialEq)]
pub struct ScGraph {
    /// Optional source provenance, bound to this exact graph's canonical text.
    pub source: Option<super::source::SourceIdentity>,
    /// Module name (used as the SV module name during emission).
    pub name: String,
    /// Operations in topological (definition) order.
    pub ops: Vec<ScOp>,
    /// Next available ValueId counter.
    pub(crate) next_id: u32,
}

impl ScGraph {
    /// Create a new empty graph.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            source: None,
            name: name.into(),
            ops: Vec::new(),
            next_id: 0,
        }
    }

    /// Allocate a fresh identifier, retaining the legacy infallible signature.
    ///
    /// # Panics
    ///
    /// Panics before mutation when all allocatable identifiers have been used.
    /// Use [`Self::try_fresh_id`] to handle exhaustion as a normal error.
    pub fn fresh_id(&mut self) -> ValueId {
        self.try_fresh_id().expect("SC IR value ID space exhausted")
    }

    /// Allocate the next SSA identifier without wrapping or changing operations.
    ///
    /// The largest allocatable identifier is `u32::MAX - 1`; `u32::MAX` is the
    /// exhausted-counter sentinel, matching the text parser's accepted range.
    ///
    /// # Errors
    ///
    /// Returns [`ValueIdExhausted`] without changing the graph when the next
    /// identifier would consume the sentinel. Repeated refusals are non-mutating.
    pub fn try_fresh_id(&mut self) -> Result<ValueId, ValueIdExhausted> {
        let next = self.next_id.checked_add(1).ok_or(ValueIdExhausted)?;
        let id = ValueId(self.next_id);
        self.next_id = next;
        Ok(id)
    }

    /// Append an operation and return its result ValueId.
    pub fn push(&mut self, op: ScOp) -> ValueId {
        let id = op.result_id();
        self.ops.push(op);
        id
    }

    /// Number of operations.
    pub fn len(&self) -> usize {
        self.ops.len()
    }

    /// Whether the graph is empty.
    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }

    /// Collect all Input operations.
    pub fn inputs(&self) -> Vec<&ScOp> {
        self.ops
            .iter()
            .filter(|op| matches!(op, ScOp::Input { .. }))
            .collect()
    }

    /// Collect all Output operations.
    pub fn outputs(&self) -> Vec<&ScOp> {
        self.ops
            .iter()
            .filter(|op| matches!(op, ScOp::Output { .. }))
            .collect()
    }
}
