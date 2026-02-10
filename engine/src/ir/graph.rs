//! SC Compute Graph data structures.

use std::fmt;

// Types

/// Type system for SC IR values.
#[derive(Debug, Clone, PartialEq)]
pub enum ScType {
    /// Packed u64 bitstream of a given length.
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
/// Maps to `sc_dense_layer_core` Verilog module.
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

// Operations

/// A single operation in the SC compute graph.
///
/// Each variant produces exactly one value identified by `id`.
/// Input operands reference values produced by earlier operations.
#[derive(Debug, Clone, PartialEq)]
pub enum ScOp {
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

    /// Population count: count 1-bits in a bitstream.
    /// Part of `sc_dotproduct_to_current` in HDL.
    Popcount { id: ValueId, input: ValueId },

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
    /// Maps to `sc_dense_layer_core` in HDL.
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

    /// Integer division by a constant (for rate computation).
    DivConst {
        id: ValueId,
        input: ValueId,
        divisor: u64,
    },
}

impl ScOp {
    /// Return the ValueId produced by this operation.
    pub fn result_id(&self) -> ValueId {
        match self {
            Self::Input { id, .. }
            | Self::Output { id, .. }
            | Self::Constant { id, .. }
            | Self::Encode { id, .. }
            | Self::BitwiseAnd { id, .. }
            | Self::Popcount { id, .. }
            | Self::LifStep { id, .. }
            | Self::DenseForward { id, .. }
            | Self::Scale { id, .. }
            | Self::Offset { id, .. }
            | Self::DivConst { id, .. } => *id,
        }
    }

    /// Return all ValueIds consumed by this operation.
    pub fn operands(&self) -> Vec<ValueId> {
        match self {
            Self::Input { .. } | Self::Constant { .. } => vec![],
            Self::Output { source, .. } => vec![*source],
            Self::Encode { prob, .. } => vec![*prob],
            Self::BitwiseAnd { lhs, rhs, .. } => vec![*lhs, *rhs],
            Self::Popcount { input, .. } => vec![*input],
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
            Self::Input { .. } => "sc.input",
            Self::Output { .. } => "sc.output",
            Self::Constant { .. } => "sc.constant",
            Self::Encode { .. } => "sc.encode",
            Self::BitwiseAnd { .. } => "sc.and",
            Self::Popcount { .. } => "sc.popcount",
            Self::LifStep { .. } => "sc.lif_step",
            Self::DenseForward { .. } => "sc.dense_forward",
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
            name: name.into(),
            ops: Vec::new(),
            next_id: 0,
        }
    }

    /// Allocate a fresh ValueId.
    pub fn fresh_id(&mut self) -> ValueId {
        let id = ValueId(self.next_id);
        self.next_id += 1;
        id
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
