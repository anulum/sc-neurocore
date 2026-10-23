// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Explicit scalar word operations

//! Raw integer operations; scaling, rounding, saturation and clocked delays are explicit.

use super::graph::{ScType, ValueId};
use std::{fmt, str::FromStr};

macro_rules! operation_names {
    ($name:ident { $($variant:ident => $text:literal),+ $(,)? }) => {
        #[doc = concat!("Operation kinds for ", stringify!($name), ".")]
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        pub enum $name { $(#[doc = $text] $variant),+ }
        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str(match self { $(Self::$variant => $text),+ })
            }
        }
        impl FromStr for $name {
            type Err = String;
            fn from_str(s: &str) -> Result<Self, String> {
                match s { $($text => Ok(Self::$variant)),+, _ => Err(format!("unknown {}: {s}", stringify!($name))) }
            }
        }
    };
}
operation_names!(WordBinaryOp {
    Add => "add", Sub => "sub", Mul => "mul", Div => "div", Rem => "rem",
    And => "and", Or => "or", Xor => "xor", Shl => "shl", LShr => "lshr", AShr => "ashr"
});
operation_names!(WordUnaryOp { Neg => "neg", Not => "not" });
operation_names!(WordCompareOp { Eq => "eq", Ne => "ne", Lt => "lt", Le => "le", Gt => "gt", Ge => "ge" });
operation_names!(WordRounding {
    Floor => "floor", HalfUp => "half_up", TowardZero => "toward_zero",
    HalfEven => "half_even", HalfAway => "half_away"
});

/// Scalar SSA instructions with explicit result geometry.
#[derive(Debug, Clone, PartialEq)]
pub enum WordOp {
    /// Capture an integer word on each rising clock, asynchronously resetting to zero.
    ///
    /// Input and result have the same explicit integer type, including widths
    /// above 64 bits. One operation introduces exactly one clock of latency;
    /// it never saturates, rescales or narrows the physical word.
    Delay {
        /// Earlier SSA word sampled on each rising edge of clk.
        input: ValueId,
        /// Identical input, register and result geometry.
        ty: ScType,
    },
    /// Modulo-2^width arithmetic with identically typed integer operands.
    ///
    /// Division truncates toward zero. Remainder has the dividend's sign.
    /// Both division and remainder return zero for a zero divisor. Signed
    /// MIN / -1 wraps to MIN and MIN rem -1 returns zero.
    ///
    /// Shift counts are the unsigned physical rhs bits. Counts at least the
    /// word width yield zero for shl/lshr or sign fill for signed-only ashr.
    /// Unlike Rust wrapping_shl, counts are never reduced modulo the width.
    Binary {
        /// Integer instruction applied at the declared word width.
        kind: WordBinaryOp,
        /// Left integer operand.
        lhs: ValueId,
        /// Right integer operand; shift counts use its unsigned physical bits.
        rhs: ValueId,
        /// Explicit result geometry; arithmetic operands must have this identical integer type.
        ty: ScType,
    },
    /// Two's-complement negation or bitwise complement.
    Unary {
        /// Integer instruction applied at the declared word width.
        kind: WordUnaryOp,
        /// SSA source word.
        input: ValueId,
        /// Explicit result geometry; arithmetic operands must have this identical integer type.
        ty: ScType,
    },
    /// Signedness comes from the identically typed integer operands.
    Compare {
        /// Comparison predicate, with ordering determined by operand signedness.
        predicate: WordCompareOp,
        /// Left integer operand.
        lhs: ValueId,
        /// Right integer operand, identically typed to the left operand.
        rhs: ValueId,
    },
    /// Sign/zero extend according to the input type, or discard high bits.
    Resize {
        /// Source word whose physical width is extended or truncated.
        input: ValueId,
        /// Destination integer type, including the new width and signedness.
        ty: ScType,
    },
    /// Preserve physical bits and change their interpretation without rescaling.
    Bitcast {
        /// Source scalar word whose physical bits remain unchanged.
        input: ValueId,
        /// Destination scalar interpretation with the same physical width.
        ty: ScType,
    },
    /// A Boolean condition chooses between identically typed scalar words.
    Select {
        /// Boolean selector; numeric word conditions require an explicit conversion.
        condition: ValueId,
        /// Scalar branch returned when the condition is true.
        when_true: ValueId,
        /// Scalar branch returned when the condition is false.
        when_false: ValueId,
        /// Explicit destination scalar interpretation; bitcasts preserve width and selects require identical branch types.
        ty: ScType,
    },
    /// Divide by 2^shift using an explicit rounding rule, without intermediate overflow.
    Round {
        /// SSA source word.
        input: ValueId,
        /// Nonnegative power-of-two exponent, strictly smaller than the word width.
        shift: u32,
        /// Rounding rule; half-up ties go toward positive infinity, half-away away from zero, and half-even to the even neighbour.
        mode: WordRounding,
        /// Explicit result geometry; arithmetic operands must have this identical integer type.
        ty: ScType,
    },
    /// Saturate at the supplied inclusive encoded-integer bounds.
    ///
    /// Bounds are i64 literals even when the data word is wider than 64 bits.
    /// Bounds above i64::MAX are not representable; no implicit narrowing occurs.
    Clamp {
        /// SSA source word.
        input: ValueId,
        /// Inclusive lower encoded-integer bound, representable as i64.
        min: i64,
        /// Inclusive upper encoded-integer bound, representable as i64.
        max: i64,
        /// Explicit result geometry; arithmetic operands must have this identical integer type.
        ty: ScType,
    },
}

impl WordOp {
    /// Declared result type; comparisons produce one Boolean bit.
    pub fn result_type(&self) -> &ScType {
        match self {
            Self::Compare { .. } => &ScType::Bool,
            Self::Delay { ty, .. }
            | Self::Binary { ty, .. }
            | Self::Unary { ty, .. }
            | Self::Resize { ty, .. }
            | Self::Bitcast { ty, .. }
            | Self::Select { ty, .. }
            | Self::Round { ty, .. }
            | Self::Clamp { ty, .. } => ty,
        }
    }
    /// SSA dependencies in source order.
    pub fn operands(&self) -> Vec<ValueId> {
        match self {
            Self::Binary { lhs, rhs, .. } | Self::Compare { lhs, rhs, .. } => vec![*lhs, *rhs],
            Self::Delay { input, .. }
            | Self::Unary { input, .. }
            | Self::Resize { input, .. }
            | Self::Bitcast { input, .. }
            | Self::Round { input, .. }
            | Self::Clamp { input, .. } => vec![*input],
            Self::Select {
                condition,
                when_true,
                when_false,
                ..
            } => vec![*condition, *when_true, *when_false],
        }
    }
}

impl fmt::Display for WordOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Delay { input, .. } => write!(f, "delay {input}"),
            Self::Binary { kind, lhs, rhs, .. } => write!(f, "{kind} {lhs}, {rhs}"),
            Self::Unary { kind, input, .. } => write!(f, "{kind} {input}"),
            Self::Compare {
                predicate,
                lhs,
                rhs,
            } => write!(f, "cmp_{predicate} {lhs}, {rhs}"),
            Self::Resize { input, .. } => write!(f, "resize {input}"),
            Self::Bitcast { input, .. } => write!(f, "bitcast {input}"),
            Self::Select {
                condition,
                when_true,
                when_false,
                ..
            } => write!(f, "select {condition}, {when_true}, {when_false}"),
            Self::Round {
                input, shift, mode, ..
            } => write!(f, "round {input}, {shift}, {mode}"),
            Self::Clamp {
                input, min, max, ..
            } => write!(f, "clamp {input}, {min}, {max}"),
        }?;
        write!(f, " : {}", self.result_type())
    }
}
