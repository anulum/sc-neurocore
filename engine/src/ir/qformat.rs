// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust Q-format and mixed dense contracts

use std::error::Error;
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QFormat {
    pub integer_bits: u8,
    pub fraction_bits: u8,
}

impl QFormat {
    pub const fn q8_8() -> Self {
        Self {
            integer_bits: 8,
            fraction_bits: 8,
        }
    }

    pub const fn q16_16() -> Self {
        Self {
            integer_bits: 16,
            fraction_bits: 16,
        }
    }

    pub fn new(integer_bits: u8, fraction_bits: u8) -> Result<Self, QFormatError> {
        if integer_bits == 0 {
            return Err(QFormatError::MissingSignBit);
        }
        let total_bits = u16::from(integer_bits) + u16::from(fraction_bits);
        if total_bits == 0 || total_bits > 63 {
            return Err(QFormatError::TotalBitsTooWide(total_bits));
        }
        Ok(Self {
            integer_bits,
            fraction_bits,
        })
    }

    pub fn total_bits(self) -> u8 {
        self.integer_bits + self.fraction_bits
    }

    pub fn scale(self) -> i128 {
        1_i128 << self.fraction_bits
    }

    pub fn min_value(self) -> f64 {
        -((1_i128 << (self.total_bits() - 1)) as f64) / self.scale() as f64
    }

    pub fn max_value(self) -> f64 {
        ((1_i128 << (self.total_bits() - 1)) - 1) as f64 / self.scale() as f64
    }

    pub fn label(self) -> String {
        format!("Q{}.{}", self.integer_bits, self.fraction_bits)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QFormatError {
    MissingSignBit,
    TotalBitsTooWide(u16),
    AccumulatorNarrower,
    AccumulatorFractionLoss,
    AccumulatorRangeLoss,
}

impl fmt::Display for QFormatError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingSignBit => write!(f, "integer_bits must include the sign bit"),
            Self::TotalBitsTooWide(bits) => {
                write!(f, "Q-format total bits exceed i64 range: {bits}")
            }
            Self::AccumulatorNarrower => write!(
                f,
                "accumulator format must not be narrower than weight format"
            ),
            Self::AccumulatorFractionLoss => {
                write!(
                    f,
                    "accumulator format must preserve weight fractional precision"
                )
            }
            Self::AccumulatorRangeLoss => {
                write!(f, "accumulator format must cover the full weight range")
            }
        }
    }
}

impl Error for QFormatError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QFormatMixed {
    pub weight_fmt: QFormat,
    pub accum_fmt: QFormat,
    pub scale_per_tensor: bool,
}

impl QFormatMixed {
    pub fn q8_8_q16_16() -> Self {
        Self {
            weight_fmt: QFormat::q8_8(),
            accum_fmt: QFormat::q16_16(),
            scale_per_tensor: true,
        }
    }

    pub fn new(
        weight_fmt: QFormat,
        accum_fmt: QFormat,
        scale_per_tensor: bool,
    ) -> Result<Self, QFormatError> {
        if accum_fmt.total_bits() < weight_fmt.total_bits() {
            return Err(QFormatError::AccumulatorNarrower);
        }
        if accum_fmt.fraction_bits < weight_fmt.fraction_bits {
            return Err(QFormatError::AccumulatorFractionLoss);
        }
        if accum_fmt.min_value() > weight_fmt.min_value()
            || accum_fmt.max_value() < weight_fmt.max_value()
        {
            return Err(QFormatError::AccumulatorRangeLoss);
        }
        Ok(Self {
            weight_fmt,
            accum_fmt,
            scale_per_tensor,
        })
    }

    pub fn accumulator_guard_bits(self) -> u8 {
        self.accum_fmt.total_bits() - self.weight_fmt.total_bits()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockFloatingMode {
    pub mantissa_bits: u8,
    pub exponent_bits: u8,
    pub block_size: usize,
}

impl BlockFloatingMode {
    pub fn new(
        mantissa_bits: u8,
        exponent_bits: u8,
        block_size: usize,
    ) -> Result<Self, BlockFloatingError> {
        if mantissa_bits < 2 {
            return Err(BlockFloatingError::MantissaTooNarrow);
        }
        if exponent_bits == 0 || exponent_bits > 7 {
            return Err(BlockFloatingError::InvalidExponentBits);
        }
        if block_size == 0 {
            return Err(BlockFloatingError::EmptyBlock);
        }
        Ok(Self {
            mantissa_bits,
            exponent_bits,
            block_size,
        })
    }

    pub fn bfp16_e3_x32() -> Self {
        Self {
            mantissa_bits: 16,
            exponent_bits: 3,
            block_size: 32,
        }
    }

    pub fn exponent_bias(self) -> i32 {
        (1_i32 << (self.exponent_bits - 1)) - 1
    }

    pub fn min_exponent(self) -> i32 {
        -self.exponent_bias()
    }

    pub fn max_exponent(self) -> i32 {
        ((1_i32 << self.exponent_bits) - 1) - self.exponent_bias()
    }

    pub fn mantissa_range(self) -> i128 {
        (1_i128 << (self.mantissa_bits - 1)) - 1
    }

    pub fn exponent_code_max(self) -> u8 {
        ((1_u16 << self.exponent_bits) - 1) as u8
    }

    pub fn block_exponent_count(self, parameter_count: usize) -> Result<usize, BlockFloatingError> {
        if parameter_count == 0 {
            return Ok(0);
        }
        parameter_count
            .checked_add(self.block_size - 1)
            .map(|value| value / self.block_size)
            .ok_or(BlockFloatingError::ParameterCountOverflow)
    }

    pub fn block_exponent_layout(
        self,
        parameter_count: usize,
    ) -> Result<BlockExponentLayout, BlockFloatingError> {
        Ok(BlockExponentLayout {
            parameter_count,
            block_size: self.block_size,
            exponent_count: self.block_exponent_count(parameter_count)?,
            last_block_size: if parameter_count == 0 {
                0
            } else {
                let remainder = parameter_count % self.block_size;
                if remainder == 0 {
                    self.block_size
                } else {
                    remainder
                }
            },
        })
    }

    pub fn validate_exponent_count(
        self,
        parameter_count: usize,
        exponent_count: usize,
    ) -> Result<(), BlockFloatingError> {
        let expected = self.block_exponent_count(parameter_count)?;
        if exponent_count != expected {
            return Err(BlockFloatingError::ExponentCountMismatch {
                expected,
                actual: exponent_count,
            });
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockExponentLayout {
    pub parameter_count: usize,
    pub block_size: usize,
    pub exponent_count: usize,
    pub last_block_size: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BlockFloatingError {
    MantissaTooNarrow,
    InvalidExponentBits,
    EmptyBlock,
    ParameterCountOverflow,
    ExponentCountMismatch { expected: usize, actual: usize },
}

impl fmt::Display for BlockFloatingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MantissaTooNarrow => write!(f, "mantissa bits must be at least 2"),
            Self::InvalidExponentBits => write!(f, "exponent bits must be in 1..=7"),
            Self::EmptyBlock => write!(f, "block size must be positive"),
            Self::ParameterCountOverflow => write!(f, "parameter count overflows block layout"),
            Self::ExponentCountMismatch { expected, actual } => {
                write!(
                    f,
                    "exponent count mismatch: expected {expected}, got {actual}"
                )
            }
        }
    }
}

impl Error for BlockFloatingError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MixedDenseResult {
    pub outputs_q1616: Vec<i32>,
    pub overflow: bool,
    pub overflow_count: usize,
    pub underflow_count: usize,
    pub abs_bounds_q1616: Vec<i64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrecisionTrapReport {
    pub output_count: usize,
    pub overflow: bool,
    pub overflow_count: usize,
    pub underflow: bool,
    pub underflow_count: usize,
    pub saturated_min_count: usize,
    pub saturated_max_count: usize,
}

impl PrecisionTrapReport {
    pub fn from_q1616(
        outputs_q1616: &[i32],
        overflow_count: usize,
        underflow_count: usize,
    ) -> Self {
        let saturated_min_count = outputs_q1616
            .iter()
            .filter(|&&value| value == i32::MIN)
            .count();
        let saturated_max_count = outputs_q1616
            .iter()
            .filter(|&&value| value == i32::MAX)
            .count();
        Self {
            output_count: outputs_q1616.len(),
            overflow: overflow_count > 0,
            overflow_count,
            underflow: underflow_count > 0,
            underflow_count,
            saturated_min_count,
            saturated_max_count,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrecisionEnvelopeReport {
    pub output_count: usize,
    pub overflow: bool,
    pub overflow_count: usize,
    pub underflow: bool,
    pub underflow_count: usize,
    pub observed_overflow_free: bool,
    pub observed_underflow_free: bool,
    pub conservative_overflow_free: bool,
    pub max_abs_output_q1616: i64,
    pub max_abs_bound_q1616: i64,
    pub conservative_safe_bound_q1616: i64,
    pub min_headroom_q1616: i64,
    pub required_total_bits_q1616: u8,
    pub required_integer_bits_q1616: u8,
    pub width_headroom_bits_q1616: i16,
    pub saturation_required: bool,
    pub static_overflow_proven_safe: bool,
}

impl MixedDenseResult {
    pub fn precision_trap_report(&self) -> PrecisionTrapReport {
        PrecisionTrapReport::from_q1616(
            &self.outputs_q1616,
            self.overflow_count,
            self.underflow_count,
        )
    }

    pub fn precision_envelope_report(&self) -> PrecisionEnvelopeReport {
        let max_abs_output_q1616 = self
            .outputs_q1616
            .iter()
            .map(|&value| abs_i32_to_i64(value))
            .max()
            .unwrap_or(0);
        let max_abs_bound_q1616 = self.abs_bounds_q1616.iter().copied().max().unwrap_or(0);
        let conservative_safe_bound_q1616 = i64::from(i32::MAX);
        let min_headroom_q1616 = conservative_safe_bound_q1616.saturating_sub(max_abs_bound_q1616);
        let required_total_bits_q1616 = required_signed_total_bits(max_abs_bound_q1616);
        let required_integer_bits_q1616 = required_integer_bits_q1616(required_total_bits_q1616);
        let width_headroom_bits_q1616 = 32_i16 - i16::from(required_total_bits_q1616);
        let saturation_required = required_total_bits_q1616 > 32;
        PrecisionEnvelopeReport {
            output_count: self.outputs_q1616.len(),
            overflow: self.overflow,
            overflow_count: self.overflow_count,
            underflow: self.underflow_count > 0,
            underflow_count: self.underflow_count,
            observed_overflow_free: self.overflow_count == 0,
            observed_underflow_free: self.underflow_count == 0,
            conservative_overflow_free: max_abs_bound_q1616 <= conservative_safe_bound_q1616,
            max_abs_output_q1616,
            max_abs_bound_q1616,
            conservative_safe_bound_q1616,
            min_headroom_q1616,
            required_total_bits_q1616,
            required_integer_bits_q1616,
            width_headroom_bits_q1616,
            saturation_required,
            static_overflow_proven_safe: !saturation_required,
        }
    }
}

fn required_signed_total_bits(abs_bound_q1616: i64) -> u8 {
    if abs_bound_q1616 <= 0 {
        return 1;
    }
    (64 - (abs_bound_q1616 as u64).leading_zeros()) as u8 + 1
}

fn required_integer_bits_q1616(required_total_bits_q1616: u8) -> u8 {
    required_total_bits_q1616.saturating_sub(16).max(1)
}

fn abs_i32_to_i64(value: i32) -> i64 {
    if value == i32::MIN {
        i64::from(i32::MAX) + 1
    } else {
        i64::from(value.abs())
    }
}

fn i128_to_i64_saturating(value: i128) -> i64 {
    if value > i128::from(i64::MAX) {
        i64::MAX
    } else if value < i128::from(i64::MIN) {
        i64::MIN
    } else {
        value as i64
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MixedDenseError {
    EmptyShape,
    ShapeOverflow,
    WeightLengthMismatch { expected: usize, actual: usize },
    InputLengthMismatch { expected: usize, actual: usize },
}

impl fmt::Display for MixedDenseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyShape => write!(f, "dense shape must have positive inputs and outputs"),
            Self::ShapeOverflow => write!(f, "dense shape overflows addressable memory"),
            Self::WeightLengthMismatch { expected, actual } => {
                write!(
                    f,
                    "weight length mismatch: expected {expected}, got {actual}"
                )
            }
            Self::InputLengthMismatch { expected, actual } => {
                write!(
                    f,
                    "input length mismatch: expected {expected}, got {actual}"
                )
            }
        }
    }
}

impl Error for MixedDenseError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BlockFloatingDenseError {
    EmptyShape,
    ShapeOverflow,
    MantissaLengthMismatch { expected: usize, actual: usize },
    ExponentLengthMismatch { expected: usize, actual: usize },
    InputLengthMismatch { expected: usize, actual: usize },
    MantissaOutOfRange { index: usize, value: i16 },
    ExponentOutOfRange { index: usize, value: u8 },
}

impl fmt::Display for BlockFloatingDenseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyShape => write!(f, "dense shape must have positive inputs and outputs"),
            Self::ShapeOverflow => write!(f, "dense shape overflows addressable memory"),
            Self::MantissaLengthMismatch { expected, actual } => {
                write!(
                    f,
                    "mantissa length mismatch: expected {expected}, got {actual}"
                )
            }
            Self::ExponentLengthMismatch { expected, actual } => {
                write!(
                    f,
                    "exponent length mismatch: expected {expected}, got {actual}"
                )
            }
            Self::InputLengthMismatch { expected, actual } => {
                write!(
                    f,
                    "input length mismatch: expected {expected}, got {actual}"
                )
            }
            Self::MantissaOutOfRange { index, value } => {
                write!(
                    f,
                    "mantissa at index {index} exceeds configured range: {value}"
                )
            }
            Self::ExponentOutOfRange { index, value } => {
                write!(
                    f,
                    "exponent at index {index} exceeds configured range: {value}"
                )
            }
        }
    }
}

impl Error for BlockFloatingDenseError {}

pub fn mixed_dense_q88_q1616(
    weights_q88: &[i16],
    inputs_q1616: &[i32],
    n_outputs: usize,
    n_inputs: usize,
) -> Result<MixedDenseResult, MixedDenseError> {
    if n_inputs == 0 || n_outputs == 0 {
        return Err(MixedDenseError::EmptyShape);
    }
    let expected_weights = n_outputs
        .checked_mul(n_inputs)
        .ok_or(MixedDenseError::ShapeOverflow)?;
    if weights_q88.len() != expected_weights {
        return Err(MixedDenseError::WeightLengthMismatch {
            expected: expected_weights,
            actual: weights_q88.len(),
        });
    }
    if inputs_q1616.len() != n_inputs {
        return Err(MixedDenseError::InputLengthMismatch {
            expected: n_inputs,
            actual: inputs_q1616.len(),
        });
    }

    let mut outputs_q1616 = Vec::with_capacity(n_outputs);
    let mut abs_bounds_q1616 = Vec::with_capacity(n_outputs);
    let mut overflow_count = 0_usize;
    let mut underflow_count = 0_usize;
    for output_idx in 0..n_outputs {
        let mut sum: i128 = 0;
        let mut abs_bound: i128 = 0;
        let row_start = output_idx * n_inputs;
        for input_idx in 0..n_inputs {
            let weight = i128::from(weights_q88[row_start + input_idx]);
            let input = i128::from(inputs_q1616[input_idx]);
            sum += weight * input;
            abs_bound += weight.abs() * input.abs();
        }
        let scaled = sum >> 8;
        let scaled_bound = (abs_bound + ((1_i128 << 8) - 1)) >> 8;
        abs_bounds_q1616.push(i128_to_i64_saturating(scaled_bound));
        if scaled > i128::from(i32::MAX) {
            outputs_q1616.push(i32::MAX);
            overflow_count += 1;
        } else if scaled < i128::from(i32::MIN) {
            outputs_q1616.push(i32::MIN);
            overflow_count += 1;
        } else {
            if sum != 0 && scaled == 0 {
                underflow_count += 1;
            }
            outputs_q1616.push(scaled as i32);
        }
    }

    Ok(MixedDenseResult {
        outputs_q1616,
        overflow: overflow_count > 0,
        overflow_count,
        underflow_count,
        abs_bounds_q1616,
    })
}

/// Per-element results of a batched mixed-precision Q8.8 × Q16.16 dense MAC.
///
/// Each vector is row-major `n_batch * n_outputs`; element `(b, o)` lives at
/// index `b * n_outputs + o`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MixedDenseBatchResult {
    /// Saturated Q16.16 accumulator codes.
    pub outputs_q1616: Vec<i32>,
    /// `true` where the accumulator left the Q16.16 range.
    pub overflow: Vec<bool>,
    /// `true` where a non-zero contraction rounded to zero without overflowing.
    pub underflow: Vec<bool>,
}

/// Batched integer mixed-precision Q8.8 × Q16.16 dense MAC.
///
/// `weights_q88` is a row-major `n_outputs * n_inputs` Q8.8 matrix; `inputs_q1616`
/// is a row-major `n_batch * n_inputs` Q16.16 code buffer. Each output divides the
/// integer contraction by the Q8.8 weight scale (an arithmetic shift, i.e. floor
/// division) and saturates to the Q16.16 code range, matching the Python floor and
/// the Julia/Go/Mojo backends bit-for-bit.
pub fn mixed_dense_forward_batch_q88_q1616(
    weights_q88: &[i16],
    inputs_q1616: &[i32],
    n_outputs: usize,
    n_inputs: usize,
) -> Result<MixedDenseBatchResult, MixedDenseError> {
    if n_inputs == 0 || n_outputs == 0 {
        return Err(MixedDenseError::EmptyShape);
    }
    let expected_weights = n_outputs
        .checked_mul(n_inputs)
        .ok_or(MixedDenseError::ShapeOverflow)?;
    if weights_q88.len() != expected_weights {
        return Err(MixedDenseError::WeightLengthMismatch {
            expected: expected_weights,
            actual: weights_q88.len(),
        });
    }
    if inputs_q1616.is_empty() || !inputs_q1616.len().is_multiple_of(n_inputs) {
        return Err(MixedDenseError::InputLengthMismatch {
            expected: n_inputs,
            actual: inputs_q1616.len(),
        });
    }

    let n_batch = inputs_q1616.len() / n_inputs;
    let count = n_batch * n_outputs;
    let mut outputs_q1616 = Vec::with_capacity(count);
    let mut overflow = Vec::with_capacity(count);
    let mut underflow = Vec::with_capacity(count);
    for batch_idx in 0..n_batch {
        let input_row = &inputs_q1616[batch_idx * n_inputs..(batch_idx + 1) * n_inputs];
        for output_idx in 0..n_outputs {
            let weight_row = &weights_q88[output_idx * n_inputs..(output_idx + 1) * n_inputs];
            let mut sum: i128 = 0;
            for input_idx in 0..n_inputs {
                sum += i128::from(weight_row[input_idx]) * i128::from(input_row[input_idx]);
            }
            let scaled = sum >> 8;
            if scaled > i128::from(i32::MAX) {
                outputs_q1616.push(i32::MAX);
                overflow.push(true);
                underflow.push(false);
            } else if scaled < i128::from(i32::MIN) {
                outputs_q1616.push(i32::MIN);
                overflow.push(true);
                underflow.push(false);
            } else {
                outputs_q1616.push(scaled as i32);
                overflow.push(false);
                underflow.push(sum != 0 && scaled == 0);
            }
        }
    }
    Ok(MixedDenseBatchResult {
        outputs_q1616,
        overflow,
        underflow,
    })
}

pub fn block_floating_dense_q16(
    mantissas: &[i16],
    exponents: &[u8],
    inputs_q1616: &[i32],
    n_outputs: usize,
    n_inputs: usize,
    mode: BlockFloatingMode,
) -> Result<MixedDenseResult, BlockFloatingDenseError> {
    if n_inputs == 0 || n_outputs == 0 {
        return Err(BlockFloatingDenseError::EmptyShape);
    }
    let expected_weights = n_outputs
        .checked_mul(n_inputs)
        .ok_or(BlockFloatingDenseError::ShapeOverflow)?;
    let expected_blocks = mode
        .block_exponent_count(expected_weights)
        .map_err(|_| BlockFloatingDenseError::ShapeOverflow)?;

    if mantissas.len() != expected_weights {
        return Err(BlockFloatingDenseError::MantissaLengthMismatch {
            expected: expected_weights,
            actual: mantissas.len(),
        });
    }
    if exponents.len() != expected_blocks {
        return Err(BlockFloatingDenseError::ExponentLengthMismatch {
            expected: expected_blocks,
            actual: exponents.len(),
        });
    }
    if inputs_q1616.len() != n_inputs {
        return Err(BlockFloatingDenseError::InputLengthMismatch {
            expected: n_inputs,
            actual: inputs_q1616.len(),
        });
    }

    let mantissa_range = mode.mantissa_range();
    for (index, &mantissa) in mantissas.iter().enumerate() {
        if i128::from(mantissa).abs() > mantissa_range {
            return Err(BlockFloatingDenseError::MantissaOutOfRange {
                index,
                value: mantissa,
            });
        }
    }
    let exponent_code_max = mode.exponent_code_max();
    for (index, &exponent) in exponents.iter().enumerate() {
        if exponent > exponent_code_max {
            return Err(BlockFloatingDenseError::ExponentOutOfRange {
                index,
                value: exponent,
            });
        }
    }

    let mut outputs_q1616 = Vec::with_capacity(n_outputs);
    let mut abs_bounds_q1616 = Vec::with_capacity(n_outputs);
    let mut overflow_count = 0_usize;
    let mut underflow_count = 0_usize;
    for output_idx in 0..n_outputs {
        let mut sum: i128 = 0;
        let mut abs_bound: i128 = 0;
        let mut dropped_sub_lsb_product = false;
        let row_start = output_idx * n_inputs;
        for input_idx in 0..n_inputs {
            let linear_idx = row_start + input_idx;
            let block_idx = linear_idx / mode.block_size;
            let product = i128::from(mantissas[linear_idx]) * i128::from(inputs_q1616[input_idx]);
            let shift = i32::from(exponents[block_idx]) - mode.exponent_bias();
            if shift >= 0 {
                sum += product << shift;
                abs_bound += product.abs() << shift;
            } else {
                sum += product >> (-shift);
                let divisor_shift = -shift;
                if product != 0 && (product >> divisor_shift) == 0 {
                    dropped_sub_lsb_product = true;
                }
                abs_bound += (product.abs() + ((1_i128 << divisor_shift) - 1)) >> divisor_shift;
            }
        }
        abs_bounds_q1616.push(i128_to_i64_saturating(abs_bound));
        if sum > i128::from(i32::MAX) {
            outputs_q1616.push(i32::MAX);
            overflow_count += 1;
        } else if sum < i128::from(i32::MIN) {
            outputs_q1616.push(i32::MIN);
            overflow_count += 1;
        } else {
            if sum == 0 && dropped_sub_lsb_product {
                underflow_count += 1;
            }
            outputs_q1616.push(sum as i32);
        }
    }

    Ok(MixedDenseResult {
        outputs_q1616,
        overflow: overflow_count > 0,
        overflow_count,
        underflow_count,
        abs_bounds_q1616,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mixed_dense_batch_matches_single_per_output() {
        let weights = [256_i16, -128, 64, 512];
        let inputs = [512_i32, 1024, 256, 768];
        let batch = mixed_dense_forward_batch_q88_q1616(&weights, &inputs, 2, 2).unwrap();
        // n_batch = 2, n_outputs = 2.
        assert_eq!(batch.outputs_q1616.len(), 4);
        for batch_idx in 0..2 {
            let row = &inputs[batch_idx * 2..batch_idx * 2 + 2];
            let single = mixed_dense_q88_q1616(&weights, row, 2, 2).unwrap();
            for output_idx in 0..2 {
                assert_eq!(
                    batch.outputs_q1616[batch_idx * 2 + output_idx],
                    single.outputs_q1616[output_idx]
                );
            }
        }
    }

    #[test]
    fn mixed_dense_batch_floor_division_is_signed() {
        // raw = -1 -> -1 >> 8 = -1 (floor, not truncation toward zero).
        let batch = mixed_dense_forward_batch_q88_q1616(&[1], &[-1], 1, 1).unwrap();
        assert_eq!(batch.outputs_q1616, vec![-1]);
        assert!(!batch.overflow[0]);
        assert!(!batch.underflow[0]);
    }

    #[test]
    fn mixed_dense_batch_flags_overflow_and_underflow() {
        let weights = [i16::MAX; 4];
        let inputs = [2_000_000_000_i32, 2_000_000_000, 1, 1];
        let batch = mixed_dense_forward_batch_q88_q1616(&weights, &inputs, 1, 4).unwrap();
        assert!(batch.overflow[0]);
        assert_eq!(batch.outputs_q1616[0], i32::MAX);
        // A tiny non-zero contraction that rounds to zero is an underflow.
        let under = mixed_dense_forward_batch_q88_q1616(&[1], &[1], 1, 1).unwrap();
        assert_eq!(under.outputs_q1616, vec![0]);
        assert!(under.underflow[0]);
        assert!(!under.overflow[0]);
    }

    #[test]
    fn mixed_dense_batch_rejects_bad_shapes() {
        assert_eq!(
            mixed_dense_forward_batch_q88_q1616(&[1], &[1], 0, 1).unwrap_err(),
            MixedDenseError::EmptyShape
        );
        assert!(matches!(
            mixed_dense_forward_batch_q88_q1616(&[1, 1], &[1], 1, 1).unwrap_err(),
            MixedDenseError::WeightLengthMismatch { .. }
        ));
        assert!(matches!(
            mixed_dense_forward_batch_q88_q1616(&[1, 1], &[1, 1, 1], 1, 2).unwrap_err(),
            MixedDenseError::InputLengthMismatch { .. }
        ));
    }

    #[test]
    fn qformat_mixed_default_matches_python_contract() {
        let fmt = QFormatMixed::q8_8_q16_16();

        assert_eq!(fmt.weight_fmt.label(), "Q8.8");
        assert_eq!(fmt.accum_fmt.label(), "Q16.16");
        assert_eq!(fmt.accumulator_guard_bits(), 16);
    }

    #[test]
    fn rejects_accumulator_precision_loss() {
        let result = QFormatMixed::new(
            QFormat::new(8, 12).unwrap(),
            QFormat::new(16, 8).unwrap(),
            true,
        );

        assert_eq!(result.unwrap_err(), QFormatError::AccumulatorFractionLoss);
    }

    #[test]
    fn mixed_dense_matches_manual_q88_q1616_codes() {
        let weights = [128_i16, -64_i16, 256_i16, 32_i16];
        let inputs = [32768_i32, -16384_i32];

        let result = mixed_dense_q88_q1616(&weights, &inputs, 2, 2).unwrap();

        assert_eq!(result.outputs_q1616, vec![20480, 30720]);
        assert!(!result.overflow);
        assert_eq!(result.overflow_count, 0);
        assert_eq!(result.underflow_count, 0);
        assert_eq!(result.abs_bounds_q1616, vec![20480, 34816]);

        let envelope = result.precision_envelope_report();
        assert!(envelope.observed_overflow_free);
        assert!(envelope.observed_underflow_free);
        assert!(envelope.conservative_overflow_free);
        assert_eq!(envelope.max_abs_output_q1616, 30720);
        assert_eq!(envelope.max_abs_bound_q1616, 34816);
        assert_eq!(envelope.required_total_bits_q1616, 17);
        assert_eq!(envelope.required_integer_bits_q1616, 1);
        assert_eq!(envelope.width_headroom_bits_q1616, 15);
        assert!(!envelope.saturation_required);
        assert!(envelope.static_overflow_proven_safe);
    }

    #[test]
    fn mixed_dense_negative_products_follow_arithmetic_shift() {
        let result = mixed_dense_q88_q1616(&[128_i16], &[-1_i32], 1, 1).unwrap();

        assert_eq!(result.outputs_q1616, vec![-1]);
    }

    #[test]
    fn mixed_dense_reports_sub_lsb_underflow() {
        let result = mixed_dense_q88_q1616(&[1_i16], &[1_i32], 1, 1).unwrap();

        assert_eq!(result.outputs_q1616, vec![0]);
        assert_eq!(result.overflow_count, 0);
        assert_eq!(result.underflow_count, 1);

        let report = result.precision_trap_report();
        assert!(!report.overflow);
        assert!(report.underflow);
        assert_eq!(report.underflow_count, 1);

        let envelope = result.precision_envelope_report();
        assert!(envelope.observed_overflow_free);
        assert!(!envelope.observed_underflow_free);
    }

    #[test]
    fn mixed_dense_saturates_overflow() {
        let weights = [i16::MAX, i16::MAX];
        let inputs = [i32::MAX, i32::MAX];

        let result = mixed_dense_q88_q1616(&weights, &inputs, 1, 2).unwrap();

        assert_eq!(result.outputs_q1616, vec![i32::MAX]);
        assert!(result.overflow);
        assert_eq!(result.overflow_count, 1);
        assert_eq!(result.underflow_count, 0);

        let report = result.precision_trap_report();
        assert_eq!(report.output_count, 1);
        assert!(report.overflow);
        assert_eq!(report.overflow_count, 1);
        assert!(!report.underflow);
        assert_eq!(report.underflow_count, 0);
        assert_eq!(report.saturated_max_count, 1);
        assert_eq!(report.saturated_min_count, 0);

        let envelope = result.precision_envelope_report();
        assert!(!envelope.observed_overflow_free);
        assert!(envelope.observed_underflow_free);
        assert!(!envelope.conservative_overflow_free);
        assert_eq!(envelope.output_count, 1);
        assert_eq!(envelope.overflow_count, 1);
        assert_eq!(envelope.underflow_count, 0);
        assert!(envelope.max_abs_bound_q1616 > envelope.conservative_safe_bound_q1616);
        assert!(envelope.saturation_required);
        assert!(!envelope.static_overflow_proven_safe);
    }

    #[test]
    fn mixed_dense_rejects_shape_mismatches() {
        assert_eq!(
            mixed_dense_q88_q1616(&[], &[1], 1, 0).unwrap_err(),
            MixedDenseError::EmptyShape
        );
        assert_eq!(
            mixed_dense_q88_q1616(&[1], &[1], 2, 1).unwrap_err(),
            MixedDenseError::WeightLengthMismatch {
                expected: 2,
                actual: 1,
            }
        );
        assert_eq!(
            mixed_dense_q88_q1616(&[1, 2], &[1], 1, 2).unwrap_err(),
            MixedDenseError::InputLengthMismatch {
                expected: 2,
                actual: 1,
            }
        );
    }

    #[test]
    fn block_floating_mode_reports_full_exponent_range() {
        let mode = BlockFloatingMode::new(8, 2, 2).unwrap();

        assert_eq!(mode.exponent_bias(), 1);
        assert_eq!(mode.min_exponent(), -1);
        assert_eq!(mode.max_exponent(), 2);
        assert_eq!(mode.exponent_code_max(), 3);
    }

    #[test]
    fn block_floating_mode_computes_exponent_layout() {
        let mode = BlockFloatingMode::new(16, 3, 32).unwrap();
        let layout = mode.block_exponent_layout(65).unwrap();

        assert_eq!(layout.parameter_count, 65);
        assert_eq!(layout.block_size, 32);
        assert_eq!(layout.exponent_count, 3);
        assert_eq!(layout.last_block_size, 1);
        assert_eq!(mode.block_exponent_count(0).unwrap(), 0);
        assert_eq!(
            mode.validate_exponent_count(65, 2).unwrap_err(),
            BlockFloatingError::ExponentCountMismatch {
                expected: 3,
                actual: 2,
            }
        );
    }

    #[test]
    fn block_floating_dense_matches_manual_shifted_products() {
        let mode = BlockFloatingMode::new(16, 3, 2).unwrap();
        let bias = mode.exponent_bias() as u8;
        let mantissas = [2_i16, -4_i16, 8_i16, 16_i16];
        let exponents = [bias, bias - 1];
        let inputs = [32768_i32, -16384_i32];

        let result = block_floating_dense_q16(&mantissas, &exponents, &inputs, 2, 2, mode).unwrap();

        assert_eq!(result.outputs_q1616, vec![131072, 0]);
        assert!(!result.overflow);
        assert_eq!(result.underflow_count, 0);
        assert_eq!(result.abs_bounds_q1616, vec![131072, 262144]);

        let envelope = result.precision_envelope_report();
        assert!(envelope.observed_overflow_free);
        assert!(envelope.observed_underflow_free);
        assert!(envelope.conservative_overflow_free);
        assert_eq!(envelope.max_abs_output_q1616, 131072);
        assert_eq!(envelope.max_abs_bound_q1616, 262144);
    }

    #[test]
    fn block_floating_dense_seeded_exponent_edges_match_manual_q1616_codes() {
        let mode = BlockFloatingMode::new(16, 3, 2).unwrap();
        let mantissas = [
            1_i16,
            -2_i16,
            i16::MAX,
            -i16::MAX,
            -3_i16,
            4_i16,
            -i16::MAX,
            i16::MAX,
        ];
        let exponents = [
            0_u8,
            mode.exponent_code_max(),
            0_u8,
            mode.exponent_code_max(),
        ];
        let inputs = [32768_i32, -16384_i32, 1_i32, -1_i32];

        let result = block_floating_dense_q16(&mantissas, &exponents, &inputs, 2, 4, mode)
            .expect("seeded exponent-edge dimensions are valid");

        assert_eq!(result.outputs_q1616, vec![1_056_736, -1_069_024]);
        assert_eq!(result.overflow_count, 0);
        assert_eq!(result.underflow_count, 0);
        assert_eq!(result.abs_bounds_q1616, vec![1_056_736, 1_069_024]);

        let envelope = result.precision_envelope_report();
        assert!(envelope.observed_overflow_free);
        assert!(envelope.observed_underflow_free);
        assert!(envelope.conservative_overflow_free);
        assert_eq!(envelope.max_abs_bound_q1616, 1_069_024);
        assert_eq!(envelope.min_headroom_q1616, 2_146_414_623);
    }

    #[test]
    fn block_floating_dense_max_exponent_edge_saturates_and_reports_trap() {
        let mode = BlockFloatingMode::new(16, 3, 2).unwrap();
        let mantissas = [i16::MAX, i16::MAX];
        let exponents = [mode.exponent_code_max()];
        let inputs = [32767_i32 << 16, 32767_i32 << 16];

        let result = block_floating_dense_q16(&mantissas, &exponents, &inputs, 1, 2, mode)
            .expect("max-exponent trap dimensions are valid");

        assert_eq!(result.outputs_q1616, vec![i32::MAX]);
        assert!(result.overflow);
        assert_eq!(result.overflow_count, 1);
        assert_eq!(result.underflow_count, 0);

        let report = result.precision_trap_report();
        assert!(report.overflow);
        assert_eq!(report.overflow_count, 1);
        assert!(!report.underflow);
        assert_eq!(report.saturated_max_count, 1);

        let envelope = result.precision_envelope_report();
        assert!(!envelope.observed_overflow_free);
        assert!(envelope.observed_underflow_free);
        assert!(!envelope.conservative_overflow_free);
        assert!(envelope.max_abs_bound_q1616 > envelope.conservative_safe_bound_q1616);
    }

    #[test]
    fn block_floating_dense_reports_sub_lsb_underflow() {
        let mode = BlockFloatingMode::new(16, 3, 1).unwrap();
        let result = block_floating_dense_q16(&[1_i16], &[0_u8], &[1_i32], 1, 1, mode).unwrap();

        assert_eq!(result.outputs_q1616, vec![0]);
        assert_eq!(result.overflow_count, 0);
        assert_eq!(result.underflow_count, 1);

        let report = result.precision_trap_report();
        assert!(!report.overflow);
        assert!(report.underflow);
        assert_eq!(report.underflow_count, 1);

        let envelope = result.precision_envelope_report();
        assert!(envelope.observed_overflow_free);
        assert!(!envelope.observed_underflow_free);
        assert_eq!(envelope.max_abs_bound_q1616, 1);
    }

    #[test]
    fn block_floating_dense_saturates_large_outputs() {
        let mode = BlockFloatingMode::bfp16_e3_x32();
        let mantissas = vec![i16::MAX; 64];
        let exponents = vec![mode.exponent_code_max(); 2];
        let inputs = vec![i32::MAX; 64];

        let result =
            block_floating_dense_q16(&mantissas, &exponents, &inputs, 1, 64, mode).unwrap();

        assert_eq!(result.outputs_q1616, vec![i32::MAX]);
        assert!(result.overflow);
        assert_eq!(result.overflow_count, 1);
        assert_eq!(result.underflow_count, 0);

        let report = result.precision_trap_report();
        assert_eq!(report.output_count, 1);
        assert!(report.overflow);
        assert_eq!(report.overflow_count, 1);
        assert!(!report.underflow);
        assert_eq!(report.underflow_count, 0);
        assert_eq!(report.saturated_max_count, 1);
        assert_eq!(report.saturated_min_count, 0);

        let envelope = result.precision_envelope_report();
        assert!(!envelope.observed_overflow_free);
        assert!(envelope.observed_underflow_free);
        assert!(!envelope.conservative_overflow_free);
        assert_eq!(envelope.output_count, 1);
        assert_eq!(envelope.overflow_count, 1);
        assert_eq!(envelope.underflow_count, 0);
        assert!(envelope.max_abs_bound_q1616 > envelope.conservative_safe_bound_q1616);
    }

    #[test]
    fn block_floating_dense_rejects_invalid_lengths_and_ranges() {
        let mode = BlockFloatingMode::new(8, 2, 2).unwrap();

        assert_eq!(
            block_floating_dense_q16(&[], &[1], &[1], 1, 0, mode).unwrap_err(),
            BlockFloatingDenseError::EmptyShape
        );
        assert_eq!(
            block_floating_dense_q16(&[1], &[1], &[1], 2, 1, mode).unwrap_err(),
            BlockFloatingDenseError::MantissaLengthMismatch {
                expected: 2,
                actual: 1,
            }
        );
        assert_eq!(
            block_floating_dense_q16(&[1, 2], &[], &[1, 2], 1, 2, mode).unwrap_err(),
            BlockFloatingDenseError::ExponentLengthMismatch {
                expected: 1,
                actual: 0,
            }
        );
        assert_eq!(
            block_floating_dense_q16(&[128, 0], &[1], &[1, 2], 1, 2, mode).unwrap_err(),
            BlockFloatingDenseError::MantissaOutOfRange {
                index: 0,
                value: 128,
            }
        );
    }
}

#[cfg(test)]
mod block_floating_benchmark_contract_tests {
    use super::*;

    const N_INPUTS: usize = 64;
    const N_OUTPUTS: usize = 32;

    fn round_div_nearest_even(value: i32, divisor: i32) -> i16 {
        let sign = if value < 0 { -1 } else { 1 };
        let magnitude = value.abs();
        let quotient = magnitude / divisor;
        let remainder = magnitude % divisor;
        let rounded_magnitude = if remainder * 2 < divisor {
            quotient
        } else if remainder * 2 > divisor {
            quotient + 1
        } else if quotient % 2 == 0 {
            quotient
        } else {
            quotient + 1
        };
        (sign * rounded_magnitude) as i16
    }

    #[test]
    fn block_floating_benchmark_matches_python_quantiser_envelope() {
        let mode = BlockFloatingMode::bfp16_e3_x32();
        let mantissas = (0..(N_INPUTS * N_OUTPUTS))
            .map(|idx| {
                let raw_weight_code = ((idx * 23 + 3) % 1025) as i32 - 512;
                round_div_nearest_even(raw_weight_code, 64)
            })
            .collect::<Vec<_>>();
        let exponents = vec![0_u8; (N_INPUTS * N_OUTPUTS + mode.block_size - 1) / mode.block_size];
        let inputs = (0..N_INPUTS)
            .map(|idx| (((idx * 19 + 5) % 257) as i32 - 128) << 8)
            .collect::<Vec<_>>();

        let result =
            block_floating_dense_q16(&mantissas, &exponents, &inputs, N_OUTPUTS, N_INPUTS, mode)
                .expect("benchmark contract dimensions are valid");
        let envelope = result.precision_envelope_report();

        assert_eq!(result.overflow_count, 0);
        assert_eq!(envelope.max_abs_bound_q1616, 610_816);
        assert!(envelope.conservative_overflow_free);

        let saturating_mantissas = vec![16_384_i16; N_INPUTS * N_OUTPUTS];
        let saturating_exponents =
            vec![2_u8; (N_INPUTS * N_OUTPUTS + mode.block_size - 1) / mode.block_size];
        let saturating_inputs = vec![32767_i32 << 16; N_INPUTS];
        let saturating_result = block_floating_dense_q16(
            &saturating_mantissas,
            &saturating_exponents,
            &saturating_inputs,
            N_OUTPUTS,
            N_INPUTS,
            mode,
        )
        .expect("saturating benchmark contract dimensions are valid");
        let saturating_envelope = saturating_result.precision_envelope_report();

        assert_eq!(saturating_result.overflow_count, N_OUTPUTS);
        assert_eq!(
            saturating_envelope.max_abs_bound_q1616,
            1_125_865_547_104_256
        );
        assert!(!saturating_envelope.conservative_overflow_free);
    }
}
#[cfg(test)]
mod mixed_dense_benchmark_contract_tests {
    use super::*;

    #[test]
    fn mixed_dense_benchmark_contract_matches_python_envelope() {
        const N_INPUTS: usize = 64;
        const N_OUTPUTS: usize = 32;

        let weights = (0..(N_INPUTS * N_OUTPUTS))
            .map(|idx| (((idx * 17 + 11) % 513) as i32 - 256) as i16)
            .collect::<Vec<_>>();
        let inputs = (0..N_INPUTS)
            .map(|idx| (((idx as i32 * 19 + 5) % 257) - 128) << 8)
            .collect::<Vec<_>>();
        let safe = mixed_dense_q88_q1616(&weights, &inputs, N_OUTPUTS, N_INPUTS)
            .expect("benchmark contract dimensions must be valid");
        let safe_envelope = safe.precision_envelope_report();

        assert_eq!(safe.overflow_count, 0);
        assert_eq!(safe_envelope.max_abs_bound_q1616, 531_400);
        assert!(safe_envelope.conservative_overflow_free);
        assert_eq!(safe_envelope.min_headroom_q1616, 2_146_952_247);
        assert_eq!(safe_envelope.required_total_bits_q1616, 21);
        assert_eq!(safe_envelope.required_integer_bits_q1616, 5);
        assert_eq!(safe_envelope.width_headroom_bits_q1616, 11);
        assert!(!safe_envelope.saturation_required);

        let probe_weights = vec![127_i16 << 8; N_INPUTS * N_OUTPUTS];
        let probe_inputs = vec![32767_i32 << 16; N_INPUTS];
        let probe = mixed_dense_q88_q1616(&probe_weights, &probe_inputs, N_OUTPUTS, N_INPUTS)
            .expect("saturating probe dimensions must be valid");
        let probe_envelope = probe.precision_envelope_report();

        assert_eq!(probe.overflow_count, N_OUTPUTS);
        assert_eq!(probe_envelope.max_abs_bound_q1616, 17_454_214_414_336);
        assert!(!probe_envelope.conservative_overflow_free);
        assert_eq!(probe_envelope.required_total_bits_q1616, 45);
        assert_eq!(probe_envelope.required_integer_bits_q1616, 29);
        assert_eq!(probe_envelope.width_headroom_bits_q1616, -13);
        assert!(probe_envelope.saturation_required);
    }
}
