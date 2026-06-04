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
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BlockFloatingError {
    MantissaTooNarrow,
    InvalidExponentBits,
    EmptyBlock,
}

impl fmt::Display for BlockFloatingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MantissaTooNarrow => write!(f, "mantissa bits must be at least 2"),
            Self::InvalidExponentBits => write!(f, "exponent bits must be in 1..=7"),
            Self::EmptyBlock => write!(f, "block size must be positive"),
        }
    }
}

impl Error for BlockFloatingError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MixedDenseResult {
    pub outputs_q1616: Vec<i32>,
    pub overflow: bool,
    pub overflow_count: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrecisionTrapReport {
    pub output_count: usize,
    pub overflow: bool,
    pub overflow_count: usize,
    pub saturated_min_count: usize,
    pub saturated_max_count: usize,
}

impl PrecisionTrapReport {
    pub fn from_q1616(outputs_q1616: &[i32], overflow_count: usize) -> Self {
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
            saturated_min_count,
            saturated_max_count,
        }
    }
}

impl MixedDenseResult {
    pub fn precision_trap_report(&self) -> PrecisionTrapReport {
        PrecisionTrapReport::from_q1616(&self.outputs_q1616, self.overflow_count)
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
    let mut overflow_count = 0_usize;
    for output_idx in 0..n_outputs {
        let mut sum: i128 = 0;
        let row_start = output_idx * n_inputs;
        for input_idx in 0..n_inputs {
            let weight = i128::from(weights_q88[row_start + input_idx]);
            let input = i128::from(inputs_q1616[input_idx]);
            sum += weight * input;
        }
        let scaled = sum >> 8;
        if scaled > i128::from(i32::MAX) {
            outputs_q1616.push(i32::MAX);
            overflow_count += 1;
        } else if scaled < i128::from(i32::MIN) {
            outputs_q1616.push(i32::MIN);
            overflow_count += 1;
        } else {
            outputs_q1616.push(scaled as i32);
        }
    }

    Ok(MixedDenseResult {
        outputs_q1616,
        overflow: overflow_count > 0,
        overflow_count,
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
    let expected_blocks = expected_weights
        .checked_add(mode.block_size - 1)
        .ok_or(BlockFloatingDenseError::ShapeOverflow)?
        / mode.block_size;

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
    let mut overflow_count = 0_usize;
    for output_idx in 0..n_outputs {
        let mut sum: i128 = 0;
        let row_start = output_idx * n_inputs;
        for input_idx in 0..n_inputs {
            let linear_idx = row_start + input_idx;
            let block_idx = linear_idx / mode.block_size;
            let product = i128::from(mantissas[linear_idx]) * i128::from(inputs_q1616[input_idx]);
            let shift = i32::from(exponents[block_idx]) - mode.exponent_bias();
            if shift >= 0 {
                sum += product << shift;
            } else {
                sum += product >> (-shift);
            }
        }
        if sum > i128::from(i32::MAX) {
            outputs_q1616.push(i32::MAX);
            overflow_count += 1;
        } else if sum < i128::from(i32::MIN) {
            outputs_q1616.push(i32::MIN);
            overflow_count += 1;
        } else {
            outputs_q1616.push(sum as i32);
        }
    }

    Ok(MixedDenseResult {
        outputs_q1616,
        overflow: overflow_count > 0,
        overflow_count,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

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
    }

    #[test]
    fn mixed_dense_negative_products_follow_arithmetic_shift() {
        let result = mixed_dense_q88_q1616(&[128_i16], &[-1_i32], 1, 1).unwrap();

        assert_eq!(result.outputs_q1616, vec![-1]);
    }

    #[test]
    fn mixed_dense_saturates_overflow() {
        let weights = [i16::MAX, i16::MAX];
        let inputs = [i32::MAX, i32::MAX];

        let result = mixed_dense_q88_q1616(&weights, &inputs, 1, 2).unwrap();

        assert_eq!(result.outputs_q1616, vec![i32::MAX]);
        assert!(result.overflow);
        assert_eq!(result.overflow_count, 1);

        let report = result.precision_trap_report();
        assert_eq!(report.output_count, 1);
        assert!(report.overflow);
        assert_eq!(report.overflow_count, 1);
        assert_eq!(report.saturated_max_count, 1);
        assert_eq!(report.saturated_min_count, 0);
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
    fn block_floating_dense_matches_manual_shifted_products() {
        let mode = BlockFloatingMode::new(16, 3, 2).unwrap();
        let bias = mode.exponent_bias() as u8;
        let mantissas = [2_i16, -4_i16, 8_i16, 16_i16];
        let exponents = [bias, bias - 1];
        let inputs = [32768_i32, -16384_i32];

        let result = block_floating_dense_q16(&mantissas, &exponents, &inputs, 2, 2, mode).unwrap();

        assert_eq!(result.outputs_q1616, vec![131072, 0]);
        assert!(!result.overflow);
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

        let report = result.precision_trap_report();
        assert_eq!(report.output_count, 1);
        assert!(report.overflow);
        assert_eq!(report.overflow_count, 1);
        assert_eq!(report.saturated_max_count, 1);
        assert_eq!(report.saturated_min_count, 0);
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
