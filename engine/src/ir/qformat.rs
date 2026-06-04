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
            Self::TotalBitsTooWide(bits) => write!(f, "Q-format total bits exceed i64 range: {bits}"),
            Self::AccumulatorNarrower => write!(f, "accumulator format must not be narrower than weight format"),
            Self::AccumulatorFractionLoss => {
                write!(f, "accumulator format must preserve weight fractional precision")
            }
            Self::AccumulatorRangeLoss => write!(f, "accumulator format must cover the full weight range"),
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MixedDenseResult {
    pub outputs_q1616: Vec<i32>,
    pub overflow: bool,
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
                write!(f, "weight length mismatch: expected {expected}, got {actual}")
            }
            Self::InputLengthMismatch { expected, actual } => {
                write!(f, "input length mismatch: expected {expected}, got {actual}")
            }
        }
    }
}

impl Error for MixedDenseError {}

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
    let mut overflow = false;
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
            overflow = true;
        } else if scaled < i128::from(i32::MIN) {
            outputs_q1616.push(i32::MIN);
            overflow = true;
        } else {
            outputs_q1616.push(scaled as i32);
        }
    }

    Ok(MixedDenseResult {
        outputs_q1616,
        overflow,
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
}
