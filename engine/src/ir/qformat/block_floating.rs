// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Block-floating dense quantisation

use std::error::Error;
use std::fmt;

use super::dense_result::{i128_to_i64_saturating, MixedDenseResult};

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
        let exponents = vec![0_u8; (N_INPUTS * N_OUTPUTS).div_ceil(mode.block_size)];
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
        let saturating_exponents = vec![2_u8; (N_INPUTS * N_OUTPUTS).div_ceil(mode.block_size)];
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
