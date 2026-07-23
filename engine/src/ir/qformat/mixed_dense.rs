// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Mixed-precision dense quantisation

use std::error::Error;
use std::fmt;

use super::dense_result::{i128_to_i64_saturating, MixedDenseResult};

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
