// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Dense quantisation result and precision reports

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

pub(super) fn i128_to_i64_saturating(value: i128) -> i64 {
    if value > i128::from(i64::MAX) {
        i64::MAX
    } else if value < i128::from(i64::MIN) {
        i64::MIN
    } else {
        value as i64
    }
}
