// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// Copyright (C) 2020-2026 Miroslav Sotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore - Delay-coded learnable-spike Q8.8 reference

//! Bit-true Q8.8 reference for the delay-coded learnable-spike tent kernel.

use std::error::Error;
use std::fmt;

/// Default fractional precision for Q8.8 DCLS contracts.
pub const DEFAULT_FRACTION: u32 = 8;

/// Default fixed-point data width for DCLS weights and outputs.
pub const DEFAULT_DATA_WIDTH: u32 = 16;

/// Default accumulator width for Q16.16 DCLS accumulators.
pub const DEFAULT_ACCUMULATOR_WIDTH: u32 = 32;

const Q88_ONE: i64 = 1_i64 << DEFAULT_FRACTION;
const I32_MAX_AS_I64: i64 = i32::MAX as i64;
const I32_MIN_AS_I64: i64 = i32::MIN as i64;
const I16_MAX_Q16_16: i64 = (i16::MAX as i64) << DEFAULT_FRACTION;
const I16_MIN_Q16_16: i64 = (i16::MIN as i64) << DEFAULT_FRACTION;

/// Configuration for the bit-true DCLS Q8.8 forward pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DclsLayerConfig {
    pub data_width: u32,
    pub fraction: u32,
    pub accumulator_width: u32,
}

impl Default for DclsLayerConfig {
    fn default() -> Self {
        Self {
            data_width: DEFAULT_DATA_WIDTH,
            fraction: DEFAULT_FRACTION,
            accumulator_width: DEFAULT_ACCUMULATOR_WIDTH,
        }
    }
}

/// Saturating result of one DCLS tent-kernel contraction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DclsForwardResult {
    /// Saturated Q8.8 output.
    pub output_q88: i16,
    /// Saturated Q16.16 accumulator.
    pub accumulator_q16_16: i32,
    /// True when accumulator or output saturation occurred.
    pub overflow: bool,
    /// Number of non-zero spike taps consumed by the contraction.
    pub active_tap_count: usize,
    /// Largest Q8.8 tent gate applied to an active spike.
    pub max_gate_q88: i16,
}

/// DCLS arithmetic contract errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DclsError {
    UnsupportedFormat {
        data_width: u32,
        fraction: u32,
        accumulator_width: u32,
    },
    EmptyTaps,
    MismatchedLengths {
        spikes: usize,
        weights: usize,
    },
    InvalidSigma {
        sigma_q88: i16,
    },
    TapIndexOverflow {
        tap_index: usize,
    },
}

impl fmt::Display for DclsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedFormat {
                data_width,
                fraction,
                accumulator_width,
            } => write!(
                f,
                "unsupported DCLS format data_width={data_width}, fraction={fraction}, accumulator_width={accumulator_width}"
            ),
            Self::EmptyTaps => write!(f, "DCLS forward pass requires at least one tap"),
            Self::MismatchedLengths { spikes, weights } => write!(
                f,
                "DCLS spike/weight length mismatch: spikes={spikes}, weights={weights}"
            ),
            Self::InvalidSigma { sigma_q88 } => {
                write!(f, "DCLS tent sigma must be positive, got {sigma_q88}")
            }
            Self::TapIndexOverflow { tap_index } => {
                write!(f, "DCLS tap index {tap_index} cannot be represented as Q8.8")
            }
        }
    }
}

impl Error for DclsError {}

impl DclsLayerConfig {
    fn validate(self) -> Result<(), DclsError> {
        if self.data_width != DEFAULT_DATA_WIDTH
            || self.fraction != DEFAULT_FRACTION
            || self.accumulator_width != DEFAULT_ACCUMULATOR_WIDTH
        {
            return Err(DclsError::UnsupportedFormat {
                data_width: self.data_width,
                fraction: self.fraction,
                accumulator_width: self.accumulator_width,
            });
        }
        Ok(())
    }
}

/// Return the Q8.8 triangular tent gate for a tap delay index.
pub fn tent_gate_q88(tap_index: usize, centre_q88: i16, sigma_q88: i16) -> Result<i16, DclsError> {
    if sigma_q88 <= 0 {
        return Err(DclsError::InvalidSigma { sigma_q88 });
    }
    let delay_q88 = i64::try_from(tap_index)
        .ok()
        .and_then(|index| index.checked_shl(DEFAULT_FRACTION))
        .ok_or(DclsError::TapIndexOverflow { tap_index })?;
    let centre = i64::from(centre_q88);
    let sigma = i64::from(sigma_q88);
    let distance = (delay_q88 - centre).abs();
    if distance >= sigma {
        return Ok(0);
    }
    let numerator = sigma - distance;
    let gate = (numerator << DEFAULT_FRACTION) / sigma;
    Ok(gate.clamp(0, Q88_ONE) as i16)
}

/// Execute the DCLS Q8.8 tent contraction with a Q16.16 accumulator.
pub fn dcls_max_forward_q88(
    spikes: &[u8],
    weights_q88: &[i16],
    centre_q88: i16,
    sigma_q88: i16,
) -> Result<DclsForwardResult, DclsError> {
    dcls_max_forward_q88_with_config(
        spikes,
        weights_q88,
        centre_q88,
        sigma_q88,
        DclsLayerConfig::default(),
    )
}

/// Execute the DCLS Q8.8 tent contraction with an explicit format contract.
pub fn dcls_max_forward_q88_with_config(
    spikes: &[u8],
    weights_q88: &[i16],
    centre_q88: i16,
    sigma_q88: i16,
    config: DclsLayerConfig,
) -> Result<DclsForwardResult, DclsError> {
    config.validate()?;
    if spikes.is_empty() {
        return Err(DclsError::EmptyTaps);
    }
    if spikes.len() != weights_q88.len() {
        return Err(DclsError::MismatchedLengths {
            spikes: spikes.len(),
            weights: weights_q88.len(),
        });
    }
    if sigma_q88 <= 0 {
        return Err(DclsError::InvalidSigma { sigma_q88 });
    }

    let mut accumulator = 0_i64;
    let mut active_tap_count = 0_usize;
    let mut max_gate_q88 = 0_i16;
    for (tap_index, (&spike, &weight)) in spikes.iter().zip(weights_q88.iter()).enumerate() {
        if spike == 0 {
            continue;
        }
        active_tap_count += 1;
        let gate = tent_gate_q88(tap_index, centre_q88, sigma_q88)?;
        max_gate_q88 = max_gate_q88.max(gate);
        accumulator += i64::from(weight) * i64::from(gate);
    }

    let (accumulator_q16_16, accumulator_overflow) = saturate_i32(accumulator);
    let (output_q88, output_overflow) = saturate_q88_output(accumulator);
    Ok(DclsForwardResult {
        output_q88,
        accumulator_q16_16,
        overflow: accumulator_overflow || output_overflow,
        active_tap_count,
        max_gate_q88,
    })
}

fn saturate_i32(value: i64) -> (i32, bool) {
    if value > I32_MAX_AS_I64 {
        (i32::MAX, true)
    } else if value < I32_MIN_AS_I64 {
        (i32::MIN, true)
    } else {
        (value as i32, false)
    }
}

fn saturate_q88_output(accumulator_q16_16: i64) -> (i16, bool) {
    if accumulator_q16_16 > I16_MAX_Q16_16 {
        (i16::MAX, true)
    } else if accumulator_q16_16 < I16_MIN_Q16_16 {
        (i16::MIN, true)
    } else {
        ((accumulator_q16_16 >> DEFAULT_FRACTION) as i16, false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tent_gate_matches_special_cases() {
        assert_eq!(tent_gate_q88(1, 256, 512).unwrap(), 256);
        assert_eq!(tent_gate_q88(0, 256, 512).unwrap(), 128);
        assert_eq!(tent_gate_q88(3, 256, 512).unwrap(), 0);
    }

    #[test]
    fn forward_matches_hand_computed_q16_16_accumulator() {
        let result = dcls_max_forward_q88(&[1, 1, 1], &[256, 128, -64], 256, 512).unwrap();
        assert_eq!(result.accumulator_q16_16, 57_344);
        assert_eq!(result.output_q88, 224);
        assert_eq!(result.active_tap_count, 3);
        assert_eq!(result.max_gate_q88, 256);
        assert!(!result.overflow);
    }

    #[test]
    fn zero_spike_taps_do_not_contribute() {
        let result = dcls_max_forward_q88(&[0, 1, 0], &[256, 128, -64], 256, 512).unwrap();
        assert_eq!(result.accumulator_q16_16, 32_768);
        assert_eq!(result.output_q88, 128);
        assert_eq!(result.active_tap_count, 1);
    }

    #[test]
    fn invalid_sigma_fails_closed() {
        assert_eq!(
            dcls_max_forward_q88(&[1], &[256], 0, 0).unwrap_err(),
            DclsError::InvalidSigma { sigma_q88: 0 }
        );
    }

    #[test]
    fn mismatched_taps_fail_closed() {
        assert_eq!(
            dcls_max_forward_q88(&[1, 1], &[256], 0, 256).unwrap_err(),
            DclsError::MismatchedLengths {
                spikes: 2,
                weights: 1
            }
        );
    }

    #[test]
    fn saturating_output_sets_overflow() {
        let spikes = vec![1_u8; 1024];
        let weights = vec![i16::MAX; 1024];
        let result = dcls_max_forward_q88(&spikes, &weights, 0, i16::MAX).unwrap();
        assert_eq!(result.output_q88, i16::MAX);
        assert!(result.overflow);
    }
}
