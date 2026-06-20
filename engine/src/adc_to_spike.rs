// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// Copyright (C) 2020-2026 Miroslav Sotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore - ADC-to-spike decimating rate-code reference (per-window)

//! Bit-true integer reference for the ADC-to-spike window rate-code encoder.

use std::error::Error;
use std::fmt;

/// Per-window outputs of the ADC-to-spike encoder.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdcSpikeWindowResult {
    /// Sign-aware averaged Q-format window codes.
    pub window_values_q: Vec<i32>,
    /// Deterministic per-window spike counts (`|window| / threshold`).
    pub spike_counts: Vec<i32>,
    /// `true` where the window code is negative.
    pub polarities: Vec<bool>,
}

/// ADC-to-spike contract errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AdcSpikeError {
    InvalidAdcWidth(u32),
    InvalidQFormat { q_int: u32, q_frac: u32 },
    InvalidDecimation(u32),
    InvalidThreshold(i64),
    TooFewSamples { samples: usize, decimation: u32 },
}

impl fmt::Display for AdcSpikeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidAdcWidth(width) => {
                write!(f, "adc_width must be greater than one, got {width}")
            }
            Self::InvalidQFormat { q_int, q_frac } => write!(
                f,
                "Q-format needs positive integer bits and non-negative fraction bits, got q_int={q_int}, q_frac={q_frac}"
            ),
            Self::InvalidDecimation(decimation) => {
                write!(f, "decimation must be positive, got {decimation}")
            }
            Self::InvalidThreshold(threshold) => {
                write!(f, "threshold_q must be positive, got {threshold}")
            }
            Self::TooFewSamples {
                samples,
                decimation,
            } => write!(
                f,
                "need at least decimation={decimation} samples, got {samples}"
            ),
        }
    }
}

impl Error for AdcSpikeError {}

/// Q-format code bounds for a `q_int`/`q_frac` signed fixed-point format.
fn q_bounds(q_int: u32, q_frac: u32) -> (i64, i64) {
    let q_total = q_int + q_frac;
    let half = 1_i64 << (q_total - 1);
    (-half, half - 1)
}

/// Centre and quantise one raw ADC sample to a Q-format code.
fn quantise_adc(sample: i64, adc_width: u32, q_int: u32, q_frac: u32, signed_input: bool) -> i64 {
    let q_total = q_int + q_frac;
    let (q_min, q_max) = q_bounds(q_int, q_frac);
    let centred = if signed_input {
        let sign_bit = 1_i64 << (adc_width - 1);
        let mask = (1_i64 << adc_width) - 1;
        let masked = sample & mask;
        if masked & sign_bit != 0 {
            masked - (1_i64 << adc_width)
        } else {
            masked
        }
    } else {
        sample - (1_i64 << (adc_width - 1))
    };

    let rounded = if q_total > adc_width {
        centred << (q_total - adc_width)
    } else if adc_width > q_total {
        let shift = adc_width - q_total;
        let half = 1_i64 << (shift - 1);
        if centred >= 0 {
            (centred + half) >> shift
        } else {
            (centred - half) >> shift
        }
    } else {
        centred
    };
    rounded.clamp(q_min, q_max)
}

/// Sign-aware round-then-truncate window average (truncation toward zero).
fn average_window(total_q: i64, decimation: u32, q_min: i64, q_max: i64) -> i64 {
    let half = i64::from(decimation / 2);
    let adjusted = if total_q >= 0 {
        total_q + half
    } else {
        total_q - half
    };
    // Integer division truncates toward zero, matching `int(adjusted / decimation)`.
    let averaged = adjusted / i64::from(decimation);
    averaged.clamp(q_min, q_max)
}

/// Encode raw ADC samples into per-window spike rate codes.
///
/// Consumes the first `samples.len() / decimation` complete windows. Each window
/// is quantised sample-by-sample, sign-aware averaged, and converted into a spike
/// count of `|window| / threshold` with the window sign as polarity.
#[allow(clippy::too_many_arguments)]
pub fn adc_to_spike_windows(
    samples: &[i64],
    adc_width: u32,
    q_int: u32,
    q_frac: u32,
    decimation: u32,
    signed_input: bool,
    threshold_q: i64,
) -> Result<AdcSpikeWindowResult, AdcSpikeError> {
    if adc_width <= 1 {
        return Err(AdcSpikeError::InvalidAdcWidth(adc_width));
    }
    if q_int == 0 {
        return Err(AdcSpikeError::InvalidQFormat { q_int, q_frac });
    }
    if decimation == 0 {
        return Err(AdcSpikeError::InvalidDecimation(decimation));
    }
    if threshold_q <= 0 {
        return Err(AdcSpikeError::InvalidThreshold(threshold_q));
    }
    let decim = decimation as usize;
    let n_windows = samples.len() / decim;
    if n_windows == 0 {
        return Err(AdcSpikeError::TooFewSamples {
            samples: samples.len(),
            decimation,
        });
    }

    let (q_min, q_max) = q_bounds(q_int, q_frac);
    let mut window_values_q = Vec::with_capacity(n_windows);
    let mut spike_counts = Vec::with_capacity(n_windows);
    let mut polarities = Vec::with_capacity(n_windows);
    for window in 0..n_windows {
        let base = window * decim;
        let mut total: i64 = 0;
        for offset in 0..decim {
            total += quantise_adc(
                samples[base + offset],
                adc_width,
                q_int,
                q_frac,
                signed_input,
            );
        }
        let window_q = average_window(total, decimation, q_min, q_max);
        window_values_q.push(window_q as i32);
        spike_counts.push((window_q.abs() / threshold_q) as i32);
        polarities.push(window_q < 0);
    }
    Ok(AdcSpikeWindowResult {
        window_values_q,
        spike_counts,
        polarities,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantise_equal_width_is_identity_after_centring() {
        // adc_width == q_total (16): signed centring then no rescale.
        assert_eq!(quantise_adc(0, 16, 8, 8, true), 0);
        assert_eq!(quantise_adc(1, 16, 8, 8, true), 1);
        assert_eq!(quantise_adc((1 << 16) - 1, 16, 8, 8, true), -1);
    }

    #[test]
    fn average_truncates_toward_zero() {
        // total -7, decimation 8 -> adjusted -7-4 = -11 -> -11/8 = -1 (toward zero).
        assert_eq!(average_window(-7, 8, -32768, 32767), -1);
        assert_eq!(average_window(7, 8, -32768, 32767), 1);
    }

    #[test]
    fn windows_emit_rate_code_and_polarity() {
        // Eight mid-scale-negative samples -> negative window -> polarity set.
        let samples = vec![0_i64; 8];
        let result = adc_to_spike_windows(&samples, 16, 8, 8, 8, false, 256).unwrap();
        assert_eq!(result.window_values_q.len(), 1);
        // offset-binary 0 -> centred -(1<<15) = -32768 -> averaged -32768 -> 128 spikes.
        assert_eq!(result.window_values_q[0], -32768);
        assert_eq!(result.spike_counts[0], 128);
        assert!(result.polarities[0]);
    }

    #[test]
    fn rejects_bad_config_and_short_streams() {
        assert_eq!(
            adc_to_spike_windows(&[0; 8], 1, 8, 8, 8, true, 256).unwrap_err(),
            AdcSpikeError::InvalidAdcWidth(1)
        );
        assert_eq!(
            adc_to_spike_windows(&[0; 8], 16, 0, 8, 8, true, 256).unwrap_err(),
            AdcSpikeError::InvalidQFormat {
                q_int: 0,
                q_frac: 8
            }
        );
        assert_eq!(
            adc_to_spike_windows(&[0; 8], 16, 8, 8, 0, true, 256).unwrap_err(),
            AdcSpikeError::InvalidDecimation(0)
        );
        assert_eq!(
            adc_to_spike_windows(&[0; 8], 16, 8, 8, 8, true, 0).unwrap_err(),
            AdcSpikeError::InvalidThreshold(0)
        );
        assert_eq!(
            adc_to_spike_windows(&[0; 3], 16, 8, 8, 8, true, 256).unwrap_err(),
            AdcSpikeError::TooFewSamples {
                samples: 3,
                decimation: 8
            }
        );
    }
}
