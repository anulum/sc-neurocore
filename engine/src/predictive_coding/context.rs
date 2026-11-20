// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Native context predictor for binary rasters

//! Encode and recover a binary raster using per-channel context counts.

use std::collections::HashMap;

/// Encode a row-major binary raster and count its correct initial predictions.
pub(crate) fn predict_and_xor_context(
    spikes: &[i8],
    channels: usize,
    history_bits: u8,
) -> Result<(Vec<i8>, usize), &'static str> {
    transform(spikes, channels, history_bits, false)
}

/// Recover the original binary raster from row-major context prediction errors.
pub(crate) fn xor_and_recover_context(
    errors: &[i8],
    channels: usize,
    history_bits: u8,
) -> Result<Vec<i8>, &'static str> {
    transform(errors, channels, history_bits, true).map(|(raster, _)| raster)
}

#[derive(Default)]
struct Channel {
    history: u64,
    outcomes: HashMap<u64, (usize, usize)>,
}

fn transform(
    input: &[i8],
    channels: usize,
    history_bits: u8,
    recovering: bool,
) -> Result<(Vec<i8>, usize), &'static str> {
    if history_bits > 63 {
        return Err("context history length must be between 0 and 63");
    }
    if input.is_empty() {
        return Ok((Vec::new(), 0));
    }
    if channels == 0 || !input.len().is_multiple_of(channels) {
        return Err("context raster must contain complete channel rows");
    }
    if input.iter().any(|&bit| !matches!(bit, 0 | 1)) {
        return Err("context raster samples must be 0 or 1");
    }

    let mask = (1u64 << history_bits) - 1;
    let mut histories: Vec<Channel> = (0..channels).map(|_| Channel::default()).collect();
    let mut raster = Vec::with_capacity(input.len());
    let mut correct = 0;
    for row in input.chunks_exact(channels) {
        for (&bit, channel) in row.iter().zip(&mut histories) {
            let (zeroes, ones) = channel.outcomes.entry(channel.history).or_default();
            let predicted = i8::from(*ones > *zeroes);
            let transformed = bit ^ predicted;
            raster.push(transformed);
            let actual = if recovering { transformed } else { bit };
            if actual == 1 {
                *ones += 1;
            } else {
                *zeroes += 1;
            }
            correct += usize::from(predicted == actual);
            channel.history = ((channel.history << 1) | actual as u64) & mask;
        }
    }
    Ok((raster, correct))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn context_counts_ties_and_state_are_symmetric() {
        let input = [1, 1, 0, 1, 0, 0];
        let (errors, correct) = predict_and_xor_context(&input, 1, 1).unwrap();
        assert_eq!(errors, [1, 1, 1, 0, 0, 1]);
        assert_eq!(correct, 2);
        assert_eq!(xor_and_recover_context(&errors, 1, 1).unwrap(), input);
        assert_eq!(predict_and_xor_context(&input, 1, 1).unwrap().0, errors);
    }

    #[test]
    fn context_keeps_independent_channels_at_both_history_bounds() {
        let input = [1, 0, 1, 0, 0, 1, 1, 1, 0, 0];
        for bits in [0, 63] {
            let (errors, _) = predict_and_xor_context(&input, 2, bits).unwrap();
            assert_eq!(xor_and_recover_context(&errors, 2, bits).unwrap(), input);
        }
    }

    #[test]
    fn context_admits_empty_rasters_and_refuses_invalid_shape_or_values() {
        assert_eq!(predict_and_xor_context(&[], 0, 0).unwrap(), (vec![], 0));
        assert!(xor_and_recover_context(&[], 3, 8).unwrap().is_empty());
        for (bits, channels, values) in [
            (64, 2, &[0, 1][..]),
            (2, 0, &[1]),
            (2, 2, &[1]),
            (2, 1, &[2]),
        ] {
            assert!(predict_and_xor_context(values, channels, bits).is_err());
            assert!(xor_and_recover_context(values, channels, bits).is_err());
        }
    }
}
