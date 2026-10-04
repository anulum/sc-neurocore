// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Retained fixed-point LIF unit contracts

use super::{mask, FixedPointLif};

#[test]
fn branchless_mask_matches_signed_reference() {
    for &width in &[16_u32, 32] {
        for value in [
            -32768_i32,
            -1,
            0,
            1,
            32767,
            65535,
            -65536,
            i16::MAX as i32,
            i16::MIN as i32,
        ] {
            let result = mask(value, width);
            let bit_mask = (1_i64 << width) - 1;
            let mut expected = (value as i64) & bit_mask;
            if expected >= (1_i64 << (width - 1)) {
                expected -= 1_i64 << width;
            }
            let expected = if width >= 32 {
                expected as i32 as i16
            } else {
                expected as i16
            };
            assert_eq!(result, expected, "value={value}, width={width}");
        }
    }
}

#[test]
fn refractory_period_enforces_two_silent_steps() {
    let mut neuron = FixedPointLif::new(16, 8, 0, 0, 256, 2);
    let spikes: Vec<_> = (0..30).map(|_| neuron.step(1, 256, 50, 0).0).collect();
    assert!(spikes.iter().sum::<i32>() > 0);
    for (index, &spike) in spikes.iter().enumerate() {
        if spike == 1 && index + 2 < spikes.len() {
            assert_eq!(spikes[index + 1], 0);
            assert_eq!(spikes[index + 2], 0);
        }
    }
}

#[test]
fn zero_refractory_period_allows_repeated_firing() {
    let mut neuron = FixedPointLif::new(16, 8, 0, 0, 256, 0);
    let spikes: i32 = (0..20).map(|_| neuron.step(1, 256, 50, 0).0).sum();
    assert!(spikes > 0);
}
