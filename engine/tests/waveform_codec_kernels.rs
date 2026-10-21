// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — WaveformCodec kernel tests

//! Exact summation, lossless round trips, Rice refusals, templates and background kernels.

use sc_neurocore_engine::codecs::waveform_codec::{
    detect_spikes, exact_dot, exact_sum, extract_background, lossless_decode_channel,
    lossless_encode_channel, noise_thresholds, quantize_background, reconstruct_background,
    rice_decode, rice_encode, template_match, LOSSLESS_MAX_RICE_K,
};

/// Deterministic pseudo-random integers in `[low, high]` (64-bit LCG).
fn lcg_values(seed: u64, count: usize, low: i64, high: i64) -> Vec<i64> {
    let mut state = seed;
    (0..count)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            low + ((state >> 33) % ((high - low + 1) as u64)) as i64
        })
        .collect()
}

#[test]
fn exact_sum_matches_correctly_rounded_references() {
    assert_eq!(exact_sum([1e16, 1.0, -1e16]), 1.0);
    assert_eq!(exact_sum(std::iter::repeat_n(0.1, 10)), 1.0);
    assert_eq!(exact_sum([1e16, 1.0, 1e-16]), 10_000_000_000_000_002.0);
    assert_eq!(exact_sum(std::iter::empty()), 0.0);
    assert_eq!(exact_dot(&[3.0, 4.0], &[3.0, 4.0]), 25.0);
}

#[test]
fn lossless_round_trips_every_mapping_and_order() {
    let full_range = lcg_values(1, 3000, -32768, 32767);
    let lattice: Vec<i64> = lcg_values(2, 3000, -500, 500)
        .iter()
        .map(|v| v * 4 + 3)
        .collect();
    let scaled: Vec<i64> = (0..4000)
        .map(|t| ((300.0 * (f64::from(t) / 50.0).sin()).round() * 64.0626).round() as i64)
        .collect();
    let ramp: Vec<i64> = (0..4000_i64).map(|t| t * t / 1000 - 8000).collect();
    let mut escapes = vec![0_i64; 600];
    for (i, value) in escapes.iter_mut().enumerate().skip(300) {
        *value = if i % 2 == 0 { 32767 } else { -32768 };
    }
    for samples in [
        full_range,
        lattice,
        scaled,
        ramp,
        escapes,
        vec![5],
        vec![7, -7],
        vec![-32768; 9],
    ] {
        let section = lossless_encode_channel(&samples);
        assert_eq!(
            lossless_decode_channel(&section, samples.len()).unwrap(),
            samples
        );
    }
}

#[test]
fn lossless_mapping_choice_follows_section_length() {
    let lattice: Vec<i64> = lcg_values(3, 5000, -1000, 1000)
        .iter()
        .map(|v| v * 4 + 3)
        .collect();
    assert_eq!(
        lossless_encode_channel(&lattice)[0],
        0,
        "dense lattice uses the affine mapping"
    );
    let sparse: Vec<i64> = (0..600)
        .map(|t| ((t % 37) as f64 * 64.06).round() as i64)
        .collect();
    assert_eq!(
        lossless_encode_channel(&sparse)[0],
        1,
        "sparse values use the value table"
    );
}

#[test]
fn lossless_decoder_refuses_corrupted_sections() {
    let samples: Vec<i64> = (0..600)
        .map(|t| ((t % 37) as f64 * 64.06).round() as i64)
        .collect();
    let section = lossless_encode_channel(&samples);
    let mut bad_kind = section.clone();
    bad_kind[0] = 7;
    assert!(lossless_decode_channel(&bad_kind, 600)
        .unwrap_err()
        .contains("level mapping"));
    let mut huge_table = section.clone();
    huge_table[1..5].copy_from_slice(&1_000_000_u32.to_be_bytes());
    assert!(lossless_decode_channel(&huge_table, 600)
        .unwrap_err()
        .contains("value table length"));
    assert!(lossless_decode_channel(&section[..section.len() - 1], 600).is_err());
    assert!(lossless_decode_channel(&[], 4)
        .unwrap_err()
        .contains("empty"));

    let mut affine = vec![0_u8];
    affine.extend_from_slice(&32767_i32.to_be_bytes());
    affine.extend_from_slice(&1_u32.to_be_bytes());
    affine.push(0);
    affine.extend(rice_encode(&[0, 2]));
    assert!(lossless_decode_channel(&affine, 2)
        .unwrap_err()
        .contains("int16 range"));
    let mut order = affine.clone();
    order[9] = 9;
    assert!(lossless_decode_channel(&order, 2)
        .unwrap_err()
        .contains("predictor order"));

    let mut table = vec![1_u8];
    table.extend_from_slice(&1_u32.to_be_bytes());
    table.push(0);
    table.push(0);
    table.extend(rice_encode(&[0, 6]));
    assert!(lossless_decode_channel(&table, 2)
        .unwrap_err()
        .contains("outside the value table"));
}

#[test]
fn rice_round_trip_and_consistency_checks() {
    let values = [0_u64, 1, 5, 100, 3, 1 << 20];
    let encoded = rice_encode(&values);
    assert_eq!(rice_decode(&encoded, values.len()).unwrap(), values);
    assert!(rice_decode(&encoded, values.len() - 1)
        .unwrap_err()
        .contains("number of values"));
    let mut extra = encoded.clone();
    extra.push(0);
    assert!(rice_decode(&extra, values.len())
        .unwrap_err()
        .contains("wrong length"));
    let mut bad_k = encoded.clone();
    bad_k[12] = (LOSSLESS_MAX_RICE_K + 1) as u8;
    assert!(rice_decode(&bad_k, values.len())
        .unwrap_err()
        .contains("out of range"));
    assert!(rice_decode(&encoded[..8], values.len())
        .unwrap_err()
        .contains("too short"));
}

#[test]
fn templates_scales_and_fallback() {
    let first = [1.0_f32, -1.0, 0.0, 0.0];
    let second = [0.0_f32, 0.0, 1.0, -1.0];
    let scaled = [2.0_f32, -2.0, 0.0, 0.0];
    let snippets: Vec<f32> = first
        .iter()
        .chain(&second)
        .chain(&scaled)
        .copied()
        .collect();
    let result = template_match(&snippets, 3, 4, 4, 0.9);
    assert_eq!(result.template_sources, vec![0, 1]);
    assert_eq!(result.template_ids, vec![0, 1, 0]);
    assert_eq!(result.scales, vec![1.0, 1.0, 2.0]);
    assert!(result.residuals.iter().all(|r| r.abs() < 1e-6));

    let full: Vec<f32> = first.iter().chain(&[0.0; 4]).copied().collect();
    let fallback = template_match(&full, 2, 4, 1, 0.9);
    assert_eq!(fallback.template_ids, vec![0, 0]);
    assert_eq!(&fallback.residuals[4..], &[-1.0, 1.0, 0.0, 0.0]);
    assert!(template_match(&[], 0, 4, 4, 0.9).template_ids.is_empty());
}

#[test]
fn closed_loop_background_stays_within_one_step() {
    let rows = 400;
    let background: Vec<f32> = (0..rows)
        .flat_map(|t| {
            let value = 1500.0 + 200.0 * (t as f32 / 40.0).sin();
            [value, value - 300.0]
        })
        .collect();
    let largest = background
        .chunks(2)
        .zip(background.chunks(2).skip(1))
        .map(|(a, b)| (b[0] - a[0]).abs())
        .fold(0.0_f32, f32::max);
    let step = largest / 127.0;
    let codes = quantize_background(&background, rows, 2, step, 128);
    let recon = reconstruct_background(&codes, rows, 2, &background[..2], step);
    let error = recon
        .iter()
        .zip(&background)
        .map(|(r, b)| (r - b).abs())
        .fold(0.0_f32, f32::max);
    assert!(error <= step, "error {error} above one step {step}");
    assert!(quantize_background(&[], 0, 2, 1.0, 4).is_empty());
    assert!(reconstruct_background(&[], 0, 2, &[0.0, 0.0], 1.0).is_empty());
}

#[test]
fn detection_respects_threshold_slope_and_refractory() {
    let mut waveform = vec![0.0_f32; 40];
    for &t in &[5_usize, 7, 20] {
        waveform[t] = -10.0;
    }
    let times = detect_spikes(&waveform, 40, 1, &[-5.0], 3);
    assert_eq!(
        times,
        vec![vec![5, 20]],
        "7 lies inside the refractory period"
    );
    let mut flat = vec![-10.0_f32; 8];
    flat[0] = 0.0;
    assert_eq!(
        detect_spikes(&flat, 8, 1, &[-5.0], 0),
        vec![vec![1]],
        "a non-falling sample is no spike"
    );
}

#[test]
fn noise_thresholds_use_the_median_absolute_sample() {
    let odd = [3.0_f32, -1.0, 2.0, -5.0, 4.0];
    assert_eq!(
        noise_thresholds(&odd, 5, 1, -2.0),
        vec![-2.0 * (3.0 / 0.6745_f32)]
    );
    let even = [1.0_f32, -2.0, 3.0, -4.0];
    assert_eq!(
        noise_thresholds(&even, 4, 1, -1.0),
        vec![-(2.5_f32 / 0.6745_f32)]
    );
    assert_eq!(noise_thresholds(&[0.0, 0.0], 2, 1, -1.0), vec![-1e-6_f32]);
    let two_channels = [1.0_f32, 10.0, -3.0, 30.0, 2.0, -20.0];
    let thresholds = noise_thresholds(&two_channels, 3, 2, -1.0);
    assert_eq!(thresholds, vec![-(2.0 / 0.6745_f32), -(20.0 / 0.6745_f32)]);
}

#[test]
fn background_zeroes_spike_windows_and_averages_sixteen_rows() {
    let t = 40;
    let waveform: Vec<f32> = (0..t).map(|row| row as f32).collect();
    let (rows, averaged) = extract_background(&waveform, t, 1, &[vec![2]], 3);
    assert_eq!(rows, 2);
    let first_block: f32 = (5..16).map(|row| row as f32).sum::<f32>() / 16.0;
    assert_eq!(
        averaged[0], first_block,
        "rows 0..5 lie in the spike window"
    );
    assert_eq!(
        averaged[1],
        (16..32).map(|row| row as f32).sum::<f32>() / 16.0
    );
    let (short_rows, short) = extract_background(&[1.0, 2.0, 3.0], 3, 1, &[vec![]], 3);
    assert_eq!((short_rows, short), (3, vec![1.0, 2.0, 3.0]));
}
