// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Streaming lossless waveform coder tests

//! Exact round trips across window sizes and groups, ADC detection, refusals and packet checks.

use sc_neurocore_engine::codecs::waveform_stream::{
    detect_adc_bits, level_of, nominal_value, StreamDecoder, StreamEncoder,
};

/// Deterministic pseudo-random integers in `[low, high]` (64-bit LCG).
fn lcg(seed: u64, count: usize, low: i64, high: i64) -> Vec<i64> {
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

/// A 10-bit ADC style signal: slow oscillation plus noise, scaled to int16 by 65535/1023.
fn adc_stream(samples: usize, channels: usize, seed: u64) -> Vec<i16> {
    let noise = lcg(seed, samples * channels, -40, 40);
    (0..samples * channels)
        .map(|i| {
            let (t, c) = (i / channels, i % channels);
            let level = 512.0 + 300.0 * ((t as f64) / (40.0 + c as f64)).sin() + noise[i] as f64;
            nominal_value(level.round().clamp(0.0, 1023.0) as i64, 10) as i16
        })
        .collect()
}

fn round_trip(
    data: &[i16],
    samples: usize,
    channels: usize,
    bits: u32,
    groups: usize,
    window: usize,
) -> usize {
    let mut encoder = StreamEncoder::new(channels, bits, groups).unwrap();
    let mut decoder = StreamDecoder::new(channels, bits, groups).unwrap();
    let mut total = 0;
    let mut start = 0;
    while start < samples {
        let rows = window.min(samples - start);
        let slice = &data[start * channels..(start + rows) * channels];
        let packet = encoder.encode(slice, rows).unwrap();
        total += packet.len();
        assert_eq!(decoder.decode(&packet, rows).unwrap(), slice);
        start += rows;
    }
    total
}

#[test]
fn level_mapping_round_trips_every_level() {
    for bits in [1_u32, 10, 12, 16] {
        for level in 0..(1_i64 << bits) {
            assert_eq!(
                level_of(nominal_value(level, bits), bits),
                level,
                "bits {bits} level {level}"
            );
        }
    }
    assert_eq!(level_of(-32768, 16), 0);
    assert_eq!(level_of(32767, 10), 1023);
}

#[test]
fn every_window_size_and_group_count_decodes_exactly() {
    let (samples, channels) = (2000, 12);
    let data = adc_stream(samples, channels, 1);
    let whole = round_trip(&data, samples, channels, 10, 1, samples);
    for window in [1, 7, 20, 333] {
        for groups in [1, 3, 12] {
            let size = round_trip(&data, samples, channels, 10, groups, window);
            // Each packet costs about 9 bytes per group (length and range-coder flush), so
            // only windows of about 1 ms (20 samples) and longer are required to compress.
            if window >= 20 {
                assert!(
                    size < data.len() * 2,
                    "window {window} groups {groups} did not compress"
                );
            }
        }
    }
    let one_ms = round_trip(&data, samples, channels, 10, 1, 20);
    assert!(
        (one_ms as f64) < 1.1 * whole as f64,
        "1 ms windows cost {one_ms} vs {whole}"
    );
}

/// Narrow noise around ADC level 500, which is a missing code: it becomes 501, except every
/// `rare`-th sample, which keeps it (a sample on an excluded level).
fn missing_code_stream(samples: usize, channels: usize, rare: usize) -> Vec<i16> {
    let noise = lcg(9, samples * channels, -3, 3);
    (0..samples * channels)
        .map(|i| {
            let t = (i / channels) as f64;
            let mut level = 500 + (2.0 * (t / 50.0).sin()).round() as i64 + noise[i];
            if level == 500 && i % rare != 0 {
                level = 501;
            }
            nominal_value(level, 10) as i16
        })
        .collect()
}

#[test]
fn a_missing_code_costs_almost_nothing_and_rare_hits_decode() {
    let (samples, channels) = (6000, 4);
    let gap = missing_code_stream(samples, channels, usize::MAX);
    // The same signal without the gap: every level above 500 moves down by one.
    let dense: Vec<i16> = gap
        .iter()
        .map(|&value| {
            let level = level_of(i64::from(value), 10);
            nominal_value(if level > 500 { level - 1 } else { level }, 10) as i16
        })
        .collect();
    let gap_size = round_trip(&gap, samples, channels, 10, 1, samples);
    let dense_size = round_trip(&dense, samples, channels, 10, 1, samples);
    assert!(
        (gap_size as f64) < 1.01 * dense_size as f64,
        "missing code {gap_size} vs dense {dense_size}"
    );
    let hits = missing_code_stream(samples, channels, 97);
    for (window, groups) in [(samples, 1), (20, 2), (1, 4)] {
        round_trip(&hits, samples, channels, 10, groups, window);
    }
}

#[test]
fn level_counts_halve_on_long_streams() {
    // A near-constant channel: level 512 (85 % of 80 000 samples) passes the count limit
    // of 65536 occurrences, so every count is halved once.
    let samples = 80_000;
    let noise = lcg(10, samples, 0, 19);
    let data: Vec<i16> = noise
        .iter()
        .map(|&n| {
            let level = match n {
                0 => 511,
                1 => 513,
                2 => 514,
                _ => 512,
            };
            nominal_value(level, 10) as i16
        })
        .collect();
    round_trip(&data, samples, 1, 10, 1, 8000);
}

#[test]
fn full_range_noise_and_extremes_decode_at_sixteen_bits() {
    let noise: Vec<i16> = lcg(2, 3000, -32768, 32767)
        .into_iter()
        .map(|v| v as i16)
        .collect();
    round_trip(&noise, 1000, 3, 16, 1, 1000);
    let mut extremes = vec![0_i16; 600];
    for (i, value) in extremes.iter_mut().enumerate() {
        *value = if i % 2 == 0 { 32767 } else { -32768 };
    }
    round_trip(&extremes, 600, 1, 16, 1, 50);
    round_trip(&[5], 1, 1, 16, 1, 1);
}

#[test]
fn groups_are_deterministic_and_independent_of_threads() {
    let data = adc_stream(500, 64, 3);
    let packet = |groups| {
        StreamEncoder::new(64, 10, groups)
            .unwrap()
            .encode(&data, 500)
            .unwrap()
    };
    assert_eq!(packet(8), packet(8));
    assert_ne!(packet(1), packet(8));
}

#[test]
fn adc_width_detection() {
    assert_eq!(detect_adc_bits(&adc_stream(400, 4, 4)), 10);
    let noise: Vec<i16> = lcg(5, 4000, -32768, 32767)
        .into_iter()
        .map(|v| v as i16)
        .collect();
    assert_eq!(detect_adc_bits(&noise), 16);
    assert_eq!(detect_adc_bits(&[0, 0, 0]), 1);
}

#[test]
fn a_contradicting_sample_is_refused_and_leaves_the_encoder_unchanged() {
    let data = adc_stream(100, 2, 6);
    let mut encoder = StreamEncoder::new(2, 10, 1).unwrap();
    let mut reference = StreamEncoder::new(2, 10, 1).unwrap();
    encoder.encode(&data, 100).unwrap();
    reference.encode(&data, 100).unwrap();
    let mut bad = data[..20].to_vec();
    bad[0] = data[0] + 1;
    let error = encoder.encode(&bad, 10).unwrap_err();
    assert!(error.contains("contradicts"), "{error}");
    assert_eq!(
        encoder.encode(&data[..20], 10).unwrap(),
        reference.encode(&data[..20], 10).unwrap()
    );
}

#[test]
fn configuration_and_shape_refusals() {
    assert!(StreamEncoder::new(0, 10, 1).is_err());
    assert!(StreamEncoder::new(4, 0, 1).is_err());
    assert!(StreamEncoder::new(4, 17, 1).is_err());
    assert!(StreamEncoder::new(4, 10, 5).is_err());
    assert!(StreamDecoder::new(4, 10, 0).is_err());
    let mut encoder = StreamEncoder::new(4, 10, 1).unwrap();
    assert!(encoder
        .encode(&[0; 7], 2)
        .unwrap_err()
        .contains("window size"));
}

#[test]
fn corrupted_packets_are_refused_and_leave_the_decoder_unchanged() {
    let data = adc_stream(300, 4, 7);
    let mut encoder = StreamEncoder::new(4, 10, 2).unwrap();
    let packet = encoder.encode(&data, 300).unwrap();
    let fresh = || StreamDecoder::new(4, 10, 2).unwrap();
    assert!(fresh()
        .decode(&packet[..3], 300)
        .unwrap_err()
        .contains("missing group length"));
    assert!(fresh()
        .decode(&packet[..packet.len() - 1], 300)
        .unwrap_err()
        .contains("longer than packet"));
    let mut extra = packet.clone();
    extra.push(0);
    assert!(fresh()
        .decode(&extra, 300)
        .unwrap_err()
        .contains("trailing bytes"));
    let mut bad_start = packet.clone();
    bad_start[4] = 1;
    assert!(fresh()
        .decode(&bad_start, 300)
        .unwrap_err()
        .contains("range-coder start"));
    assert!(
        fresh().decode(&packet, 301).is_err(),
        "asking for more samples than coded must fail"
    );
    let mut refused = 0;
    for position in 9..packet.len() {
        let mut flipped = packet.clone();
        flipped[position] ^= 0x5A;
        let mut decoder = fresh();
        // A flip either fails a check or yields some window; the last flush bytes may not
        // influence any decision, so an identical window is possible and harmless.
        if decoder.decode(&flipped, 300).is_err() {
            refused += 1;
        }
    }
    assert!(refused > 0);
    let mut decoder = fresh();
    assert!(decoder.decode(&packet[..3], 300).is_err());
    assert_eq!(
        decoder.decode(&packet, 300).unwrap(),
        data,
        "a refused packet leaves the state unchanged"
    );
}

#[test]
fn content_corruption_breaks_the_decoder_until_restarted() {
    let data = adc_stream(400, 2, 8);
    let mut encoder = StreamEncoder::new(2, 10, 1).unwrap();
    let first = encoder.encode(&data[..400], 200).unwrap();
    let second = encoder.encode(&data[400..], 200).unwrap();
    let mut decoder = StreamDecoder::new(2, 10, 1).unwrap();
    assert!(
        decoder.decode(&first, 201).is_err(),
        "asking for more samples than coded fails"
    );
    let error = decoder.decode(&first, 200).unwrap_err();
    assert!(error.contains("start a new decoder"), "{error}");
    let mut fresh = StreamDecoder::new(2, 10, 1).unwrap();
    assert_eq!(fresh.decode(&first, 200).unwrap(), &data[..400]);
    assert_eq!(fresh.decode(&second, 200).unwrap(), &data[400..]);
}
