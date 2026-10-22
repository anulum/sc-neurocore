// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Spike-event layer tests

//! Detection of known spikes, window and group independence, filter design, refusals.

use sc_neurocore_engine::codecs::spike_events::{
    highpass, lowpass, threshold_factor, SpikeEventConfig, SpikeEventDecoder, SpikeEventEncoder,
    SpikeEventWindow,
};

const RATE: f64 = 19531.0;

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

/// Noise (sum of four uniform draws, about 1150 units RMS) with a slow drift, plus negative
/// spikes of -12000 at `spikes[c]` for channel `c`.
fn recording(samples: usize, channels: usize, spikes: &[Vec<usize>]) -> Vec<i16> {
    let noise: Vec<Vec<i64>> = (0..4)
        .map(|k| lcg(11 + k, samples * channels, -1000, 1000))
        .collect();
    let mut data: Vec<i64> = (0..samples * channels)
        .map(|i| {
            let t = (i / channels) as f64;
            noise.iter().map(|n| n[i]).sum::<i64>() + (3000.0 * (t / 3000.0).sin()) as i64
        })
        .collect();
    let shape = [-3000, -9000, -12000, -7000, -2000, 1500, 2500, 2000, 1000];
    for (channel, times) in spikes.iter().enumerate() {
        for &t in times {
            for (k, v) in shape.iter().enumerate() {
                data[(t + k) * channels + channel] += v;
            }
        }
    }
    data.into_iter()
        .map(|v| v.clamp(-32768, 32767) as i16)
        .collect()
}

fn config(
    channels: usize,
    sample_timing: bool,
    power_bins: usize,
    groups: usize,
) -> SpikeEventConfig {
    SpikeEventConfig {
        channels,
        sample_rate_hz: RATE,
        bin_samples: 20,
        refractory_samples: 20,
        threshold_sigma: 4.5,
        highpass_hz: 250.0,
        lowpass_hz: 1000.0,
        warmup_samples: 1953,
        sample_timing,
        power_bins,
        power_shift: 0,
        groups,
    }
}

/// Encode in windows of `window` samples and decode; return all windows.
fn run(
    data: &[i16],
    samples: usize,
    config: SpikeEventConfig,
    window: usize,
) -> Vec<SpikeEventWindow> {
    let channels = config.channels;
    let mut encoder = SpikeEventEncoder::new(config.clone()).unwrap();
    let mut decoder = SpikeEventDecoder::new(
        channels,
        config.bin_samples,
        config.sample_timing,
        config.power_bins,
        config.groups,
    )
    .unwrap();
    let mut out = Vec::new();
    let mut start = 0;
    while start < samples {
        let rows = window.min(samples - start);
        let packet = encoder
            .encode(&data[start * channels..(start + rows) * channels], rows)
            .unwrap();
        out.push(decoder.decode(&packet, rows).unwrap());
        start += rows;
    }
    out
}

/// Event sample times per channel (bin start plus offset, or bin start without offsets).
fn event_times(windows: &[SpikeEventWindow], channels: usize, bin: u64) -> Vec<Vec<u64>> {
    let mut times = vec![Vec::new(); channels];
    for w in windows {
        for (i, &event) in w.events.iter().enumerate() {
            if event == 1 {
                let (row, channel) = (i / channels, i % channels);
                let offset = if w.offsets.is_empty() {
                    0
                } else {
                    w.offsets[i] as u64
                };
                times[channel].push((w.first_bin + row as u64) * bin + offset);
            }
        }
    }
    times
}

#[test]
fn filter_design_matches_the_documented_values() {
    let h = highpass(250.0, RATE).unwrap();
    assert_eq!(
        [h.b0, h.b1, h.b2, h.a1, h.a2],
        [15478, -30956, 15478, -30906, 14623]
    );
    let l = lowpass(1000.0, RATE).unwrap();
    assert_eq!(
        [l.b0, l.b1, l.b2, l.a1, l.a2],
        [343, 687, 343, -25407, 10397]
    );
    assert_eq!(threshold_factor(4.5), 27327);
    for (cutoff, rate) in [
        (10.0, RATE),
        (9000.0, RATE),
        (f64::NAN, RATE),
        (250.0, f64::INFINITY),
    ] {
        assert!(highpass(cutoff, rate).is_err(), "{cutoff} {rate}");
    }
    // Near rate/1000 the low-pass gain rounds to zero in Q14.
    let error = lowpass(20.0, RATE).unwrap_err();
    assert!(error.contains("cannot be represented"), "{error}");
}

#[test]
fn large_spikes_are_found_once_each_at_their_samples() {
    let spikes = vec![
        vec![3000, 3500, 9000, 15000],
        vec![],
        vec![4000, 4030, 12000],
    ];
    let data = recording(20000, 3, &spikes);
    let times = event_times(&run(&data, 20000, config(3, true, 0, 1), 20000), 3, 20);
    for (channel, expected) in spikes.iter().enumerate() {
        assert_eq!(
            times[channel].len(),
            expected.len(),
            "channel {channel}: {:?}",
            times[channel]
        );
        for (&found, &spike) in times[channel].iter().zip(expected) {
            assert!(
                found >= spike as u64 && found <= spike as u64 + 3,
                "{found} vs {spike}"
            );
        }
    }
}

#[test]
fn spikes_inside_the_warmup_or_refractory_period_are_not_reported() {
    let data = recording(6000, 1, &[vec![500, 3000, 3010]]);
    let times = event_times(&run(&data, 6000, config(1, true, 0, 1), 6000), 1, 20);
    assert_eq!(times[0].len(), 1, "{:?}", times[0]);
    assert!(times[0][0] >= 3000 && times[0][0] <= 3003);
}

#[test]
fn windows_and_groups_do_not_change_the_result() {
    let spikes = vec![
        vec![2500, 7000],
        vec![3100],
        vec![],
        vec![5000, 9000],
        vec![4444],
    ];
    let data = recording(12000, 5, &spikes);
    let reference = run(&data, 12000, config(5, true, 7, 1), 12000);
    let flat = |windows: &[SpikeEventWindow]| {
        let events: Vec<u8> = windows.iter().flat_map(|w| w.events.clone()).collect();
        let offsets: Vec<i16> = windows.iter().flat_map(|w| w.offsets.clone()).collect();
        let power: Vec<i32> = windows.iter().flat_map(|w| w.power.clone()).collect();
        (events, offsets, power)
    };
    let expected = flat(&reference);
    assert_eq!(expected.0.len(), 600 * 5);
    assert_eq!(expected.2.len(), 85 * 5);
    for window in [1, 13, 20, 137, 4000] {
        for groups in [1, 2, 5] {
            let windows = run(&data, 12000, config(5, true, 7, groups), window);
            assert_eq!(flat(&windows), expected, "window {window} groups {groups}");
        }
    }
    // Band power follows the noise: a channel with half the noise has lower power.
    let quiet: Vec<i16> = data.iter().map(|&v| v / 2).collect();
    let loud_power: i64 = expected.2.iter().map(|&v| i64::from(v)).sum();
    let quiet_power: i64 = flat(&run(&quiet, 12000, config(5, true, 7, 1), 12000))
        .2
        .iter()
        .map(|&v| i64::from(v))
        .sum();
    assert!(
        quiet_power * 3 < loud_power * 2,
        "{quiet_power} vs {loud_power}"
    );
}

#[test]
fn bin_timing_reports_the_same_bins_without_offsets() {
    let data = recording(8000, 2, &[vec![2500, 6000], vec![4100]]);
    let with = run(&data, 8000, config(2, true, 0, 1), 8000);
    let without = run(&data, 8000, config(2, false, 0, 1), 8000);
    assert_eq!(with[0].events, without[0].events);
    assert!(without[0].offsets.is_empty());
    assert_eq!(
        event_times(&without, 2, 20),
        vec![vec![2500 / 20 * 20, 6000 / 20 * 20], vec![4100 / 20 * 20]]
    );
}

#[test]
fn settings_outside_their_ranges_are_refused() {
    let good = config(4, false, 20, 2);
    let mutations: Vec<fn(&mut SpikeEventConfig)> = vec![
        |c| c.channels = 0,
        |c| c.bin_samples = 0,
        |c| c.bin_samples = 1025,
        |c| c.power_bins = 65536,
        |c| c.groups = 5,
        |c| c.sample_rate_hz = 0.0,
        |c| c.refractory_samples = 19,
        |c| c.threshold_sigma = 0.0,
        |c| c.threshold_sigma = f64::NAN,
        |c| c.power_shift = 16,
        |c| c.warmup_samples = (1 << 40) + 1,
        |c| c.highpass_hz = 15000.0,
        |c| c.lowpass_hz = 1.0,
    ];
    for (index, mutate) in mutations.iter().enumerate() {
        let mut c = good.clone();
        mutate(&mut c);
        assert!(SpikeEventEncoder::new(c).is_err(), "mutation {index}");
    }
    assert!(SpikeEventDecoder::new(4, 20, false, 20, 0).is_err());
    let mut encoder = SpikeEventEncoder::new(good).unwrap();
    assert!(encoder
        .encode(&[0; 7], 2)
        .unwrap_err()
        .contains("window size"));
}

#[test]
fn corrupted_packets_are_refused() {
    let data = recording(4000, 2, &[vec![2500], vec![3000]]);
    let mut encoder = SpikeEventEncoder::new(config(2, true, 5, 2)).unwrap();
    let packet = encoder.encode(&data, 4000).unwrap();
    let fresh = || SpikeEventDecoder::new(2, 20, true, 5, 2).unwrap();
    let cases: Vec<(Vec<u8>, &str)> = vec![
        (packet[..3].to_vec(), "missing group length"),
        (packet[..packet.len() - 1].to_vec(), "longer than packet"),
        ([packet.clone(), vec![0]].concat(), "trailing bytes"),
        (
            [&packet[..4], &[1], &packet[5..]].concat(),
            "range-coder start",
        ),
    ];
    for (broken, message) in cases {
        let error = fresh().decode(&broken, 4000).unwrap_err();
        assert!(error.contains(message), "{error}");
    }
    let mut decoder = fresh();
    assert!(
        decoder.decode(&packet, 8000).is_err(),
        "more samples than coded"
    );
    let error = decoder.decode(&packet, 4000).unwrap_err();
    assert!(error.contains("start a new decoder"), "{error}");
    let mut decoder = fresh();
    assert!(decoder.decode(&packet[..3], 4000).is_err());
    let window = decoder.decode(&packet, 4000).unwrap();
    assert_eq!(window.events.iter().map(|&e| u32::from(e)).sum::<u32>(), 2);
    let mut refused = 0;
    for position in 9..packet.len() {
        let mut flipped = packet.clone();
        flipped[position] ^= 0x5A;
        if fresh().decode(&flipped, 4000).is_err() {
            refused += 1;
        }
    }
    assert!(refused > 0);
}
