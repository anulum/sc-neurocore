// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Public native waveform container contracts

//! Exercise complete packets through the standalone Rust API.

use sc_neurocore_engine::codecs::waveform_container::{
    decode, encode_lossless, encode_lossy, WaveformConfig, WaveformMode, WaveformSamples,
};

fn trace(rows: usize, channels: usize) -> Vec<i16> {
    (0..rows)
        .flat_map(|t| (0..channels).map(move |c| (t, c)))
        .map(|(t, c)| match t {
            1 | 136 | 258 => -12000 + 100 * c as i16,
            _ => 40 * c as i16 + (t % 29) as i16 - 17,
        })
        .collect()
}

#[test]
fn every_mode_encodes_and_decodes_its_declared_geometry() {
    let (rows, channels) = (259, 3);
    let samples = trace(rows, channels);
    let values: Vec<_> = samples.iter().copied().map(f32::from).collect();
    for mode in [
        WaveformMode::Spike,
        WaveformMode::Waveform,
        WaveformMode::Full,
        WaveformMode::Lossless,
        WaveformMode::LosslessRice,
    ] {
        let config = WaveformConfig {
            mode,
            snippet_samples: 13,
            denoise_background: true,
            ..WaveformConfig::default()
        };
        let encoded = if mode.is_lossless() {
            encode_lossless(&samples, rows, channels, &config).unwrap()
        } else {
            encode_lossy(&values, rows, channels, &config).unwrap()
        };
        assert!(encoded.packet.starts_with(b"WFC2"));
        let restored = decode(&encoded.packet, rows * channels).unwrap();
        assert_eq!(
            (restored.mode, restored.rows, restored.channels),
            (mode, rows, channels)
        );
        if mode.is_lossless() {
            assert!(restored.raster.is_none());
            assert!(
                matches!(restored.samples, Some(WaveformSamples::Exact(values)) if values == samples)
            );
            assert_eq!(encoded.spikes, 0);
        } else {
            let raster = restored.raster.unwrap();
            assert_eq!(encoded.spikes, 9);
            assert_eq!(
                raster.iter().filter(|&&active| active != 0).count(),
                encoded.spikes
            );
            for time in [1, 136, 258] {
                assert_eq!(&raster[time * channels..(time + 1) * channels], &[1; 3]);
            }
            if mode == WaveformMode::Spike {
                assert!(restored.samples.is_none());
                assert_eq!(encoded.snippet_bytes + encoded.background_bytes, 0);
            } else {
                assert!(
                    matches!(restored.samples, Some(WaveformSamples::Approximate(values)) if values.len() == rows * channels)
                );
                assert!(encoded.snippet_bytes > 0);
                assert_eq!(encoded.background_bytes > 0, mode == WaveformMode::Full);
            }
        }
    }
}

#[test]
fn original_waveform_packets_keep_checksum_sample_cap_and_legacy_spikes() {
    let rows = 259;
    let values: Vec<_> = trace(rows, 1).into_iter().map(f32::from).collect();
    let config = WaveformConfig {
        mode: WaveformMode::Spike,
        ..WaveformConfig::default()
    };
    let packet = encode_lossy(&values, rows, 1, &config).unwrap().packet;
    assert!(decode(&packet, rows - 1)
        .unwrap_err()
        .contains("max_samples"));
    let mut invalid = packet.clone();
    invalid[20] ^= 1;
    assert!(decode(&invalid, rows).unwrap_err().contains("checksum"));
    let mut old = packet[..packet.len() - 4].to_vec();
    old[..4].copy_from_slice(b"WFCX");
    assert_eq!(
        decode(&old, rows).unwrap().raster,
        decode(&packet, rows).unwrap().raster
    );
    old[19] = WaveformMode::Full as u8;
    assert!(decode(&old, rows).unwrap_err().contains("legacy WFCX"));
}

#[cfg(feature = "waveform-zstd")]
#[test]
fn native_zstd_frames_match_zlib_reconstruction() {
    use sc_neurocore_engine::codecs::waveform_container::SectionCompression;
    let rows = 513;
    let values: Vec<_> = trace(rows, 2).into_iter().map(f32::from).collect();
    let mut config = WaveformConfig {
        denoise_background: true,
        ..WaveformConfig::default()
    };
    let zlib = encode_lossy(&values, rows, 2, &config).unwrap();
    config.compression = SectionCompression::Zstandard;
    let zstd = encode_lossy(&values, rows, 2, &config).unwrap();
    for (packet, magic) in [
        (&zstd.packet, [0x28, 0xb5, 0x2f, 0xfd]),
        (&zlib.packet, [0x78, 0xda, 0, 0]),
    ] {
        assert!(packet.windows(2).any(|window| window == &magic[..2]));
    }
    let left = decode(&zlib.packet, rows * 2).unwrap();
    let right = decode(&zstd.packet, rows * 2).unwrap();
    assert_eq!(left.raster, right.raster);
    match (left.samples, right.samples) {
        (Some(WaveformSamples::Approximate(a)), Some(WaveformSamples::Approximate(b))) => {
            assert_eq!(a, b)
        }
        _ => panic!("full packets must both reconstruct the waveform"),
    }
}
