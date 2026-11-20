// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Standalone WFC2 encoder and decoder example

//! Read row-major PCM16LE and write a WFC2 file, or decode one to raw samples.

use sc_neurocore_engine::codecs::waveform_container::{
    decode, encode_lossless, encode_lossy, SectionCompression, WaveformConfig, WaveformMode,
    WaveformSamples, MAX_DECODE_SAMPLES,
};
use std::fs;
use std::process::ExitCode;

fn main() -> ExitCode {
    match run(std::env::args().skip(1).collect()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error}");
            ExitCode::FAILURE
        }
    }
}

fn run(arguments: Vec<String>) -> Result<(), String> {
    let string_args: Vec<_> = arguments.iter().map(String::as_str).collect();
    match string_args.as_slice() {
        ["encode", mode, rows, channels, input, output, flags @ ..] => {
            let rows = rows.parse().map_err(|_| "rows must be an integer")?;
            let channels = channels
                .parse()
                .map_err(|_| "channels must be an integer")?;
            if flags
                .iter()
                .any(|flag| !matches!(*flag, "--zstd" | "--db4"))
            {
                return Err("unknown flag; choose --zstd or --db4".into());
            }
            let config = WaveformConfig {
                mode: WaveformMode::parse(mode)?,
                compression: if flags.contains(&"--zstd") {
                    SectionCompression::Zstandard
                } else {
                    SectionCompression::Zlib
                },
                denoise_background: flags.contains(&"--db4"),
                ..WaveformConfig::default()
            };
            let bytes = fs::read(input).map_err(|error| error.to_string())?;
            let (pairs, tail) = bytes.as_chunks::<2>();
            if !tail.is_empty() {
                return Err("PCM16LE input has a partial sample".into());
            }
            let samples: Vec<_> = pairs.iter().copied().map(i16::from_le_bytes).collect();
            let encoded = if config.mode.is_lossless() {
                encode_lossless(&samples, rows, channels, &config)?
            } else {
                let float_samples: Vec<_> = samples.into_iter().map(f32::from).collect();
                encode_lossy(&float_samples, rows, channels, &config)?
            };
            fs::write(output, encoded.packet).map_err(|error| error.to_string())
        }
        ["decode", input, output] => {
            let packet = fs::read(input).map_err(|error| error.to_string())?;
            let decoded = decode(&packet, MAX_DECODE_SAMPLES)?;
            let (bytes, dtype): (Vec<u8>, _) = match decoded.samples {
                Some(WaveformSamples::Exact(values)) => (
                    values.into_iter().flat_map(i16::to_le_bytes).collect(),
                    "int16le",
                ),
                Some(WaveformSamples::Approximate(values)) => (
                    values.into_iter().flat_map(f32::to_le_bytes).collect(),
                    "float32le",
                ),
                None => (
                    decoded
                        .raster
                        .ok_or("spike packet is missing its raster")?
                        .into_iter()
                        .map(|value| value as u8)
                        .collect(),
                    "uint8",
                ),
            };
            fs::write(output, bytes).map_err(|error| error.to_string())?;
            println!(
                "{} {} {} {}",
                decoded.mode.as_str(),
                decoded.rows,
                decoded.channels,
                dtype
            );
            Ok(())
        }
        _ => Err(
            "usage: waveform_container encode MODE ROWS CHANNELS PCM16LE WFC2 [--zstd] [--db4]\n\
             or: waveform_container decode WFC2 OUT"
                .into(),
        ),
    }
}
