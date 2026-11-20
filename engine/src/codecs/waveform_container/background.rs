// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — WFC2 continuous background section

use super::compression;
use super::wavelet;
use super::wire::{Header, Reader};
use super::{kernels, zeroed, WaveformConfig};

pub(super) fn encode(
    waveform: &[f32],
    time_rows: usize,
    channels: usize,
    spike_times: &[Vec<usize>],
    config: &WaveformConfig,
) -> Result<Vec<u8>, String> {
    let (rows, mut background) = kernels::extract_background(
        waveform,
        time_rows,
        channels,
        spike_times,
        config.snippet_samples / 2,
    );
    for row in background.chunks_exact_mut(channels) {
        for c in (1..channels).rev() {
            row[c] -= row[c - 1];
        }
    }
    if config.denoise_background {
        wavelet::denoise(&mut background, rows, channels);
    }
    if background.iter().any(|value| !value.is_finite()) {
        return Err("non-finite WaveformCodec background".into());
    }
    let dmax = background
        .iter()
        .skip(channels)
        .zip(&background)
        .map(|(current, previous)| (current - previous).abs())
        .fold(1e-6_f32, f32::max);
    if !dmax.is_finite() {
        return Err("non-finite WaveformCodec delta scale".into());
    }
    let half = 1_i32 << (config.quantize_bits - 1);
    let step = step(dmax, half);
    let codes = kernels::quantize_background(&background, rows, channels, step, half);
    let payload: Vec<_> = codes.into_iter().map(|code| code as u8).collect();
    let mut bytes = (rows as u32).to_be_bytes().to_vec();
    bytes.extend_from_slice(&(channels as u32).to_be_bytes());
    bytes.extend_from_slice(&dmax.to_be_bytes());
    for &sample in &background[..channels] {
        bytes.extend_from_slice(&sample.to_be_bytes());
    }
    bytes.extend_from_slice(&compression::compress(&payload, config.compression)?);
    Ok(bytes)
}

pub(super) fn decode(bytes: &[u8], header: Header) -> Result<Vec<f32>, String> {
    if bytes.len() < 12 {
        return Err("full-mode stream lacks its background section".into());
    }
    let mut reader = Reader::new(bytes);
    let rows = reader.u32()? as usize;
    let channels = reader.u32()? as usize;
    let dmax = reader.f32()?;
    let ds = kernels::BACKGROUND_DOWNSAMPLE;
    let expected_rows = if header.rows < ds {
        header.rows
    } else {
        header.rows / ds
    };
    if rows != expected_rows || channels != header.channels {
        return Err("WaveformCodec background dimensions disagree with the header".into());
    }
    let half = 1_i32 << (header.quantize_bits - 1);
    let step = step(dmax, half);
    if step <= 0.0 {
        return Err("invalid WaveformCodec background delta scale".into());
    }
    let first: Vec<_> = (0..channels)
        .map(|_| reader.f32())
        .collect::<Result<_, _>>()?;
    let data = compression::decompress(reader.rest(), rows * channels)?;
    let codes: Vec<_> = data.into_iter().map(|code| code as i8).collect();
    if codes[..channels].iter().any(|&code| code != 0)
        || codes
            .iter()
            .any(|&code| !(-half..half).contains(&i32::from(code)))
    {
        return Err("invalid WaveformCodec background code".into());
    }
    let mut restored = kernels::reconstruct_background(&codes, rows, channels, &first, step);
    for row in restored.chunks_exact_mut(channels) {
        for c in 1..channels {
            row[c] += row[c - 1];
        }
    }
    if restored.iter().any(|value| !value.is_finite()) {
        return Err("non-finite WaveformCodec reconstructed background".into());
    }
    if rows == header.rows {
        return Ok(restored);
    }
    let mut waveform = zeroed(header.rows * channels, 0.0)?;
    for (index, row) in waveform.chunks_exact_mut(channels).enumerate() {
        let source = (index / ds).min(rows - 1);
        row.copy_from_slice(&restored[source * channels..(source + 1) * channels]);
    }
    Ok(waveform)
}

fn step(dmax: f32, half: i32) -> f32 {
    dmax / (half - 1).max(1) as f32
}
