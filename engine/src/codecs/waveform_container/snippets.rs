// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — WFC2 snippet template and residual sections

use super::compression;
use super::wire::{Header, Reader};
use super::{SectionCompression, WaveformConfig, WaveformMode};
use crate::codecs::waveform_codec::template_match;

pub(super) struct SnippetSection {
    pub(super) templates: usize,
    pub(super) bytes: Vec<u8>,
}

/// Extract and match channel-major snippets, retaining the historical zero padding.
pub(super) fn encode(
    waveform: &[f32],
    rows: usize,
    channels: usize,
    times: &[Vec<usize>],
    count: usize,
    config: &WaveformConfig,
) -> Result<SnippetSection, String> {
    let width = config.snippet_samples;
    let half = width / 2;
    let mut clips = Vec::with_capacity(count * width);
    for (ch, channel_times) in times.iter().enumerate() {
        for &time in channel_times {
            let clip_end = clips.len() + width;
            let start = time.saturating_sub(half);
            let end = rows.min(time + half);
            clips.extend((start..end).map(|row| waveform[row * channels + ch]));
            clips.resize(clip_end, 0.0);
        }
    }
    let matches = template_match(
        &clips,
        count,
        width,
        config.max_templates,
        config.template_threshold,
    );
    let template_count = matches.template_sources.len();
    if config.mode == WaveformMode::Spike {
        return Ok(SnippetSection {
            templates: template_count,
            bytes: Vec::new(),
        });
    }
    let mut section = Vec::new();
    section.extend_from_slice(&(template_count as u16).to_be_bytes());
    let templates = matches
        .template_sources
        .iter()
        .flat_map(|&index| &clips[index * width..(index + 1) * width]);
    let template_scale = if template_count == 0 {
        0.0
    } else {
        max_magnitude(templates.clone().copied())
    };
    section.extend_from_slice(&template_scale.to_be_bytes());
    section.extend(
        templates
            .map(|&sample| ((sample / template_scale * 127.0).clamp(-127.0, 127.0) as i8) as u8),
    );
    section.extend_from_slice(&(count as u32).to_be_bytes());
    for id in matches.template_ids {
        put_id(&mut section, id);
    }
    for scale in matches.scales {
        let scale = scale as f32;
        if !scale.is_finite() {
            return Err("non-finite WaveformCodec template scale".into());
        }
        section.extend_from_slice(&scale.to_be_bytes());
    }
    encode_residuals(&mut section, &matches.residuals, config.compression)?;
    Ok(SnippetSection {
        templates: template_count,
        bytes: section,
    })
}

/// Add the quantizer scale, padded sample count and a single compressed nibble frame.
fn encode_residuals(
    section: &mut Vec<u8>,
    residuals: &[f32],
    method: SectionCompression,
) -> Result<(), String> {
    if residuals.is_empty() {
        section.extend_from_slice(&[0; 8]);
        return Ok(());
    }
    if residuals.iter().any(|sample| !sample.is_finite()) {
        return Err("non-finite WaveformCodec snippet residual".into());
    }
    let scale = max_magnitude(residuals.iter().copied());
    let quantize =
        |sample: f32| ((sample / scale * 7.0).round_ties_even().clamp(-7.0, 7.0) as i8 + 8) as u8;
    let packed: Vec<u8> = residuals
        .chunks(2)
        .map(|pair| (quantize(pair[0]) << 4) | pair.get(1).map_or(8, |&sample| quantize(sample)))
        .collect();
    section.extend_from_slice(&scale.to_be_bytes());
    section.extend_from_slice(&((packed.len() * 2) as u32).to_be_bytes());
    section.extend_from_slice(&compression::compress(&packed, method)?);
    Ok(())
}

fn max_magnitude(values: impl Iterator<Item = f32>) -> f32 {
    values.map(f32::abs).fold(1e-6, f32::max)
}

fn put_id(output: &mut Vec<u8>, mut value: usize) {
    while value >= 128 {
        output.push(value as u8 | 0x80);
        value >>= 7;
    }
    output.push(value as u8);
}

/// Read a bounded uint16-sized LEB128 template index and resolve it in the library.
fn read_id(reader: &mut Reader<'_>, count: usize) -> Result<usize, String> {
    let mut value = 0;
    for shift in [0, 7, 14] {
        let byte = reader.u8()?;
        value |= usize::from(byte & 0x7f) << shift;
        if byte < 128 {
            return (value < count)
                .then_some(value)
                .ok_or_else(|| "WaveformCodec template ID out of range".into());
        }
    }
    Err("WaveformCodec template ID exceeds uint16".into())
}

/// Reconstruct snippets in channel order and overwrite their clipped positions.
pub(super) fn apply(
    bytes: &[u8],
    header: Header,
    raster: &[i8],
    waveform: &mut [f32],
) -> Result<(), String> {
    let mut reader = Reader::new(bytes);
    let count = usize::from(reader.u16()?);
    let scale = reader.f32()?;
    if count != header.templates {
        return Err("template count disagrees with the header".into());
    }
    if scale < 0.0 {
        return Err("negative WaveformCodec template scale".into());
    }
    let width = header.snippet_samples;
    let library = reader.take(header.templates * width)?;
    if reader.u32()? as usize != header.snippets {
        return Err("snippet count disagrees with the header".into());
    }
    let ids = (0..header.snippets)
        .map(|_| read_id(&mut reader, header.templates))
        .collect::<Result<Vec<_>, _>>()?;
    let scales = (0..header.snippets)
        .map(|_| reader.f32())
        .collect::<Result<Vec<_>, _>>()?;
    let residual_scale = reader.f32()?;
    let padded = header.snippets * width;
    let packed_bytes = padded.div_ceil(2);
    if reader.u32()? as usize != packed_bytes * 2 || residual_scale < 0.0 {
        return Err("invalid WaveformCodec residual sample count or scale".into());
    }
    let residuals = if packed_bytes == 0 {
        reader.finish()?;
        Vec::new()
    } else {
        compression::decompress(reader.rest(), packed_bytes)?
    };
    if padded % 2 == 1 && residuals.last().is_some_and(|last| last & 0x0f != 8) {
        return Err("WaveformCodec residual padding is nonzero".into());
    }
    let mut index = 0;
    let half = width / 2;
    for channel in 0..header.channels {
        for time in 0..header.rows {
            if raster[time * header.channels + channel] == 0 {
                continue;
            }
            let start = time.saturating_sub(half);
            let end = header.rows.min(time + half);
            let template = &library[ids[index] * width..(ids[index] + 1) * width];
            for (offset, row) in (start..end).enumerate() {
                let sample_index = index * width + offset;
                let byte = residuals[sample_index / 2];
                let code = if sample_index.is_multiple_of(2) {
                    byte >> 4
                } else {
                    byte & 0x0f
                };
                let residual = f32::from(code as i8 - 8) * (residual_scale / 7.0);
                let value = scales[index] * (f32::from(template[offset] as i8) * (scale / 127.0))
                    + residual;
                if !value.is_finite() {
                    return Err("non-finite WaveformCodec reconstructed snippet".into());
                }
                waveform[row * header.channels + channel] = value;
            }
            index += 1;
        }
    }
    Ok(())
}
