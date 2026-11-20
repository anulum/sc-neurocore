// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — WFC2 framing and checked section boundaries

use super::WaveformMode;

const MAGIC: &[u8; 4] = b"WFC2";
const LEGACY_MAGIC: &[u8; 4] = b"WFCX";
const HEADER_SIZE: usize = 20;

#[derive(Clone, Copy)]
pub(super) struct Header {
    pub(super) rows: usize,
    pub(super) channels: usize,
    pub(super) templates: usize,
    pub(super) snippets: usize,
    pub(super) snippet_samples: usize,
    pub(super) quantize_bits: u8,
    pub(super) spike_length_bits: u8,
    pub(super) mode: WaveformMode,
    pub(super) legacy: bool,
}

impl Header {
    /// Validate the version, checksum and geometry before any decoded allocation.
    pub(super) fn read(packet: &[u8], max_samples: usize) -> Result<(Self, Reader<'_>), String> {
        let bytes = if packet.len() < HEADER_SIZE {
            return Err("stream shorter than the WaveformCodec header".into());
        } else if packet.starts_with(MAGIC) {
            if packet.len() < HEADER_SIZE + 4 {
                return Err("stream shorter than the WaveformCodec header and checksum".into());
            }
            let (body, tail) = packet.split_at(packet.len() - 4);
            let mut checksum = Reader::new(tail);
            if crc32fast::hash(body) != checksum.u32()? {
                return Err("WaveformCodec checksum mismatch: the stream is corrupted".into());
            }
            body
        } else if packet.starts_with(LEGACY_MAGIC) {
            packet
        } else {
            return Err("not a WaveformCodec stream (invalid magic)".into());
        };

        let mut reader = Reader::new(&bytes[4..]);
        let rows = reader.u32()? as usize;
        let channels = reader.u32()? as usize;
        dimensions(rows, channels, max_samples)?;
        let templates = usize::from(reader.u16()?);
        let snippets = usize::from(reader.u16()?);
        let snippet_samples = usize::from(reader.u8()?);
        let quantize_bits = reader.u8()?;
        let spike_length_bits = reader.u8()?;
        let mode = WaveformMode::from_byte(reader.u8()?)?;
        let legacy = packet.starts_with(LEGACY_MAGIC);
        if legacy {
            if mode.is_lossless() {
                return Err("lossless streams exist only in the WFC2 format".into());
            }
            if mode != WaveformMode::Spike {
                return Err("legacy WFCX streams can decode only their spike raster".into());
            }
        } else {
            validate_quantization(snippet_samples, quantize_bits)?;
            if templates > snippets {
                return Err("invalid WaveformCodec template count".into());
            }
            if snippets > rows * channels {
                return Err("invalid WaveformCodec snippet count".into());
            }
            if mode.is_lossless() && (templates != 0 || snippets != 0 || spike_length_bits != 0) {
                return Err("lossless WaveformCodec metadata must have zero spike counts".into());
            }
        }
        Ok((
            Self {
                rows,
                channels,
                templates,
                snippets,
                snippet_samples,
                quantize_bits,
                spike_length_bits,
                mode,
                legacy,
            },
            reader,
        ))
    }

    /// Write a versioned header and length-prefixed sections with a final checksum.
    pub(super) fn pack(self, sections: &[&[u8]]) -> Result<Vec<u8>, String> {
        let mut packet = MAGIC.to_vec();
        packet.extend_from_slice(&(self.rows as u32).to_be_bytes());
        packet.extend_from_slice(&(self.channels as u32).to_be_bytes());
        packet.extend_from_slice(&(self.templates as u16).to_be_bytes());
        packet.extend_from_slice(&(self.snippets as u16).to_be_bytes());
        packet.extend_from_slice(&[
            self.snippet_samples as u8,
            self.quantize_bits,
            self.spike_length_bits,
            self.mode as u8,
        ]);
        for section in sections {
            write_section(&mut packet, section)?;
        }
        packet.extend_from_slice(&crc32fast::hash(&packet).to_be_bytes());
        Ok(packet)
    }
}

/// Refuse invalid dimensions and ensure the sample product fits the caller's limit.
pub(super) fn dimensions(rows: usize, channels: usize, limit: usize) -> Result<usize, String> {
    rows.checked_mul(channels)
        .filter(|&cells| {
            cells > 0 && cells <= limit && u32::try_from(rows).is_ok() && u32::try_from(channels).is_ok()
        })
        .ok_or_else(|| format!("stream declares {rows} x {channels} samples; at least one each and at most max_samples={limit} in total"))
}

/// Check quantizer ranges represented in both the options and wire header.
pub(super) fn validate_quantization(width: usize, bits: u8) -> Result<(), String> {
    if !(1..=255).contains(&width) {
        return Err("snippet_samples must be in 1..=255".into());
    }
    if !(1..=8).contains(&bits) {
        return Err("quantize_bits must be in 1..=8".into());
    }
    Ok(())
}

/// Append a section with its unsigned big-endian byte length.
pub(super) fn write_section(packet: &mut Vec<u8>, bytes: &[u8]) -> Result<(), String> {
    let length = u32::try_from(bytes.len()).map_err(|_| "WaveformCodec section exceeds uint32")?;
    packet.extend_from_slice(&length.to_be_bytes());
    packet.extend_from_slice(bytes);
    Ok(())
}

/// A cursor over bytes declared by a header or a length-prefixed section.
pub(super) struct Reader<'a>(&'a [u8]);

impl<'a> Reader<'a> {
    pub(super) fn new(bytes: &'a [u8]) -> Self {
        Self(bytes)
    }

    pub(super) fn take(&mut self, size: usize) -> Result<&'a [u8], String> {
        let (value, tail) = self
            .0
            .split_at_checked(size)
            .ok_or_else(|| "malformed WaveformCodec stream: truncated section".to_string())?;
        self.0 = tail;
        Ok(value)
    }

    pub(super) fn u8(&mut self) -> Result<u8, String> {
        Ok(self.take(1)?[0])
    }

    pub(super) fn u16(&mut self) -> Result<u16, String> {
        Ok(u16::from_be_bytes(self.array()?))
    }

    pub(super) fn u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_be_bytes(self.array()?))
    }

    pub(super) fn f32(&mut self) -> Result<f32, String> {
        let value = f32::from_be_bytes(self.array()?);
        if !value.is_finite() {
            return Err("non-finite WaveformCodec section value".into());
        }
        Ok(value)
    }

    pub(super) fn section(&mut self) -> Result<&'a [u8], String> {
        let length = self.u32()?;
        self.take(length as usize)
    }

    pub(super) fn rest(&self) -> &'a [u8] {
        self.0
    }

    pub(super) fn finish(self) -> Result<(), String> {
        if !self.0.is_empty() {
            return Err("trailing bytes after the WaveformCodec stream".into());
        }
        Ok(())
    }

    fn array<const N: usize>(&mut self) -> Result<[u8; N], String> {
        let mut result = [0; N];
        result.copy_from_slice(self.take(N)?);
        Ok(result)
    }
}
