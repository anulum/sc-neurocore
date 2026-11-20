// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Independent raster frames and SSCF containers

//! Encode and decode independently framed binary spike rasters.

#![warn(missing_docs)]

pub(crate) mod bindings;
mod frame;

use frame::Frame;

type RasterError = &'static str;
const MAGIC: &[u8] = b"SSCF";

/// An encoded SSCF packet and the frame statistics required by the public codec.
pub struct EncodedRaster {
    /// Entire container, including header and final padded frame.
    pub packet: Vec<u8>,
    /// Spike count in the original raster.
    pub spikes: usize,
    /// Sum of active channel counts over all frames.
    pub active_channels: usize,
    /// Size of the largest encoded frame including its header.
    pub max_frame_bytes: usize,
}

/// A decoded binary raster in row-major order.
pub struct DecodedRaster {
    /// Samples in row-major order.
    pub samples: Vec<i8>,
    /// Number of time samples.
    pub rows: usize,
    /// Number of channels.
    pub channels: usize,
}

fn check_shape(samples: &[i8], rows: usize, channels: usize) -> Result<(), RasterError> {
    if rows.checked_mul(channels) != Some(samples.len()) {
        return Err("invalid raster shape");
    }
    Ok(())
}

fn allocate(rows: usize, channels: usize) -> Result<DecodedRaster, RasterError> {
    let len = rows
        .checked_mul(channels)
        .ok_or("raster shape is too large")?;
    let mut samples = Vec::new();
    samples
        .try_reserve_exact(len)
        .map_err(|_| "raster allocation failed")?;
    samples.resize(len, 0);
    Ok(DecodedRaster {
        samples,
        rows,
        channels,
    })
}

/// Encode a single unpadded frame.
///
/// # Errors
/// Returns an error for nonbinary samples, invalid shape or dimensions beyond uint16.
pub fn encode_frame(samples: &[i8], rows: usize, channels: usize) -> Result<Vec<u8>, RasterError> {
    check_shape(samples, rows, channels)?;
    let mut output = Vec::new();
    frame::append(&mut output, samples, rows, channels, rows)?;
    Ok(output)
}

/// Decode a single standalone frame.
///
/// # Errors
/// Returns an error for a truncated frame, trailing bytes, nonzero mask padding
/// or a failed allocation.
pub fn decode_frame(packet: &[u8]) -> Result<DecodedRaster, RasterError> {
    let (frame, rest) = Frame::parse(packet)?;
    if !rest.is_empty() {
        return Err("trailing raster frame bytes");
    }
    frame.validate_padding(frame.rows)?;
    let mut decoded = allocate(frame.rows, frame.channels)?;
    frame.decode_into(&mut decoded.samples);
    Ok(decoded)
}

/// Encode a row-major raster as an SSCF container.
///
/// # Errors
/// Returns an error for nonbinary samples, invalid shape, a zero window or
/// dimensions outside the packet field widths.
pub fn encode_raster(
    samples: &[i8],
    rows: usize,
    channels: usize,
    window: usize,
) -> Result<EncodedRaster, RasterError> {
    check_shape(samples, rows, channels)?;
    let w = u16::try_from(window)
        .ok()
        .filter(|&value| value != 0)
        .ok_or("raster window size must be in 1..65535")?;
    let t = u32::try_from(rows).map_err(|_| "raster rows exceed uint32")?;
    let n = u16::try_from(channels).map_err(|_| "raster channels exceed uint16")?;
    let mut packet = MAGIC.to_vec();
    packet.extend_from_slice(&w.to_be_bytes());
    packet.extend_from_slice(&t.to_be_bytes());
    packet.extend_from_slice(&n.to_be_bytes());
    packet.extend_from_slice(&t.div_ceil(u32::from(w)).to_be_bytes());

    let mut spikes = 0;
    let mut active_channels = 0;
    let mut max_frame_bytes = 0;
    for start in (0..rows).step_by(window) {
        let count = window.min(rows - start);
        let end = start + count;
        let raster = &samples[start * channels..end * channels];
        let previous = packet.len();
        let (frame_spikes, frame_active) =
            frame::append(&mut packet, raster, count, channels, window)?;
        spikes += frame_spikes;
        active_channels += frame_active;
        max_frame_bytes = max_frame_bytes.max(packet.len() - previous);
    }
    Ok(EncodedRaster {
        packet,
        spikes,
        active_channels,
        max_frame_bytes,
    })
}

struct Container<'a> {
    body: &'a [u8],
    window: usize,
    rows: usize,
    channels: usize,
}

impl<'a> Container<'a> {
    fn parse(packet: &'a [u8]) -> Result<Self, RasterError> {
        let packet = packet
            .strip_prefix(MAGIC)
            .ok_or("Invalid header magic for SSCF")?;
        let [w0, w1, t0, t1, t2, t3, n0, n1, f0, f1, f2, f3, body @ ..] = packet else {
            return Err("truncated SSCF header");
        };
        let window = usize::from(u16::from_be_bytes([*w0, *w1]));
        let rows = u32::from_be_bytes([*t0, *t1, *t2, *t3]) as usize;
        let channels = usize::from(u16::from_be_bytes([*n0, *n1]));
        let frames = u32::from_be_bytes([*f0, *f1, *f2, *f3]) as usize;
        if window == 0 || frames != rows.div_ceil(window) {
            return Err("invalid SSCF frame count or window size");
        }
        if frames > body.len() / (frame::HEADER_SIZE + channels.div_ceil(8)) {
            return Err("truncated SSCF frames");
        }
        Ok(Self {
            body,
            window,
            rows,
            channels,
        })
    }

    fn visit(&self, mut read: impl FnMut(Frame<'a>, usize)) -> Result<(), RasterError> {
        let mut remaining = self.body;
        for start in (0..self.rows).step_by(self.window) {
            let (frame, rest) = Frame::parse(remaining)?;
            if frame.rows != self.window || frame.channels != self.channels {
                return Err("raster frame dimensions do not match SSCF header");
            }
            frame.validate_padding(self.window.min(self.rows - start))?;
            read(frame, start);
            remaining = rest;
        }
        if !remaining.is_empty() {
            return Err("trailing SSCF bytes");
        }
        Ok(())
    }
}

/// Decode an SSCF packet, optionally retaining a prefix of its recorded rows.
///
/// A zero row or channel argument selects the stored value. A nonzero
/// channel argument must match the stored count.
///
/// # Errors
/// Returns an error for inconsistent geometry, malformed frames or a failed allocation.
pub fn decode_raster(
    packet: &[u8],
    rows: usize,
    channels: usize,
) -> Result<DecodedRaster, RasterError> {
    let container = Container::parse(packet)?;
    let rows = if rows == 0 { container.rows } else { rows };
    if rows > container.rows || (channels != 0 && channels != container.channels) {
        return Err("requested raster shape does not match SSCF header");
    }

    container.visit(|_, _| {})?;
    let mut decoded = allocate(rows, container.channels)?;
    container.visit(|frame, start| {
        if start < rows {
            let end = (rows - start).min(container.window) + start;
            let channels = container.channels;
            frame.decode_into(&mut decoded.samples[start * channels..end * channels]);
        }
    })?;
    Ok(decoded)
}
