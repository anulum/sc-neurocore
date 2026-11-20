// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Address-event and silence packets

//! Encode and decode binary raster events with the AERX and AERI wire formats.

#![warn(missing_docs)]

pub(crate) mod bindings;

const EVENTS: &[u8; 4] = b"AERX";
const SILENCES: &[u8; 4] = b"AERI";
const HEADER_SIZE: usize = 17;

/// Encoded address events and their statistics.
pub struct EncodedEvents {
    /// Complete AERX or AERI packet.
    pub packet: Vec<u8>,
    /// Spike count in the original raster.
    pub spikes: usize,
    /// Count of encoded events, excluding timestamp escapes.
    pub events: usize,
}

/// Decoded binary raster in row-major order.
pub struct DecodedEvents {
    /// Samples in time-major order.
    pub samples: Vec<i8>,
    /// Number of retained time samples.
    pub rows: usize,
    /// Number of retained channels.
    pub channels: usize,
}

fn append_address(output: &mut Vec<u8>, address: u32, width: usize) {
    let bytes = address.to_be_bytes();
    let leading = width.saturating_sub(bytes.len());
    output.resize(output.len() + leading, 0);
    output.extend_from_slice(&bytes[bytes.len().saturating_sub(width)..]);
}

/// Encode a binary time-by-channel raster with the requested address and delta settings.
///
/// A zero address width selects the smallest safe byte width. The timestamp
/// setting controls escape splitting within the fixed unsigned 16-bit wire slot.
///
/// # Errors
/// Returns an error for invalid field widths, nonbinary samples or dimensions
/// beyond the wire widths.
pub fn encode_raster(
    samples: &[i8],
    rows: usize,
    channels: usize,
    timestamp_bits: u8,
    neuron_bits: u16,
) -> Result<EncodedEvents, &'static str> {
    let t = u32::try_from(rows).map_err(|_| "AER rows exceed uint32")?;
    let n = u32::try_from(channels).map_err(|_| "AER channels exceed uint32")?;
    if rows.checked_mul(channels) != Some(samples.len()) {
        return Err("invalid AER raster shape");
    }
    if !(1..=16).contains(&timestamp_bits) || neuron_bits > 2040 {
        return Err("invalid AER timestamp or neuron bit width");
    }
    let mut spikes = 0;
    for &sample in samples {
        match sample {
            0 => {}
            1 => spikes += 1,
            _ => return Err("AER raster must contain only zeros and ones"),
        }
    }
    let inverted = spikes > samples.len() / 2;
    let events = if inverted {
        samples.len() - spikes
    } else {
        spikes
    };
    let event_field = u32::try_from(events).map_err(|_| "AER event count exceeds uint32")?;
    let automatic_width = (u32::BITS - n.leading_zeros()).max(1).div_ceil(8);
    let width = usize::from(neuron_bits.div_ceil(8)).max(automatic_width as usize);
    let mut packet = Vec::new();
    packet.extend_from_slice(if inverted { SILENCES } else { EVENTS });
    packet.extend_from_slice(&t.to_be_bytes());
    packet.extend_from_slice(&n.to_be_bytes());
    packet.extend_from_slice(&event_field.to_be_bytes());
    packet.push(width as u8);
    let max_delta = (1usize << timestamp_bits) - 1;
    let value = i8::from(!inverted);
    let mut previous = 0;
    for (time, row) in samples.chunks_exact(channels.max(1)).enumerate() {
        for (address, &sample) in row.iter().enumerate() {
            if sample != value {
                continue;
            }
            let mut delta = time - previous;
            while delta > max_delta {
                packet.extend_from_slice(&(max_delta as u16).to_be_bytes());
                packet.resize(packet.len() + width, 0xff);
                delta -= max_delta;
            }
            packet.extend_from_slice(&(delta as u16).to_be_bytes());
            append_address(&mut packet, address as u32, width);
            previous = time;
        }
    }
    Ok(EncodedEvents {
        packet,
        spikes,
        events,
    })
}

struct Packet<'a> {
    body: &'a [u8],
    rows: u32,
    channels: u32,
    count: u32,
    width: usize,
    inverted: bool,
}

impl<'a> Packet<'a> {
    fn parse(packet: &'a [u8]) -> Result<Self, &'static str> {
        let inverted = if packet.starts_with(EVENTS) {
            false
        } else if packet.starts_with(SILENCES) {
            true
        } else {
            return Err("Invalid header magic for AER");
        };
        let (header, body) = packet
            .split_at_checked(HEADER_SIZE)
            .ok_or("truncated AER header")?;
        let errors = "truncated AER header";
        let rows = u32::from_be_bytes(header[4..8].try_into().map_err(|_| errors)?);
        let channels = u32::from_be_bytes(header[8..12].try_into().map_err(|_| errors)?);
        let count = u32::from_be_bytes(header[12..16].try_into().map_err(|_| errors)?);
        let width = usize::from(header[16]);
        let max_channels = (1u32 << (width.min(3) * 8)) - 1;
        if width == 0 || (width < 4 && channels > max_channels) {
            return Err("AER address width cannot store the channel count and escape");
        }
        if u64::from(count) > u64::from(rows) * u64::from(channels) {
            return Err("AER event count exceeds raster positions");
        }
        let stride = width + 2;
        if !body.len().is_multiple_of(stride) || (body.len() / stride) < count as usize {
            return Err("invalid AER event body length");
        }
        Ok(Self {
            body,
            rows,
            channels,
            count,
            width,
            inverted,
        })
    }

    fn visit(&self, mut apply: impl FnMut(usize, usize)) -> Result<(), &'static str> {
        let mut time = 0u32;
        let mut count = 0;
        let mut previous = None;
        let mut escaped = false;
        for item in self.body.chunks_exact(self.width + 2) {
            if count == self.count {
                return Err("trailing AER event bytes");
            }
            let delta = u32::from(u16::from_be_bytes([item[0], item[1]]));
            time = time
                .checked_add(delta)
                .ok_or("AER timestamp overflows uint32")?;
            if time >= self.rows {
                return Err("AER timestamp is outside the raster");
            }
            let address = &item[2..];
            if address.iter().all(|&byte| byte == 0xff) {
                if delta == 0 {
                    return Err("AER timestamp escape must advance time");
                }
                escaped = true;
                continue;
            }
            let value = address
                .iter()
                .try_fold(0u32, |value, &byte| {
                    value.checked_mul(256)?.checked_add(u32::from(byte))
                })
                .ok_or("AER neuron address exceeds uint32")?;
            if value >= self.channels {
                return Err("AER neuron address is outside the raster");
            }
            if previous.is_some_and(|event| event >= (time, value)) {
                return Err("AER events must increase by timestamp then address");
            }
            previous = Some((time, value));
            escaped = false;
            count += 1;
            apply(time as usize, value as usize);
        }
        if escaped || count != self.count {
            return Err("incomplete AER events or timestamp escape");
        }
        Ok(())
    }
}

/// Decode an AERX or AERI packet, optionally cropping its rows and channels.
///
/// A zero argument selects the stored dimension. The packet is validated in
/// full even when the caller requests a prefix of its raster.
///
/// # Errors
/// Returns an error for invalid headers, escapes, dimensions, ordering,
/// overflow, trailing bytes or an allocation failure.
pub fn decode_raster(
    data: &[u8],
    rows: usize,
    channels: usize,
) -> Result<DecodedEvents, &'static str> {
    let packet = Packet::parse(data)?;
    let rows = if rows == 0 {
        packet.rows as usize
    } else {
        rows
    };
    let channels = if channels == 0 {
        packet.channels as usize
    } else {
        channels
    };
    if rows > packet.rows as usize || channels > packet.channels as usize {
        return Err("requested raster shape exceeds AER header");
    }
    packet.visit(|_, _| {})?;
    let length = rows
        .checked_mul(channels)
        .ok_or("AER raster shape is too large")?;
    let mut samples = Vec::new();
    samples
        .try_reserve_exact(length)
        .map_err(|_| "AER raster allocation failed")?;
    samples.resize(length, i8::from(packet.inverted));
    packet.visit(|time, address| {
        if time < rows && address < channels {
            samples[time * channels + address] = i8::from(!packet.inverted);
        }
    })?;
    Ok(DecodedEvents {
        samples,
        rows,
        channels,
    })
}
