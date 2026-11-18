// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Inter-spike interval raster codec

//! Encode and decode the legacy inter-spike interval raster packets.

mod framing;
mod huffman;

use framing::{write_section, write_varint, PacketReader};

/// Entropy representation used by the inter-spike interval encoder.
#[derive(Clone, Copy)]
pub enum Entropy {
    /// Select Huffman when the encoded event density exceeds three percent.
    Auto,
    /// Encode every event count, first timestamp and interval as unsigned LEB128.
    Varint,
    /// Encode intervals in a canonical Huffman body.
    Huffman,
}

fn raster_events(raster: &[i8], rows: usize, channels: usize, precision: usize) -> Vec<Vec<u32>> {
    let mut events = vec![Vec::new(); channels];
    if channels == 0 {
        return events;
    }
    // Ignore a final partial block; the packet retains the original row count.
    let complete_rows = rows - rows % precision;
    for (sample, row) in raster
        .chunks_exact(channels)
        .take(complete_rows)
        .enumerate()
    {
        let bin = (sample / precision) as u32;
        for (&active, times) in row.iter().zip(&mut events) {
            if active > 0 && times.last() != Some(&bin) {
                times.push(bin);
            }
        }
    }
    events
}

/// Encode a row-major raster and return the packet and encoded event count.
///
/// A positive `precision` groups complete blocks and leaves the original
/// dimensions in the header. A precision of one preserves the raster.
///
/// # Errors
///
/// Returns an error for invalid dimensions, invalid precision, ambiguous
/// varint framing, or an unrepresentable Huffman table or bit count.
pub fn encode_raster(
    raster: &[i8],
    rows: usize,
    channels: usize,
    precision: usize,
    entropy: Entropy,
) -> Result<(Vec<u8>, usize), &'static str> {
    let t = u32::try_from(rows).map_err(|_| "ISI rows exceed uint32")?;
    let n = u32::try_from(channels).map_err(|_| "ISI channels exceed uint32")?;
    if precision == 0 || rows.checked_mul(channels) != Some(raster.len()) {
        return Err("invalid ISI raster dimensions or timing precision");
    }
    let events = raster_events(raster, rows, channels, precision);
    let count: usize = events.iter().map(Vec::len).sum();
    let use_huffman = match entropy {
        Entropy::Auto => count as f64 / raster.len().max(1) as f64 > 0.03,
        Entropy::Varint => false,
        Entropy::Huffman => true,
    };
    let mut packet = Vec::new();
    if use_huffman {
        packet.push(1);
    } else if t >= 1 << 24 {
        return Err(
            "varint packets require T < 2**24 to avoid the Huffman marker collision; use entropy='huffman' for longer rasters",
        );
    }
    packet.extend_from_slice(&t.to_be_bytes());
    packet.extend_from_slice(&n.to_be_bytes());
    if use_huffman {
        encode_huffman(&events, &mut packet)?;
    } else {
        for times in events {
            write_varint(times.len() as u32, &mut packet);
            if let Some(&first) = times.first() {
                write_varint(first, &mut packet);
            }
            for times in times.windows(2) {
                write_varint(times[1] - times[0], &mut packet);
            }
        }
    }
    Ok((packet, count))
}

fn encode_huffman(events: &[Vec<u32>], output: &mut Vec<u8>) -> Result<(), &'static str> {
    let mut counts = Vec::new();
    let mut first_times = Vec::new();
    let mut intervals = Vec::new();
    for times in events {
        write_varint(times.len() as u32, &mut counts);
        if let Some(&first) = times.first() {
            write_varint(first, &mut first_times);
        }
        for pair in times.windows(2) {
            let interval = i32::try_from(pair[1] - pair[0]).map_err(|_| "ISI exceeds int32")?;
            intervals.push(interval);
        }
    }
    write_section(&counts, output)?;
    write_section(&first_times, output)?;
    huffman::encode(&intervals, output)
}

fn count(reader: &mut PacketReader<'_>, limit: usize) -> Result<usize, &'static str> {
    let count = reader.varint()? as usize;
    if count > limit {
        return Err("ISI event count exceeds the number of time bins");
    }
    Ok(count)
}

fn read_channel(
    first: u32,
    intervals: impl IntoIterator<Item = Result<u32, &'static str>>,
    bins: usize,
    mut event: impl FnMut(usize),
) -> Result<(), &'static str> {
    if first as usize >= bins {
        return Err("ISI timestamp is outside the raster");
    }
    let mut time = first;
    event(time as usize);
    for interval in intervals {
        let interval = interval?;
        if interval == 0 {
            return Err("ISI intervals must be positive");
        }
        time = time
            .checked_add(interval)
            .ok_or("ISI timestamp exceeds uint32")?;
        if time as usize >= bins {
            return Err("ISI timestamp is outside the raster");
        }
        event(time as usize);
    }
    Ok(())
}

fn decode_varint(
    mut reader: PacketReader<'_>,
    channels: usize,
    bins: usize,
    mut event: impl FnMut(usize, usize),
) -> Result<(), &'static str> {
    for channel in 0..channels {
        let count = count(&mut reader, bins)?;
        if count > reader.tail().len() {
            return Err("truncated ISI channel");
        }
        if count > 0 {
            let first = reader.varint()?;
            read_channel(first, (1..count).map(|_| reader.varint()), bins, |time| {
                event(time, channel);
            })?;
        }
    }
    reader.finish()
}

fn decode_huffman(
    mut reader: PacketReader<'_>,
    channels: usize,
    bins: usize,
    mut event: impl FnMut(usize, usize),
) -> Result<(), &'static str> {
    let mut count_section = reader.section()?;
    let mut first_section = reader.section()?;
    if count_section.tail().len() < channels {
        return Err("truncated ISI count section");
    }
    let counts: Vec<_> = (0..channels)
        .map(|_| count(&mut count_section, bins))
        .collect::<Result<_, _>>()?;
    count_section.finish()?;
    let interval_count = counts.iter().try_fold(0usize, |total, &count| {
        total
            .checked_add(count.saturating_sub(1))
            .ok_or("ISI event count overflow")
    })?;
    let mut intervals = huffman::decode(&mut reader, interval_count)?.into_iter();
    reader.finish()?;
    for (channel, count) in counts.into_iter().enumerate() {
        if count > 0 {
            let first = first_section.varint()?;
            let next = (&mut intervals).take(count - 1).map(|interval| {
                u32::try_from(interval).map_err(|_| "ISI intervals must be positive")
            });
            read_channel(first, next, bins, |time| event(time, channel))?;
        }
    }
    first_section.finish()
}

/// Decode to a flat, row-major binary raster.
///
/// The column count must match the header. The output may use the original
/// row count or, for quantized data, the complete-bin count. Events are
/// stored at bin indices in both layouts.
///
/// # Errors
///
/// Returns an error for inconsistent dimensions, impossible events, malformed
/// or trailing packet data, or an unrepresentable output raster.
pub fn decode_raster(
    packet: &[u8],
    rows: usize,
    channels: usize,
    precision: usize,
) -> Result<Vec<i8>, &'static str> {
    let mut reader = PacketReader::new(packet);
    let huffman = packet.first() == Some(&1);
    if huffman {
        reader.take(1)?;
    }
    let original_rows = reader.u32()? as usize;
    let original_channels = reader.u32()? as usize;
    if precision == 0 {
        return Err("invalid ISI timing precision");
    }
    let bins = original_rows / precision;
    if original_channels != channels || (rows != original_rows && rows != bins) {
        return Err("ISI dimensions do not match the packet");
    }
    let mut events = Vec::new();
    let push = |time, channel| events.push((time, channel));
    if huffman {
        decode_huffman(reader, channels, bins, push)?;
    } else {
        decode_varint(reader, channels, bins, push)?;
    }
    let cells = rows
        .checked_mul(channels)
        .ok_or("ISI raster size overflow")?;
    let mut raster = Vec::new();
    raster
        .try_reserve_exact(cells)
        .map_err(|_| "ISI raster allocation failed")?;
    raster.resize(cells, 0);
    for (time, channel) in events {
        raster[time * channels + channel] = 1;
    }
    Ok(raster)
}
