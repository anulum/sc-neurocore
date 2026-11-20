// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Inter-channel delta packets

//! Encode and decode DSCX packets using group-local references and ISI residuals.

#![warn(missing_docs)]

use std::cmp::Reverse;

use super::isi::{decode_raster as decode_isi, encode_raster as encode_isi, Entropy};

const MAGIC: &[u8; 4] = b"DSCX";
const HEADER_BYTES: usize = 8;

/// An encoded DSCX packet and its original and residual spike counts.
pub struct EncodedDelta {
    /// Complete packet including the group headers and residual ISI body.
    pub packet: Vec<u8>,
    /// Number of spikes in the original raster.
    pub spike_count: usize,
    /// Number of spikes in the non-reference residual channels before quantisation.
    pub residual_count: usize,
}

fn group_count(channels: usize, group_size: usize) -> Result<u16, &'static str> {
    if !(1..=256).contains(&group_size) {
        return Err("group_size must be in 1..256 for uint8 reference indices");
    }
    u16::try_from(channels.div_ceil(group_size)).map_err(|_| "DSCX group count exceeds uint16")
}

/// Encode a binary row-major raster, preserving the first maximum on reference ties.
///
/// The ISI body applies the supplied timing precision to complete residual bins.
///
/// # Errors
///
/// Returns an error for invalid raster dimensions, nonbinary input, invalid
/// grouping, or an invalid or unrepresentable ISI packet.
pub fn encode_raster(
    raster: &[i8],
    rows: usize,
    channels: usize,
    group_size: usize,
    precision: usize,
    entropy: Entropy,
) -> Result<EncodedDelta, &'static str> {
    let groups = group_count(channels, group_size)?;
    if u32::try_from(rows).is_err() || rows.checked_mul(channels) != Some(raster.len()) {
        return Err("invalid DSCX raster dimensions");
    }
    let row_width = channels.max(1);
    let mut counts = vec![0; channels];
    for row in raster.chunks_exact(row_width) {
        for (count, &sample) in counts.iter_mut().zip(row) {
            if !matches!(sample, 0 | 1) {
                return Err("DSCX raster must contain only zeros and ones");
            }
            *count += sample as usize;
        }
    }
    let references: Vec<u8> = counts
        .chunks(group_size)
        .map(|group| {
            group
                .iter()
                .enumerate()
                .max_by_key(|&(channel, count)| (count, Reverse(channel)))
                .map_or(0, |(channel, _)| channel as u8)
        })
        .collect();
    let mut residual = raster.to_vec();
    let mut residual_count = 0;
    for row in residual.chunks_exact_mut(row_width) {
        for (group, &reference) in row.chunks_mut(group_size).zip(&references) {
            let reference = usize::from(reference);
            let bit = group[reference];
            for (channel, sample) in group.iter_mut().enumerate() {
                if channel != reference {
                    *sample ^= bit;
                    residual_count += *sample as usize;
                }
            }
        }
    }
    let (body, _) = encode_isi(&residual, rows, channels, precision, entropy)?;
    let mut packet = Vec::with_capacity(HEADER_BYTES + references.len() + body.len());
    packet.extend_from_slice(MAGIC);
    packet.extend_from_slice(&(group_size as u16).to_be_bytes());
    packet.extend_from_slice(&groups.to_be_bytes());
    packet.extend_from_slice(&references);
    packet.extend_from_slice(&body);
    Ok(EncodedDelta {
        packet,
        spike_count: counts.iter().sum(),
        residual_count,
    })
}

/// Decode a DSCX packet into a binary row-major raster at the requested dimensions.
///
/// Lossy packets require the matching ISI timing precision.
///
/// # Errors
///
/// Returns an error for malformed framing, an out-of-group reference, or
/// invalid residual data or output dimensions.
pub fn decode_raster(
    packet: &[u8],
    rows: usize,
    channels: usize,
    precision: usize,
) -> Result<Vec<i8>, &'static str> {
    if !packet.starts_with(MAGIC) {
        return Err("Invalid header magic for DSCX");
    }
    let header = packet.get(..HEADER_BYTES).ok_or("truncated DSCX header")?;
    let group_size = usize::from(u16::from_be_bytes([header[4], header[5]]));
    let groups = u16::from_be_bytes([header[6], header[7]]);
    if groups != group_count(channels, group_size)? {
        return Err("DSCX group count does not match the channel count");
    }
    let (references, body) = packet[HEADER_BYTES..]
        .split_at_checked(usize::from(groups))
        .ok_or("truncated DSCX references")?;
    for (&reference, start) in references.iter().zip((0..channels).step_by(group_size)) {
        if usize::from(reference) >= group_size.min(channels - start) {
            return Err("DSCX reference index is outside its group");
        }
    }
    let mut raster = decode_isi(body, rows, channels, precision)?;
    for row in raster.chunks_exact_mut(channels.max(1)) {
        for (group, &reference) in row.chunks_mut(group_size).zip(references) {
            let reference = usize::from(reference);
            let bit = group[reference];
            for (channel, sample) in group.iter_mut().enumerate() {
                if channel != reference {
                    *sample ^= bit;
                }
            }
        }
    }
    Ok(raster)
}
