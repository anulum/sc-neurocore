// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Independent spike-raster frame encoding and parsing

use super::RasterError;

pub(super) const HEADER_SIZE: usize = 4;

pub(super) struct Frame<'a> {
    pub(super) channels: usize,
    pub(super) rows: usize,
    bitmap: &'a [u8],
    masks: &'a [u8],
}

impl<'a> Frame<'a> {
    pub(super) fn parse(packet: &'a [u8]) -> Result<(Self, &'a [u8]), RasterError> {
        let [n_hi, n_lo, w_hi, w_lo, body @ ..] = packet else {
            return Err("truncated raster frame header");
        };
        let channels = usize::from(u16::from_be_bytes([*n_hi, *n_lo]));
        let rows = usize::from(u16::from_be_bytes([*w_hi, *w_lo]));
        let (bitmap, body) = body
            .split_at_checked(channels.div_ceil(8))
            .ok_or("truncated raster frame bitmap")?;
        let active = (0..channels)
            .filter(|&channel| !silent(bitmap, channel))
            .count();
        let (masks, remaining) = body
            .split_at_checked(active * rows.div_ceil(8))
            .ok_or("truncated raster frame mask")?;
        Ok((
            Self {
                channels,
                rows,
                bitmap,
                masks,
            },
            remaining,
        ))
    }

    pub(super) fn validate_padding(&self, valid_rows: usize) -> Result<(), RasterError> {
        let complete = valid_rows / 8;
        let bits = valid_rows % 8;
        let keep = ((1u16 << bits) - 1) as u8;
        for mask in self.masks.chunks_exact(self.rows.div_ceil(8).max(1)) {
            if mask.get(complete).is_some_and(|byte| byte & !keep != 0)
                || mask.iter().skip(complete + 1).any(|&byte| byte != 0)
            {
                return Err("nonzero raster frame padding");
            }
        }
        Ok(())
    }

    pub(super) fn decode_into(&self, output: &mut [i8]) {
        let stride = self.rows.div_ceil(8);
        if output.is_empty() {
            return;
        }
        let rows = output.len() / self.channels;
        let mut masks = self.masks.chunks_exact(stride);
        for channel in (0..self.channels).filter(|&channel| !silent(self.bitmap, channel)) {
            let Some(mask) = masks.next() else {
                break;
            };
            for (byte_index, &byte) in mask.iter().take(rows.div_ceil(8)).enumerate() {
                let mut bits = byte;
                while bits != 0 {
                    let row = byte_index * 8 + bits.trailing_zeros() as usize;
                    if row < rows {
                        output[row * self.channels + channel] = 1;
                    }
                    bits &= bits - 1;
                }
            }
        }
    }
}

fn silent(bitmap: &[u8], channel: usize) -> bool {
    (bitmap[channel / 8] & (1 << (channel % 8))) != 0
}

pub(super) fn append(
    output: &mut Vec<u8>,
    raster: &[i8],
    rows: usize,
    channels: usize,
    stored_rows: usize,
) -> Result<(usize, usize), RasterError> {
    let width = u16::try_from(stored_rows).map_err(|_| "raster frame rows exceed uint16")?;
    let count = u16::try_from(channels).map_err(|_| "raster frame channels exceed uint16")?;
    let bitmap_size = channels.div_ceil(8);
    let stride = stored_rows.div_ceil(8);
    let start = output.len();
    output.extend_from_slice(&count.to_be_bytes());
    output.extend_from_slice(&width.to_be_bytes());
    output.resize(start + HEADER_SIZE + bitmap_size, 0);
    let mut mask = vec![0u8; stride];
    let mut active = 0;
    let mut spikes = 0;
    for channel in 0..channels {
        mask.fill(0);
        let mut channel_spikes = 0;
        for row in 0..rows {
            let sample = raster[row * channels + channel];
            match sample {
                0 => {}
                1 => {
                    channel_spikes += 1;
                    mask[row / 8] |= 1 << (row % 8);
                }
                _ => return Err("raster must contain only zeros and ones"),
            }
        }
        if channel_spikes != 0 {
            spikes += channel_spikes;
            active += 1;
            output.extend_from_slice(&mask);
        } else {
            output[start + HEADER_SIZE + channel / 8] |= 1 << (channel % 8);
        }
    }
    Ok((spikes, active))
}
