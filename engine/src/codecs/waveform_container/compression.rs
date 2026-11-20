// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Bounded byte compression for waveform sections

use flate2::{Compress, Compression, Decompress, FlushCompress, FlushDecompress, Status};

use super::SectionCompression;

const ZSTD_MAGIC: [u8; 4] = [0x28, 0xb5, 0x2f, 0xfd];

pub(super) fn compress(bytes: &[u8], method: SectionCompression) -> Result<Vec<u8>, String> {
    match method {
        SectionCompression::Zlib => compress_zlib(bytes),
        SectionCompression::Zstandard => compress_zstd(bytes),
    }
}

/// Use standard zlib at level 9 so DEFLATE packets retain reference bytes.
fn compress_zlib(bytes: &[u8]) -> Result<Vec<u8>, String> {
    let n = bytes.len();
    // zlib's compressBound for a default window size and memory level.
    let bound = n
        .checked_add((n >> 12) + (n >> 14) + (n >> 25) + 13)
        .ok_or("WaveformCodec compressed section is too large")?;
    let mut output = Vec::new();
    output
        .try_reserve_exact(bound)
        .map_err(|_| "WaveformCodec section allocation failed")?;
    let mut encoder = Compress::new(Compression::best(), true);
    let status = encoder
        .compress_vec(bytes, &mut output, FlushCompress::Finish)
        .map_err(compression_error)?;
    if status != Status::StreamEnd || encoder.total_in() != n as u64 {
        return Err("zlib could not finish the WaveformCodec section".into());
    }
    Ok(output)
}

/// Decode exactly one complete frame and exactly the bytes prescribed by the section.
pub(super) fn decompress(bytes: &[u8], expected: usize) -> Result<Vec<u8>, String> {
    if bytes.starts_with(&ZSTD_MAGIC) {
        return decompress_zstd(bytes, expected);
    }
    let mut output = Vec::new();
    output
        .try_reserve_exact(expected.checked_add(1).ok_or("section size overflow")?)
        .map_err(|_| "WaveformCodec section allocation failed")?;
    let mut decoder = Decompress::new(true);
    let status = decoder
        .decompress_vec(bytes, &mut output, FlushDecompress::Finish)
        .map_err(compression_error)?;
    if status != Status::StreamEnd
        || decoder.total_in() != bytes.len() as u64
        || output.len() != expected
    {
        return Err("compressed WaveformCodec section is corrupted or has the wrong length".into());
    }
    Ok(output)
}

fn compression_error(error: impl std::fmt::Display) -> String {
    format!("compressed WaveformCodec section is corrupted: {error}")
}

#[cfg(feature = "waveform-zstd")]
fn compress_zstd(bytes: &[u8]) -> Result<Vec<u8>, String> {
    zstd::bulk::compress(bytes, 19).map_err(compression_error)
}

#[cfg(not(feature = "waveform-zstd"))]
fn compress_zstd(_bytes: &[u8]) -> Result<Vec<u8>, String> {
    Err("a zstd-compressed section needs the waveform-zstd feature".into())
}

#[cfg(feature = "waveform-zstd")]
fn decompress_zstd(bytes: &[u8], expected: usize) -> Result<Vec<u8>, String> {
    let frame_size = zstd::zstd_safe::find_frame_compressed_size(bytes)
        .map_err(|code| compression_error(zstd::zstd_safe::get_error_name(code)))?;
    if frame_size != bytes.len() {
        return Err(
            "compressed WaveformCodec section is corrupted or contains multiple frames".into(),
        );
    }
    let output = zstd::bulk::decompress(bytes, expected).map_err(compression_error)?;
    if output.len() != expected {
        return Err("compressed WaveformCodec section has the wrong length".into());
    }
    Ok(output)
}

#[cfg(not(feature = "waveform-zstd"))]
fn decompress_zstd(_bytes: &[u8], _expected: usize) -> Result<Vec<u8>, String> {
    Err("a zstd-compressed section needs the waveform-zstd feature".into())
}
