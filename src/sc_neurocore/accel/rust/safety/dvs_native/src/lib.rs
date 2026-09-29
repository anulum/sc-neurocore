// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Converted DVS NPY recording

//! Read bounded real numeric NPY camera recordings into independent row-major doubles.
mod dtype;
mod header;
mod literals;
mod strings;
use std::{
    fs::File,
    io::{self, Read},
    path::Path,
};

pub(crate) fn invalid(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

/// Read one converted NPY camera recording with x, y, polarity and millisecond columns.
///
/// Versions 1.0/2.0/3.0, C/Fortran storage and either byte order preserve stored
/// real numeric values through float64 conversion. The budget bounds returned
/// doubles; source buffers and temporary allocations are additional memory.
/// Geometry, finiteness, time ordering and manifest identity belong to callers.
/// No download, pickle or synthetic substitution occurs.
///
/// # Errors
/// Refuses invalid headers/dtypes/shapes/budgets, truncation, trailing content,
/// allocation failure and unreadable files. Extended conversion requires the
/// host C long-double ABI to occupy 16 bytes; its representation is host-specific.
pub fn read_dvs_recording(path: impl AsRef<Path>, maximum_bytes: usize) -> io::Result<Vec<f64>> {
    if maximum_bytes > isize::MAX as usize {
        return Err(invalid("DVS budget exceeds native integer"));
    }
    let mut file = File::open(path)?;
    let mut prefix = [0u8; 8];
    file.read_exact(&mut prefix)?;
    if &prefix[..6] != b"\x93NUMPY" || !(1..=3).contains(&prefix[6]) || prefix[7] != 0 {
        return Err(invalid("invalid DVS NPY preamble"));
    }
    let width = if prefix[6] == 1 { 2 } else { 4 };
    let mut size = [0u8; 4];
    file.read_exact(&mut size[..width])?;
    let length = u32::from_le_bytes(size) as usize;
    if length == 0 || length > 10000 {
        return Err(invalid("invalid DVS header length"));
    }
    let mut raw = vec![0u8; length];
    file.read_exact(&mut raw)?;
    if raw.last() != Some(&b'\n') {
        return Err(invalid("DVS header requires final newline"));
    }
    let text = if prefix[6] == 3 {
        std::str::from_utf8(&raw)
            .map_err(|_| invalid("invalid DVS UTF8 header"))?
            .to_owned()
    } else {
        raw.iter().map(|b| char::from(*b)).collect()
    };
    let metadata = header::parse(&text)?;
    if metadata.rows > maximum_bytes / 32 {
        return Err(invalid("DVS recording exceeds event budget"));
    }
    let count = metadata
        .rows
        .checked_mul(4)
        .ok_or_else(|| invalid("DVS event count overflow"))?;
    let bytes = count
        .checked_mul(metadata.dtype.width)
        .filter(|n| *n <= isize::MAX as usize)
        .ok_or_else(|| invalid("DVS source payload exceeds native size"))?;
    let mut payload = Vec::new();
    payload.try_reserve_exact(bytes).map_err(io::Error::other)?;
    payload.resize(bytes, 0);
    file.read_exact(&mut payload)?;
    let mut extra = [0u8; 1];
    if file.read(&mut extra)? != 0 {
        return Err(invalid("DVS recording contains extra content"));
    }
    let mut events = Vec::new();
    events.try_reserve_exact(count).map_err(io::Error::other)?;
    events.resize(count, 0.0);
    for (index, scalar) in payload.chunks_exact(metadata.dtype.width).enumerate() {
        let destination = if metadata.fortran {
            (index % metadata.rows) * 4 + index / metadata.rows
        } else {
            index
        };
        events[destination] = metadata.dtype.decode(scalar);
    }
    Ok(events)
}
