// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Published N-MNIST records and native C interface

//! Decode Orchard et al. (2015) 40-bit records without narrowing timestamps.

use std::io;
use std::mem::{align_of, size_of};
use std::path::Path;

/// Write row-major x, y, polarity and float64 millisecond times.
///
/// All lengths are validated before any destination value changes.
pub fn decode_nmnist_into(raw: &[u8], output: &mut [f64]) -> Result<(), &'static str> {
    if !raw.len().is_multiple_of(5) {
        return Err("N-MNIST file contains an incomplete 40-bit event");
    }
    if !output.len().is_multiple_of(4) || output.len() / 4 != raw.len() / 5 {
        return Err("N-MNIST output must have four columns per event");
    }
    for (record, row) in raw
        .as_chunks::<5>()
        .0
        .iter()
        .zip(output.as_chunks_mut::<4>().0.iter_mut())
    {
        row[0] = f64::from(record[0]);
        row[1] = f64::from(record[1]);
        row[2] = f64::from(record[2] >> 7);
        let time_us = (u32::from(record[2] & 0x7f) << 16)
            | (u32::from(record[3]) << 8)
            | u32::from(record[4]);
        row[3] = f64::from(time_us) / 1000.0;
    }
    Ok(())
}

/// Read one complete recording; propagate I/O errors and refuse truncated events.
pub fn read_nmnist(path: &Path) -> io::Result<Vec<f64>> {
    let raw = std::fs::read(path)?;
    if !raw.len().is_multiple_of(5) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "N-MNIST file contains an incomplete 40-bit event",
        ));
    }
    let mut output = vec![0.0; raw.len() / 5 * 4];
    decode_nmnist_into(&raw, &mut output)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    Ok(output)
}

/// Decode into caller-owned memory, returning zero on success and -1 on refusal.
///
/// Empty records accept null pointers. Invalid lengths, null pointers, alignment,
/// address overflow and overlapping buffers are refused before output mutation.
///
/// # Safety
///
/// Nonempty pointers must reference live allocations containing the declared
/// numbers of bytes and doubles. The output allocation must be writable and
/// exclusively accessible for this call; the input must remain immutable.
#[no_mangle]
pub unsafe extern "C" fn nmnist_decode_c(
    raw: *const u8,
    byte_count: usize,
    output: *mut f64,
    value_count: usize,
) -> i32 {
    if byte_count > isize::MAX as usize
        || value_count > isize::MAX as usize / size_of::<f64>()
        || !byte_count.is_multiple_of(5)
        || !value_count.is_multiple_of(4)
        || value_count / 4 != byte_count / 5
    {
        return -1;
    }
    if byte_count == 0 {
        return 0;
    }
    if raw.is_null() || output.is_null() || !(output as usize).is_multiple_of(align_of::<f64>()) {
        return -1;
    }
    let input_start = raw as usize;
    let output_start = output as usize;
    let Some(input_end) = input_start.checked_add(byte_count) else {
        return -1;
    };
    let Some(output_end) = output_start.checked_add(value_count * size_of::<f64>()) else {
        return -1;
    };
    if input_start < output_end && output_start < input_end {
        return -1;
    }
    // The caller supplies live allocations; size, alignment and aliasing were checked above.
    let input = unsafe { std::slice::from_raw_parts(raw, byte_count) };
    // The caller grants exclusive writable access to this non-overlapping destination.
    let destination = unsafe { std::slice::from_raw_parts_mut(output, value_count) };
    match decode_nmnist_into(input, destination) {
        Ok(()) => 0,
        Err(_) => -1,
    }
}

#[cfg(test)]
mod tests {
    use super::{decode_nmnist_into, read_nmnist};

    #[test]
    fn recorded_fields_and_fractional_milliseconds_are_exact() {
        let raw = [
            33, 32, 0x80, 3, 234, 1, 0, 0, 7, 212, 0, 33, 0x7f, 0xff, 0xff,
        ];
        let mut output = [0.0; 12];
        decode_nmnist_into(&raw, &mut output).unwrap();
        assert_eq!(
            output,
            [33.0, 32.0, 1.0, 1.002, 1.0, 0.0, 0.0, 2.004, 0.0, 33.0, 0.0, 8388.607]
        );
    }

    #[test]
    fn refused_records_do_not_mutate_output() {
        let mut output = [9.0; 4];
        assert!(decode_nmnist_into(&[0; 6], &mut output).is_err());
        assert!(decode_nmnist_into(&[0; 5], &mut output[..3]).is_err());
        assert_eq!(output, [9.0; 4]);
        assert_eq!(decode_nmnist_into(&[], &mut []), Ok(()));
    }

    #[test]
    fn actual_recording_file_and_missing_file_errors() {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "sc-neurocore-nmnist-{}-{nonce}.bin",
            std::process::id()
        ));
        std::fs::write(&path, [33, 32, 0x80, 3, 234]).unwrap();
        assert_eq!(read_nmnist(&path).unwrap(), [33.0, 32.0, 1.0, 1.002]);
        std::fs::write(&path, [0; 6]).unwrap();
        assert_eq!(
            read_nmnist(&path).unwrap_err().kind(),
            std::io::ErrorKind::InvalidData
        );
        std::fs::remove_file(&path).unwrap();
        assert_eq!(
            read_nmnist(&path).unwrap_err().kind(),
            std::io::ErrorKind::NotFound
        );
    }
}
