// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust indexed SHD API and C ABI

//! Native indexed SHD recording reads and the shared owning C ABI.
#![deny(unsafe_op_in_unsafe_fn)]

mod hdf5;
mod recording;

use std::{
    ffi::{c_char, CStr, CString},
    io,
    path::Path,
    ptr,
};

/// One owned row-major float64 x/y/polarity/millisecond recording and its label.
pub struct Recording {
    /// Four consecutive values per event, with auditory y and polarity zero.
    pub events: Vec<f64>,
    /// Actual integer label read from the selected HDF5 row.
    pub label: i64,
}

/// Read one local indexed SHD row, with a four-double event result byte budget.
///
/// Numeric variable-length time/channel vectors and integer labels must have
/// matching rank-one recording counts. HDF5 buffers are additional transient
/// memory; this is not an aggregate memory cap. Event geometry, finiteness and
/// manifest identity remain caller responsibilities. No download or fallback.
///
/// # Errors
/// Refuses unreadable files, incompatible shapes/types/index, excessive result
/// size, invalid paths, HDF5 errors or failed Rust allocations.
#[cfg(unix)]
pub fn read_shd_recording(
    path: &Path,
    index: usize,
    maximum_bytes: usize,
) -> io::Result<Recording> {
    use std::os::unix::ffi::OsStrExt;
    let path = CString::new(path.as_os_str().as_bytes())
        .map_err(|_| hdf5::invalid("SHD path contains NUL"))?;
    recording::read(&path, index, maximum_bytes)
}

/// Owning C result, layout-compatible with the native Go SHD shared ABI.
#[repr(C)]
pub struct ShdRecording {
    /// Allocation owned by this struct; release only through `shd_free_c`.
    pub events: *mut f64,
    /// Number of doubles in the allocation, divisible by four.
    pub value_count: usize,
    /// Actual int64 HDF5 row label.
    pub label: i64,
}

/// Read one HDF5 row into an initially zeroed caller-owned result struct.
/// Returns zero on success, -1 on refusal; refusal leaves the result unchanged.
///
/// # Safety
/// `path` must point to a live NUL-terminated string; `output` must be writable,
/// aligned and initially zeroed with no outstanding allocation. Passing null
/// refuses safely. Free successful results exactly once; do not copy ownership.
#[no_mangle]
pub unsafe extern "C" fn shd_read_c(
    path: *const c_char,
    index: usize,
    maximum_bytes: usize,
    output: *mut ShdRecording,
) -> i32 {
    if path.is_null()
        || output.is_null()
        || index > isize::MAX as usize
        || maximum_bytes > isize::MAX as usize
    {
        return -1;
    }
    let destination = unsafe { &mut *output };
    if !destination.events.is_null() || destination.value_count != 0 || destination.label != 0 {
        return -1;
    }
    let path = unsafe { CStr::from_ptr(path) };
    let sample = match recording::read(path, index, maximum_bytes) {
        Ok(sample) => sample,
        Err(_) => return -1,
    };
    let values = sample.events.into_boxed_slice();
    let count = values.len();
    let events = if count == 0 {
        ptr::null_mut()
    } else {
        Box::into_raw(values).cast::<f64>()
    };
    *destination = ShdRecording {
        events,
        value_count: count,
        label: sample.label,
    };
    0
}

/// Release a successful result, resetting it so repeated release is safe.
///
/// # Safety
/// A non-null argument must be a live writable struct whose allocation was
/// produced by this library. Owning structs cannot be copied or freed elsewhere.
#[no_mangle]
pub unsafe extern "C" fn shd_free_c(output: *mut ShdRecording) {
    let Some(result) = (unsafe { output.as_mut() }) else {
        return;
    };
    if !result.events.is_null() {
        let values = ptr::slice_from_raw_parts_mut(result.events, result.value_count);
        unsafe {
            drop(Box::from_raw(values));
        }
    }
    *result = ShdRecording {
        events: ptr::null_mut(),
        value_count: 0,
        label: 0,
    };
}
