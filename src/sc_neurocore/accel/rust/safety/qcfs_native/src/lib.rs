// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust QCFS C library

//! C library exposing the maintained QCFS kernel over float64 arrays.
//!
//! Every call validates the grid, threshold and every span before it writes,
//! so a refusal leaves all outputs unchanged. Elements are read and written
//! one at a time through raw pointers: an output may be the same array as an
//! input, but the two outputs of `sc_qcfs_backward` must not overlap.
#![deny(missing_docs)]

#[path = "../../qcfs.rs"]
pub mod kernel;

use kernel::QCFSActivation;
use std::mem::{align_of, size_of};

/// Admit one complete aligned float64 span in the signed address domain.
fn span(pointer: *const f64, count: usize) -> bool {
    if count == 0 {
        return true;
    }
    let address = pointer as usize;
    let Some(bytes) = count.checked_mul(size_of::<f64>()) else {
        return false;
    };
    address != 0
        && address.is_multiple_of(align_of::<f64>())
        && bytes <= isize::MAX as usize
        && address
            .checked_add(bytes)
            .is_some_and(|end| end <= isize::MAX as usize)
}

/// Report the exact QCFS array ABI version.
#[no_mangle]
pub extern "C" fn sc_qcfs_abi_version() -> u32 {
    1
}

/// Quantise `count` activations into `output`. Returns 0, or -1 for an
/// invalid grid, threshold or span with every output unchanged.
///
/// # Safety
/// Nonzero spans must denote live aligned float64 arrays of `count` elements
/// that stay valid for the call; spans are checked, accessibility is not.
#[no_mangle]
pub unsafe extern "C" fn sc_qcfs_forward(
    steps: u32,
    theta: f64,
    x: *const f64,
    count: usize,
    output: *mut f64,
) -> i32 {
    let Ok(state) = QCFSActivation::with_parameters(steps, theta) else {
        return -1;
    };
    if !span(x, count) || !span(output, count) {
        return -1;
    }
    for index in 0..count {
        let value = unsafe { x.add(index).read() };
        let Ok(result) = state.forward(value) else {
            return -1;
        };
        unsafe { output.add(index).write(result) };
    }
    0
}

/// Write each element's input and threshold derivative. Returns 0, or -1 for
/// an invalid grid, threshold or span with every output unchanged.
///
/// # Safety
/// Nonzero spans must denote live aligned float64 arrays of `count` elements
/// that stay valid for the call; the two outputs must not overlap.
#[no_mangle]
pub unsafe extern "C" fn sc_qcfs_backward(
    steps: u32,
    theta: f64,
    x: *const f64,
    upstream: *const f64,
    count: usize,
    input_gradient: *mut f64,
    threshold_gradient: *mut f64,
) -> i32 {
    let Ok(state) = QCFSActivation::with_parameters(steps, theta) else {
        return -1;
    };
    if ![x, upstream, input_gradient, threshold_gradient]
        .iter()
        .all(|&pointer| span(pointer, count))
    {
        return -1;
    }
    for index in 0..count {
        let (value, gradient) = unsafe { (x.add(index).read(), upstream.add(index).read()) };
        let Ok((input, threshold)) = state.backward(value, gradient) else {
            return -1;
        };
        unsafe {
            input_gradient.add(index).write(input);
            threshold_gradient.add(index).write(threshold);
        }
    }
    0
}
