// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust owned dense IF replay C boundary

//! Borrowed requests and opaque owned results; refusal never writes caller outputs.

use crate::kernel::{ConvertedSNN, DenseLayer, OutputMode, ReplayError, ReplayResult};
use crate::types::{BufferView, ReplayRequest};
use std::ffi::c_void;
use std::mem::{align_of, size_of};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::slice;

fn span<T>(pointer: *const T, count: usize) -> bool {
    if count == 0 {
        return true;
    }
    let address = pointer as usize;
    let Some(bytes) = count.checked_mul(size_of::<T>()) else {
        return false;
    };
    address != 0
        && address.is_multiple_of(align_of::<T>())
        && bytes <= isize::MAX as usize
        && address
            .checked_add(bytes)
            .is_some_and(|end| end <= isize::MAX as usize)
}

unsafe fn borrowed<'a, T>(pointer: *const T, count: usize) -> Result<&'a [T], ReplayError> {
    if !span(pointer, count) {
        return Err(ReplayError::InvalidInput);
    }
    if count == 0 {
        return Ok(&[]);
    }
    Ok(unsafe { slice::from_raw_parts(pointer, count) })
}

fn status(error: ReplayError) -> i32 {
    match error {
        ReplayError::InvalidInput => -1,
        ReplayError::ResourceLimit => -2,
        ReplayError::Overflow => -3,
    }
}

unsafe fn execute(request: &ReplayRequest) -> Result<ReplayResult, ReplayError> {
    if request.version != 1
        || request.flags & !15 != 0
        || request.layer_count == 0
        || request.max_working_bytes == 0
        || request.max_working_bytes > isize::MAX as usize
    {
        return Err(ReplayError::InvalidInput);
    }
    if request.layer_count > request.max_working_bytes / 16 {
        return Err(ReplayError::ResourceLimit);
    }
    let specs = unsafe { borrowed(request.layers, request.layer_count)? };
    let mut coefficients = 0usize;
    for layer in specs {
        if layer.inputs == 0
            || layer.outputs == 0
            || (layer.bias_len != 0 && layer.bias_len != layer.outputs)
        {
            return Err(ReplayError::InvalidInput);
        }
        let count = layer
            .inputs
            .checked_mul(layer.outputs)
            .ok_or(ReplayError::InvalidInput)?;
        coefficients = coefficients
            .checked_add(count)
            .and_then(|v| v.checked_add(layer.bias_len))
            .ok_or(ReplayError::ResourceLimit)?;
    }
    if coefficients
        .checked_mul(16)
        .is_none_or(|bytes| bytes > request.max_working_bytes)
    {
        return Err(ReplayError::ResourceLimit);
    }
    let mut layers = Vec::with_capacity(specs.len());
    for spec in specs {
        let weight = unsafe { borrowed(spec.weights, spec.inputs * spec.outputs)? };
        let bias = if spec.bias_len == 0 {
            None
        } else {
            Some(unsafe { borrowed(spec.bias, spec.bias_len)? }.to_vec())
        };
        layers.push(
            DenseLayer::new(
                spec.inputs,
                spec.outputs,
                weight.to_vec(),
                bias,
                spec.threshold,
            )?
            .with_initial_fraction(spec.initial_fraction)?,
        );
    }
    let mode = if request.flags & 4 != 0 {
        OutputMode::Linear
    } else {
        OutputMode::Spikes
    };
    let network = ConvertedSNN::new(layers, 0.0, mode)?;
    let frames = unsafe { borrowed(request.frames, request.frames_len)? };
    let initial = if request.flags & 8 != 0 {
        Some(
            specs
                .iter()
                .map(|spec| unsafe { borrowed(spec.initial, spec.initial_len) })
                .collect::<Result<Vec<_>, _>>()?,
        )
    } else {
        None
    };
    network.replay_borrowed(
        frames,
        (request.steps, request.batch),
        initial.as_deref(),
        request.flags & 1 != 0,
        request.flags & 2 != 0,
        request.max_working_bytes,
    )
}

/// Report the stable ownership/buffer ABI version.
#[no_mangle]
pub extern "C" fn sc_if_abi_version() -> u32 {
    1
}

/// Replay a borrowed request and publish one opaque independently owned result.
///
/// Returns zero on success; -1 invalid input, -2 numeric reservation refusal,
/// -3 finite arithmetic overflow, -4 internal panic. Failure leaves `result` untouched.
///
/// # Safety
/// Request and result slots must be live aligned readable/writable objects.
/// Every nonempty input pointer must denote its complete declared live aligned
/// array, unchanged during this call. Result slot must be exclusive. Pointer
/// checks reject null/misalignment/address wrap; they cannot verify OS accessibility.
#[no_mangle]
pub unsafe extern "C" fn sc_if_replay(
    request: *const ReplayRequest,
    result: *mut *mut c_void,
) -> i32 {
    if !span(request, 1) || !span(result, 1) {
        return -1;
    }
    let outcome = catch_unwind(AssertUnwindSafe(|| unsafe { execute(&*request) }));
    match outcome {
        Ok(Ok(value)) => {
            unsafe {
                result.write(Box::into_raw(Box::new(value)).cast());
            }
            0
        }
        Ok(Err(error)) => status(error),
        Err(_) => -4,
    }
}

/// Borrow an owned result buffer: kind zero output, one final state, two state trace,
/// three spike trace. Index selects a layer and must be zero for output.
/// Returns zero on success or -1 without changing the view on invalid kind/index.
///
/// # Safety
/// Handle must be a live unmodified result from `sc_if_replay`, never concurrently
/// released. View is exclusive writable aligned storage. Buffer is valid until
/// `sc_if_free`; the caller must keep its owner alive while accessing any view.
#[no_mangle]
pub unsafe extern "C" fn sc_if_buffer(
    handle: *mut c_void,
    kind: u32,
    index: usize,
    view: *mut BufferView,
) -> i32 {
    if !span(handle.cast::<ReplayResult>(), 1) || !span(view, 1) {
        return -1;
    }
    let result = unsafe { &*handle.cast::<ReplayResult>() };
    let buffer = match kind {
        0 if index == 0 => Some(&result.output),
        1 => result.final_state.get(index),
        2 => result.state_trace.get(index),
        3 => result.spike_trace.get(index),
        _ => None,
    };
    let Some(buffer) = buffer else {
        return -1;
    };
    unsafe {
        view.write(BufferView {
            data: buffer.as_ptr().cast_mut(),
            len: buffer.len(),
        });
    }
    0
}

/// Release an opaque result after all borrowed views have expired. Null is a no-op.
///
/// # Safety
/// Non-null handle must come from this library and must be released exactly once;
/// no view may be accessed afterward or concurrently with release.
#[no_mangle]
pub unsafe extern "C" fn sc_if_free(handle: *mut c_void) {
    if !handle.is_null() {
        unsafe {
            drop(Box::from_raw(handle.cast::<ReplayResult>()));
        }
    }
}
