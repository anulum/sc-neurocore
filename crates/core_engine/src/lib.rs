// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Core Engine FFI Surface
// Authored by Anulum Fortis & Arcane Sapience (protoscience@anulum.li)

//! # Core Engine
//!
//! C-FFI surface exposing stochastic computing arithmetic to Python and Go
//! consumers. All operations work on packed `u64` bitstream slices, matching
//! the `engine::BitStreamTensor` memory layout.

pub mod bitstream;

use std::slice;

use bitstream::{Bitstream, Lfsr16};

// ---------------------------------------------------------------------------
// Scalar SC arithmetic (C-FFI)
// ---------------------------------------------------------------------------

/// SC multiplication: bitwise AND of two packed u32 words.
#[no_mangle]
pub extern "C" fn sc_multiply(a: u32, b: u32) -> u32 {
    a & b
}

/// SC scaled addition (MUX): out = (a & sel) | (b & !sel).
#[no_mangle]
pub extern "C" fn sc_mux(a: u32, b: u32, sel: u32) -> u32 {
    (a & sel) | (b & !sel)
}

/// Population count of a u32 word.
#[no_mangle]
pub extern "C" fn sc_popcount(a: u32) -> u32 {
    a.count_ones()
}

/// Population count of a u64 word.
#[no_mangle]
pub extern "C" fn sc_popcount64(a: u64) -> u32 {
    a.count_ones()
}

/// SC saturating subtraction: a AND NOT(b).
#[no_mangle]
pub extern "C" fn sc_saturating_sub(a: u32, b: u32) -> u32 {
    a & !b
}

// ---------------------------------------------------------------------------
// Packed bitstream operations (C-FFI, operates on u64 arrays)
// ---------------------------------------------------------------------------

/// Bitwise AND of two packed u64 arrays into `out`. All arrays must have
/// `word_count` elements. Caller owns all memory.
///
/// # Safety
/// Pointers must be valid for `word_count` elements.
#[no_mangle]
pub unsafe extern "C" fn sc_and_packed(
    a_ptr: *const u64,
    b_ptr: *const u64,
    out_ptr: *mut u64,
    word_count: usize,
) {
    let a = unsafe { slice::from_raw_parts(a_ptr, word_count) };
    let b = unsafe { slice::from_raw_parts(b_ptr, word_count) };
    let out = unsafe { slice::from_raw_parts_mut(out_ptr, word_count) };
    for i in 0..word_count {
        out[i] = a[i] & b[i];
    }
}

/// Bitwise MUX of two packed u64 arrays using a select bitstream.
///
/// # Safety
/// All pointers must be valid for `word_count` elements.
#[no_mangle]
pub unsafe extern "C" fn sc_mux_packed(
    a_ptr: *const u64,
    b_ptr: *const u64,
    sel_ptr: *const u64,
    out_ptr: *mut u64,
    word_count: usize,
) {
    let a = unsafe { slice::from_raw_parts(a_ptr, word_count) };
    let b = unsafe { slice::from_raw_parts(b_ptr, word_count) };
    let sel = unsafe { slice::from_raw_parts(sel_ptr, word_count) };
    let out = unsafe { slice::from_raw_parts_mut(out_ptr, word_count) };
    for i in 0..word_count {
        out[i] = (a[i] & sel[i]) | (b[i] & !sel[i]);
    }
}

/// Population count over a packed u64 array.
///
/// # Safety
/// `data_ptr` must be valid for `word_count` elements.
#[no_mangle]
pub unsafe extern "C" fn sc_popcount_packed(data_ptr: *const u64, word_count: usize) -> u64 {
    let data = unsafe { slice::from_raw_parts(data_ptr, word_count) };
    data.iter().map(|w| w.count_ones() as u64).sum()
}

/// Compute SCC between two packed bitstream arrays.
///
/// # Safety
/// Both pointers must be valid for `word_count` elements.
/// `bit_length` is the logical bitstream length (may be < word_count * 64).
#[no_mangle]
pub unsafe extern "C" fn sc_scc_packed(
    a_ptr: *const u64,
    b_ptr: *const u64,
    word_count: usize,
    bit_length: usize,
) -> f64 {
    let a_data = unsafe { slice::from_raw_parts(a_ptr, word_count) }.to_vec();
    let b_data = unsafe { slice::from_raw_parts(b_ptr, word_count) }.to_vec();
    let a = Bitstream::from_words(a_data, bit_length);
    let b = Bitstream::from_words(b_data, bit_length);
    bitstream::scc(&a, &b)
}

// ---------------------------------------------------------------------------
// LFSR encoding (C-FFI)
// ---------------------------------------------------------------------------

/// Create an LFSR encoder instance. Returns an opaque pointer.
/// Caller must free with `lfsr_destroy`.
#[no_mangle]
pub extern "C" fn lfsr_create(seed: u16) -> *mut Lfsr16 {
    Box::into_raw(Box::new(Lfsr16::new(seed)))
}

/// Advance the LFSR one step. Returns new register value.
///
/// # Safety
/// `ptr` must have been returned by `lfsr_create` and not yet destroyed.
#[no_mangle]
pub unsafe extern "C" fn lfsr_step(ptr: *mut Lfsr16) -> u16 {
    if ptr.is_null() {
        return 0;
    }
    let lfsr = unsafe { &mut *ptr };
    lfsr.step()
}

/// Encode a bitstream. Allocates a new array; caller must free with `bitstream_free`.
/// Returns the packed u64 data via `out_ptr` and word count via `out_words`.
///
/// # Safety
/// `lfsr_ptr` must be valid. `out_ptr` and `out_words` must be valid pointers.
#[no_mangle]
pub unsafe extern "C" fn lfsr_encode(
    lfsr_ptr: *mut Lfsr16,
    threshold: u16,
    length: usize,
    out_ptr: *mut *mut u64,
    out_words: *mut usize,
) {
    if lfsr_ptr.is_null() {
        return;
    }
    let lfsr = unsafe { &mut *lfsr_ptr };
    let bs = lfsr.encode(threshold, length);

    let mut boxed = bs.data.into_boxed_slice();
    let words = boxed.len();
    let ptr = boxed.as_mut_ptr();
    std::mem::forget(boxed);

    unsafe {
        *out_ptr = ptr;
        *out_words = words;
    }
}

/// Free a bitstream data array returned by `lfsr_encode`.
///
/// # Safety
/// `ptr` and `word_count` must match a previous `lfsr_encode` call.
#[no_mangle]
pub unsafe extern "C" fn bitstream_free(ptr: *mut u64, word_count: usize) {
    if !ptr.is_null() {
        let _ = unsafe { Vec::from_raw_parts(ptr, word_count, word_count) };
    }
}

/// Destroy an LFSR instance.
///
/// # Safety
/// `ptr` must have been returned by `lfsr_create`.
#[no_mangle]
pub unsafe extern "C" fn lfsr_destroy(ptr: *mut Lfsr16) {
    if !ptr.is_null() {
        let _ = unsafe { Box::from_raw(ptr) };
    }
}

// ---------------------------------------------------------------------------
// CORDIV (C-FFI)
// ---------------------------------------------------------------------------

/// Perform CORDIV stochastic division on packed bitstreams.
/// Allocates result; caller frees with `bitstream_free`.
///
/// # Safety
/// `x_ptr` and `y_ptr` must be valid for `word_count` elements.
#[no_mangle]
pub unsafe extern "C" fn sc_cordiv_packed(
    x_ptr: *const u64,
    y_ptr: *const u64,
    word_count: usize,
    bit_length: usize,
    out_ptr: *mut *mut u64,
    out_words: *mut usize,
) {
    let x_data = unsafe { slice::from_raw_parts(x_ptr, word_count) }.to_vec();
    let y_data = unsafe { slice::from_raw_parts(y_ptr, word_count) }.to_vec();
    let x = Bitstream::from_words(x_data, bit_length);
    let y = Bitstream::from_words(y_data, bit_length);
    let result = bitstream::cordiv(&x, &y);

    let mut boxed = result.data.into_boxed_slice();
    let words = boxed.len();
    let ptr = boxed.as_mut_ptr();
    std::mem::forget(boxed);

    unsafe {
        *out_ptr = ptr;
        *out_words = words;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_multiply() {
        assert_eq!(sc_multiply(0b1010, 0b1100), 0b1000);
    }

    #[test]
    fn scalar_mux() {
        assert_eq!(sc_mux(0xFF, 0x00, 0x0F), 0x0F);
        assert_eq!(sc_mux(0xFF, 0x00, 0xF0), 0xF0);
    }

    #[test]
    fn scalar_popcount() {
        assert_eq!(sc_popcount(0b1010_1010), 4);
        assert_eq!(sc_popcount64(0xFFFF_FFFF_FFFF_FFFF), 64);
    }

    #[test]
    fn scalar_saturating_sub() {
        assert_eq!(sc_saturating_sub(0b1110, 0b0110), 0b1000);
    }

    #[test]
    fn packed_and_roundtrip() {
        let a = [0xAAAA_AAAA_AAAA_AAAAu64];
        let b = [0x5555_5555_5555_5555u64];
        let mut out = [0u64];
        unsafe { sc_and_packed(a.as_ptr(), b.as_ptr(), out.as_mut_ptr(), 1) };
        assert_eq!(out[0], 0);
    }

    #[test]
    fn packed_popcount() {
        let data = [u64::MAX, u64::MAX];
        let count = unsafe { sc_popcount_packed(data.as_ptr(), 2) };
        assert_eq!(count, 128);
    }

    #[test]
    fn lfsr_lifecycle() {
        let ptr = lfsr_create(0xACE1);
        assert!(!ptr.is_null());
        let val = unsafe { lfsr_step(ptr) };
        assert_ne!(val, 0);
        unsafe { lfsr_destroy(ptr) };
    }
}
