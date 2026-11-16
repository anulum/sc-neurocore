// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Fixed-point LIF integer masking helpers

/// Mask and sign-interpret an integer to `width` bits, then narrow to `i16`.
///
/// Widths above 16 retain the low 16 bits in this public narrow-value helper.
///
/// # Panics
///
/// Panics when `width` is 0 or above 32.
#[inline]
pub fn mask(value: i32, width: u32) -> i16 {
    mask_wide(value, width) as i16
}

/// Interpret a wide intermediate without narrowing before fractional scaling.
#[inline]
pub(super) fn mask_wide(value: i32, width: u32) -> i32 {
    assert!(
        width > 0 && width <= 32,
        "mask width must be 1..=32, got {width}"
    );
    let shift = 32 - width;
    value.wrapping_shl(shift) >> shift
}
