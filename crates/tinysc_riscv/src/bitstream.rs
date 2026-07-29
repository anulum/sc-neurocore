// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — tinySC Bitstream Primitives (no_std)

//! Packed bitstream operations for bare-metal SC.
//!
//! Uses `u32` words (MCU-native) instead of `u64` to minimise register
//! pressure on RV32. All operations are `#[inline]` to enable the
//! compiler to vectorise on RV64+RVV targets.

/// Population count of a single u32 word.
///
/// On targets with the Zbb extension this compiles to a single `cpop`
/// instruction. Falls back to the compiler intrinsic otherwise.
#[inline(always)]
pub fn popcount32(word: u32) -> u32 {
    #[cfg(feature = "custom-popcount")]
    {
        // Custom CSR-mapped popcount: write operand to CSR 0x800, read
        // result from CSR 0x801. Platform integrators bind these CSRs to silicon.
        unsafe {
            let result: u32;
            core::arch::asm!(
                "csrw 0x800, {src}",
                "csrr {dst}, 0x801",
                src = in(reg) word,
                dst = out(reg) result,
            );
            result
        }
    }
    #[cfg(not(feature = "custom-popcount"))]
    {
        word.count_ones()
    }
}

/// SC multiply (bitwise AND).
#[inline(always)]
pub fn sc_and(a: u32, b: u32) -> u32 {
    a & b
}

/// SC MUX (scaled addition): `(a & sel) | (b & !sel)`.
#[inline(always)]
pub fn sc_mux(a: u32, b: u32, sel: u32) -> u32 {
    (a & sel) | (b & !sel)
}

/// SC saturating subtraction: `a & !b`.
#[inline(always)]
pub fn sc_sub(a: u32, b: u32) -> u32 {
    a & !b
}

/// SC XOR (HDC bind).
#[inline(always)]
pub fn sc_xor(a: u32, b: u32) -> u32 {
    a ^ b
}

/// Popcount over a packed word slice.
#[inline]
pub fn popcount_slice(words: &[u32]) -> u32 {
    let mut total: u32 = 0;
    for &w in words {
        total = total.wrapping_add(popcount32(w));
    }
    total
}

/// SC AND over two packed word slices into `out`.
///
/// # Panics
/// Panics (in debug) if slice lengths differ.
#[inline]
pub fn and_packed(a: &[u32], b: &[u32], out: &mut [u32]) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), out.len());
    let n = a.len();
    let mut i = 0;

    // Process 4 words at a time for better ILP
    while i + 4 <= n {
        out[i] = a[i] & b[i];
        out[i + 1] = a[i + 1] & b[i + 1];
        out[i + 2] = a[i + 2] & b[i + 2];
        out[i + 3] = a[i + 3] & b[i + 3];
        i += 4;
    }
    while i < n {
        out[i] = a[i] & b[i];
        i += 1;
    }
}

/// SC MUX over two packed word slices with a select bitstream.
#[inline]
pub fn mux_packed(a: &[u32], b: &[u32], sel: &[u32], out: &mut [u32]) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), sel.len());
    debug_assert_eq!(a.len(), out.len());
    for i in 0..a.len() {
        out[i] = (a[i] & sel[i]) | (b[i] & !sel[i]);
    }
}

/// Estimated probability from a packed bitstream.
///
/// `bit_length` is the logical length (may be < `words.len() * 32`).
pub fn probability(words: &[u32], bit_length: u32) -> f32 {
    if bit_length == 0 {
        return 0.0;
    }
    popcount_slice(words) as f32 / bit_length as f32
}

/// SCC between two packed u32 bitstreams (Alaghi & Hayes, 2013).
pub fn scc(a: &[u32], b: &[u32], bit_length: u32) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    if bit_length == 0 {
        return 0.0;
    }
    let n = bit_length as f32;
    let pa = popcount_slice(a) as f32 / n;
    let pb = popcount_slice(b) as f32 / n;

    let and_count: u32 = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| popcount32(x & y))
        .sum();
    let p_and = and_count as f32 / n;

    let num = p_and - (pa * pb);
    if num.abs() < 1e-7 {
        return 0.0;
    }
    let denom = if num > 0.0 {
        pa.min(pb) - (pa * pb)
    } else {
        (pa * pb) - (pa + pb - 1.0).max(0.0)
    };
    if denom.abs() < 1e-7 {
        0.0
    } else {
        (num / denom).clamp(-1.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_popcount32() {
        assert_eq!(popcount32(0), 0);
        assert_eq!(popcount32(u32::MAX), 32);
        assert_eq!(popcount32(0xAAAA_AAAA), 16);
    }

    #[test]
    fn test_sc_and() {
        assert_eq!(sc_and(0b1010, 0b1100), 0b1000);
    }

    #[test]
    fn test_sc_mux() {
        assert_eq!(sc_mux(0xFF, 0x00, 0x0F), 0x0F);
    }

    #[test]
    fn test_sc_sub() {
        assert_eq!(sc_sub(0b1110, 0b0110), 0b1000);
    }

    #[test]
    fn test_popcount_slice() {
        let words = [u32::MAX, u32::MAX];
        assert_eq!(popcount_slice(&words), 64);
    }

    #[test]
    fn test_and_packed() {
        let a = [0xAAAA_AAAAu32, 0xFFFF_FFFF];
        let b = [0x5555_5555u32, 0x0000_FFFF];
        let mut out = [0u32; 2];
        and_packed(&a, &b, &mut out);
        assert_eq!(out[0], 0);
        assert_eq!(out[1], 0x0000_FFFF);
    }

    #[test]
    fn test_mux_packed() {
        let a = [0xFFFF_FFFFu32];
        let b = [0x0000_0000u32];
        let s = [0x0000_FFFFu32];
        let mut out = [0u32; 1];
        mux_packed(&a, &b, &s, &mut out);
        assert_eq!(out[0], 0x0000_FFFF);
    }

    #[test]
    fn test_probability() {
        let words = [0xFFFF_FFFFu32]; // 32 ones
        assert!((probability(&words, 32) - 1.0).abs() < 1e-6);
        let words2 = [0u32];
        assert!((probability(&words2, 32) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_scc_identical() {
        let a = [0xAAAA_AAAAu32];
        let corr = scc(&a, &a, 32);
        assert!((corr - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_scc_anticorrelated() {
        let a = [0xAAAA_AAAAu32];
        let b = [0x5555_5555u32];
        let corr = scc(&a, &b, 32);
        assert!((corr - (-1.0)).abs() < 0.01);
    }
}
