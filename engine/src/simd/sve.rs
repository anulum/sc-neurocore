// SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li

//! Scalar fallback for ARM SVE targets.
//! Hardware SVE intrinsics are not yet implemented; all operations
//! use the portable scalar path from `super::mod.rs`.
//!
//! SVE operates on variable-length vectors (128–2048 bits depending on
//! hardware).  When Rust stabilises `core::arch::aarch64` SVE intrinsics,
//! replace the bodies below with predicated vector loops.
//!
//! Build with:
//!   RUSTFLAGS="-C target-feature=+sve" cargo build --target aarch64-unknown-linux-gnu

/// Pack u8 bit array into u64 words using SVE wide loads.
///
/// # Safety
/// Caller must ensure the target CPU supports SVE.
#[cfg(all(target_arch = "aarch64", target_feature = "sve"))]
pub unsafe fn pack_sve(bits: &[u8]) -> Vec<u64> {
    // SVE pack: process VL bytes at a time using svld1_u8 + bit gathering.
    // Current implementation: portable fallback (SVE intrinsics are
    // nightly-only as of Rust 1.82).  Replace with svptrue_b8 / svld1_u8 /
    // svlsr_n_u8_x / svorr_u8_x pipeline when stabilised.
    crate::bitstream::pack_fast(bits).data
}

/// Pack u8 bit array into u64 words (portable fallback).
///
/// # Safety
/// No hardware requirements in fallback mode.
#[cfg(not(all(target_arch = "aarch64", target_feature = "sve")))]
pub unsafe fn pack_sve(bits: &[u8]) -> Vec<u64> {
    crate::bitstream::pack_fast(bits).data
}

/// Count set bits using SVE BCNT instruction.
///
/// # Safety
/// Caller must ensure the target CPU supports SVE.
#[cfg(all(target_arch = "aarch64", target_feature = "sve"))]
pub unsafe fn popcount_sve(data: &[u64]) -> u64 {
    // SVE provides svcnt_u64_x (BCNT) for per-element popcount.
    // Pending stabilisation of core::arch::aarch64::sve intrinsics.
    crate::bitstream::popcount_words_portable(data)
}

/// Count set bits (portable fallback).
///
/// # Safety
/// No hardware requirements in fallback mode.
#[cfg(not(all(target_arch = "aarch64", target_feature = "sve")))]
pub unsafe fn popcount_sve(data: &[u64]) -> u64 {
    crate::bitstream::popcount_words_portable(data)
}

/// Fused AND + popcount using SVE.
///
/// # Safety
/// Caller must ensure the target CPU supports SVE.
#[cfg(all(target_arch = "aarch64", target_feature = "sve"))]
pub unsafe fn fused_and_popcount_sve(a: &[u64], b: &[u64]) -> u64 {
    // SVE: svand_u64_x + svcnt_u64_x in a single predicated loop.
    // Pending intrinsic stabilisation.
    let len = a.len().min(b.len());
    a[..len]
        .iter()
        .zip(&b[..len])
        .map(|(&wa, &wb)| (wa & wb).count_ones() as u64)
        .sum()
}

/// Fused AND + popcount (portable fallback).
///
/// # Safety
/// No hardware requirements in fallback mode.
#[cfg(not(all(target_arch = "aarch64", target_feature = "sve")))]
pub unsafe fn fused_and_popcount_sve(a: &[u64], b: &[u64]) -> u64 {
    let len = a.len().min(b.len());
    a[..len]
        .iter()
        .zip(&b[..len])
        .map(|(&wa, &wb)| (wa & wb).count_ones() as u64)
        .sum()
}

/// Fused XOR + popcount using SVE.
///
/// # Safety
/// No hardware requirements (portable implementation).
pub unsafe fn fused_xor_popcount_sve(a: &[u64], b: &[u64]) -> u64 {
    let len = a.len().min(b.len());
    a[..len]
        .iter()
        .zip(&b[..len])
        .map(|(&wa, &wb)| (wa ^ wb).count_ones() as u64)
        .sum()
}

// --- f64 operations (portable fallback, SVE intrinsics pending stabilisation) ---

/// # Safety
/// No hardware requirements (portable implementation).
pub unsafe fn dot_f64_sve(a: &[f64], b: &[f64]) -> f64 {
    let len = a.len().min(b.len());
    a[..len].iter().zip(&b[..len]).map(|(&x, &y)| x * y).sum()
}

/// # Safety
/// No hardware requirements (portable implementation).
pub unsafe fn max_f64_sve(a: &[f64]) -> f64 {
    a.iter().copied().fold(f64::NEG_INFINITY, f64::max)
}

/// # Safety
/// No hardware requirements (portable implementation).
pub unsafe fn sum_f64_sve(a: &[f64]) -> f64 {
    a.iter().sum()
}

/// # Safety
/// No hardware requirements (portable implementation).
pub unsafe fn scale_f64_sve(alpha: f64, y: &mut [f64]) {
    for v in y.iter_mut() {
        *v *= alpha;
    }
}

/// Hamming distance between two packed bitstream slices.
///
/// # Safety
/// No hardware requirements (portable implementation).
pub unsafe fn hamming_distance_sve(a: &[u64], b: &[u64]) -> u64 {
    fused_xor_popcount_sve(a, b)
}

/// In-place softmax (portable fallback for SVE).
///
/// # Safety
/// No hardware requirements (portable implementation).
pub unsafe fn softmax_inplace_f64_sve(scores: &mut [f64]) {
    if scores.is_empty() {
        return;
    }
    let max_val = max_f64_sve(scores);
    for s in scores.iter_mut() {
        *s = (*s - max_val).exp();
    }
    let exp_sum = sum_f64_sve(scores);
    if exp_sum > 0.0 {
        scale_f64_sve(1.0 / exp_sum, scores);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sve_popcount_matches_portable() {
        let data: Vec<u64> = vec![0xFFFF_FFFF_FFFF_FFFF, 0x0, 0xAAAA_AAAA_AAAA_AAAA];
        let expected = 64 + 32;
        let got = unsafe { popcount_sve(&data) };
        assert_eq!(got, expected);
    }

    #[test]
    fn sve_softmax_sums_to_one() {
        let mut scores: Vec<f64> = (0..20).map(|i| (i as f64 * 0.5) - 5.0).collect();
        unsafe { super::softmax_inplace_f64_sve(&mut scores) };
        let sum: f64 = scores.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
        assert!(scores.iter().all(|&s| s >= 0.0));
    }

    #[test]
    fn sve_hamming_distance() {
        let a = vec![0xFFu64, 0x00];
        let b = vec![0x0Fu64, 0x00];
        let expected = (0xFFu64 ^ 0x0F).count_ones() as u64;
        let got = unsafe { super::hamming_distance_sve(&a, &b) };
        assert_eq!(got, expected);
    }

    #[test]
    fn sve_fused_and_popcount() {
        let a = vec![0xFFu64, 0xF0];
        let b = vec![0x0Fu64, 0xFF];
        let expected = (0xFFu64 & 0x0F).count_ones() as u64 + (0xF0u64 & 0xFF).count_ones() as u64;
        let got = unsafe { fused_and_popcount_sve(&a, &b) };
        assert_eq!(got, expected);
    }
}
