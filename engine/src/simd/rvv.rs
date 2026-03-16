// SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li

//! Scalar fallback for RISC-V Vector (RVV) targets.
//! Hardware RVV intrinsics are not yet implemented; all operations
//! use the portable scalar path from `super::mod.rs`.
//!
//! RVV 1.0 provides variable-length SIMD (VLEN = 128–16384 bits).
//! When Rust stabilises `core::arch::riscv64` vector intrinsics,
//! replace the bodies below with vl-strided vector loops.
//!
//! Build with:
//!   RUSTFLAGS="-C target-feature=+v" cargo build --target riscv64gc-unknown-linux-gnu
//!
//! Cross-compile without hardware:
//!   cargo install cross
//!   cross build --target riscv64gc-unknown-linux-gnu --release

/// Pack u8 bit array into u64 words using RVV vector loads.
///
/// # Safety
/// Caller must ensure the target CPU supports RVV 1.0.
#[cfg(all(target_arch = "riscv64", target_feature = "v"))]
pub unsafe fn pack_rvv(bits: &[u8]) -> Vec<u64> {
    // RVV: vle8_v_u8m1 + bit gathering via vslide/vmask operations.
    // Pending stabilisation of core::arch::riscv64 vector intrinsics.
    crate::bitstream::pack_fast(bits).data
}

/// Pack u8 bit array into u64 words (portable fallback).
///
/// # Safety
/// No hardware requirements in fallback mode.
#[cfg(not(all(target_arch = "riscv64", target_feature = "v")))]
pub unsafe fn pack_rvv(bits: &[u8]) -> Vec<u64> {
    crate::bitstream::pack_fast(bits).data
}

/// Count set bits using RVV VCPOP instruction.
///
/// # Safety
/// Caller must ensure the target CPU supports RVV 1.0.
#[cfg(all(target_arch = "riscv64", target_feature = "v"))]
pub unsafe fn popcount_rvv(data: &[u64]) -> u64 {
    // RVV: vle64_v_u64m1 + vcpop.m for per-element popcount.
    // Pending stabilisation of core::arch::riscv64 vector intrinsics.
    crate::bitstream::popcount_words_portable(data)
}

/// Count set bits (portable fallback).
///
/// # Safety
/// No hardware requirements in fallback mode.
#[cfg(not(all(target_arch = "riscv64", target_feature = "v")))]
pub unsafe fn popcount_rvv(data: &[u64]) -> u64 {
    crate::bitstream::popcount_words_portable(data)
}

/// Fused AND + popcount using RVV.
///
/// # Safety
/// Caller must ensure the target CPU supports RVV 1.0.
#[cfg(all(target_arch = "riscv64", target_feature = "v"))]
pub unsafe fn fused_and_popcount_rvv(a: &[u64], b: &[u64]) -> u64 {
    // RVV: vand.vv + vcpop.m in a single vl-strided loop.
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
#[cfg(not(all(target_arch = "riscv64", target_feature = "v")))]
pub unsafe fn fused_and_popcount_rvv(a: &[u64], b: &[u64]) -> u64 {
    let len = a.len().min(b.len());
    a[..len]
        .iter()
        .zip(&b[..len])
        .map(|(&wa, &wb)| (wa & wb).count_ones() as u64)
        .sum()
}

/// Fused XOR + popcount using RVV.
///
/// # Safety
/// No hardware requirements (portable implementation).
pub unsafe fn fused_xor_popcount_rvv(a: &[u64], b: &[u64]) -> u64 {
    let len = a.len().min(b.len());
    a[..len]
        .iter()
        .zip(&b[..len])
        .map(|(&wa, &wb)| (wa ^ wb).count_ones() as u64)
        .sum()
}

// --- f64 operations (portable fallback, RVV f64 intrinsics pending stabilisation) ---

/// # Safety
/// No hardware requirements (portable implementation).
pub unsafe fn dot_f64_rvv(a: &[f64], b: &[f64]) -> f64 {
    let len = a.len().min(b.len());
    a[..len].iter().zip(&b[..len]).map(|(&x, &y)| x * y).sum()
}

/// # Safety
/// No hardware requirements (portable implementation).
pub unsafe fn max_f64_rvv(a: &[f64]) -> f64 {
    a.iter().copied().fold(f64::NEG_INFINITY, f64::max)
}

/// # Safety
/// No hardware requirements (portable implementation).
pub unsafe fn sum_f64_rvv(a: &[f64]) -> f64 {
    a.iter().sum()
}

/// # Safety
/// No hardware requirements (portable implementation).
pub unsafe fn scale_f64_rvv(alpha: f64, y: &mut [f64]) {
    for v in y.iter_mut() {
        *v *= alpha;
    }
}

/// Hamming distance between two packed bitstream slices.
///
/// # Safety
/// No hardware requirements (portable implementation).
pub unsafe fn hamming_distance_rvv(a: &[u64], b: &[u64]) -> u64 {
    fused_xor_popcount_rvv(a, b)
}

/// In-place softmax (portable fallback for RVV).
///
/// # Safety
/// No hardware requirements (portable implementation).
pub unsafe fn softmax_inplace_f64_rvv(scores: &mut [f64]) {
    if scores.is_empty() {
        return;
    }
    let max_val = max_f64_rvv(scores);
    for s in scores.iter_mut() {
        *s = (*s - max_val).exp();
    }
    let exp_sum = sum_f64_rvv(scores);
    if exp_sum > 0.0 {
        scale_f64_rvv(1.0 / exp_sum, scores);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rvv_popcount_matches_portable() {
        let data: Vec<u64> = vec![0xFFFF_FFFF_FFFF_FFFF, 0x0, 0xAAAA_AAAA_AAAA_AAAA];
        let expected = 64 + 32;
        let got = unsafe { popcount_rvv(&data) };
        assert_eq!(got, expected);
    }

    #[test]
    fn rvv_softmax_sums_to_one() {
        let mut scores: Vec<f64> = (0..20).map(|i| (i as f64 * 0.5) - 5.0).collect();
        unsafe { super::softmax_inplace_f64_rvv(&mut scores) };
        let sum: f64 = scores.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
        assert!(scores.iter().all(|&s| s >= 0.0));
    }

    #[test]
    fn rvv_hamming_distance() {
        let a = vec![0xFFu64, 0x00];
        let b = vec![0x0Fu64, 0x00];
        let expected = (0xFFu64 ^ 0x0F).count_ones() as u64;
        let got = unsafe { super::hamming_distance_rvv(&a, &b) };
        assert_eq!(got, expected);
    }

    #[test]
    fn rvv_fused_and_popcount() {
        let a = vec![0xFFu64, 0xF0];
        let b = vec![0x0Fu64, 0xFF];
        let expected = (0xFFu64 & 0x0F).count_ones() as u64 + (0xF0u64 & 0xFF).count_ones() as u64;
        let got = unsafe { fused_and_popcount_rvv(&a, &b) };
        assert_eq!(got, expected);
    }
}
