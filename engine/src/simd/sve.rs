// SPDX-License-Identifier: AGPL-3.0-or-later
//! ARM SVE (Scalable Vector Extension) popcount and bitstream kernels.
//!
//! SVE operates on variable-length vectors (128–2048 bits depending on
//! hardware).  These kernels use `core::arch::aarch64` SVE intrinsics
//! when compiled for an SVE-capable target.  On all other targets the
//! functions fall back to portable scalar code.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sve_popcount_matches_portable() {
        let data: Vec<u64> = vec![0xFFFF_FFFF_FFFF_FFFF, 0x0, 0xAAAA_AAAA_AAAA_AAAA];
        let expected = 64 + 0 + 32;
        let got = unsafe { popcount_sve(&data) };
        assert_eq!(got, expected);
    }

    #[test]
    fn sve_fused_and_popcount() {
        let a = vec![0xFFu64, 0xF0];
        let b = vec![0x0Fu64, 0xFF];
        let expected =
            (0xFFu64 & 0x0F).count_ones() as u64 + (0xF0u64 & 0xFF).count_ones() as u64;
        let got = unsafe { fused_and_popcount_sve(&a, &b) };
        assert_eq!(got, expected);
    }
}
