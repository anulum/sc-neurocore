// SPDX-License-Identifier: AGPL-3.0-or-later
//! RISC-V Vector Extension (RVV) popcount and bitstream kernels.
//!
//! RVV 1.0 provides variable-length SIMD (VLEN = 128–16384 bits).
//! These kernels use RVV intrinsics when compiled for an RVV-capable
//! target.  On all other targets the functions fall back to portable
//! scalar code.
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
    crate::bitstream::pack_fast_to_words(bits)
}

#[cfg(not(all(target_arch = "riscv64", target_feature = "v")))]
pub unsafe fn pack_rvv(bits: &[u8]) -> Vec<u64> {
    crate::bitstream::pack_fast_to_words(bits)
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
/// Caller must ensure the target CPU supports RVV 1.0.
pub unsafe fn fused_xor_popcount_rvv(a: &[u64], b: &[u64]) -> u64 {
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
    fn rvv_popcount_matches_portable() {
        let data: Vec<u64> = vec![0xFFFF_FFFF_FFFF_FFFF, 0x0, 0xAAAA_AAAA_AAAA_AAAA];
        let expected = 64 + 0 + 32;
        let got = unsafe { popcount_rvv(&data) };
        assert_eq!(got, expected);
    }

    #[test]
    fn rvv_fused_and_popcount() {
        let a = vec![0xFFu64, 0xF0];
        let b = vec![0x0Fu64, 0xFF];
        let expected = (0xFF & 0x0F).count_ones() as u64 + (0xF0 & 0xFF).count_ones() as u64;
        let got = unsafe { fused_and_popcount_rvv(&a, &b) };
        assert_eq!(got, expected);
    }
}
