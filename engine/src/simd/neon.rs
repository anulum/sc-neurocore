// SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Neon

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
/// Count set bits in 64-bit words using ARM NEON instructions.
///
/// # Safety
/// Caller must ensure the current CPU supports `neon`.
pub unsafe fn popcount_neon(data: &[u64]) -> u64 {
    let mut total = 0_u64;
    let mut chunks = data.chunks_exact(2);

    for chunk in &mut chunks {
        let v = vld1q_u8(chunk.as_ptr() as *const u8);
        let byte_counts = vcntq_u8(v);
        let sum16 = vpaddlq_u8(byte_counts);
        let sum32 = vpaddlq_u16(sum16);
        let sum64 = vpaddlq_u32(sum32);
        total += vgetq_lane_u64(sum64, 0) + vgetq_lane_u64(sum64, 1);
    }

    total + crate::bitstream::popcount_words_portable(chunks.remainder())
}

#[cfg(not(target_arch = "aarch64"))]
/// Fallback popcount when NEON is unavailable on this architecture.
///
/// # Safety
/// This function is marked unsafe for API parity with the NEON variant.
pub unsafe fn popcount_neon(data: &[u64]) -> u64 {
    crate::bitstream::popcount_words_portable(data)
}

// --- f64 SIMD operations (NEON: 2-wide f64, AArch64 only) ---

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
/// Dot product of two f64 slices using NEON.
///
/// # Safety
/// Caller must ensure the current CPU supports `neon`.
pub unsafe fn dot_f64_neon(a: &[f64], b: &[f64]) -> f64 {
    let len = a.len().min(b.len());
    let mut acc0 = vdupq_n_f64(0.0);
    let mut acc1 = vdupq_n_f64(0.0);
    let mut acc2 = vdupq_n_f64(0.0);
    let mut acc3 = vdupq_n_f64(0.0);
    
    let mut chunks_a = a[..len].chunks_exact(8);
    let mut chunks_b = b[..len].chunks_exact(8);

    for (ca, cb) in chunks_a.by_ref().zip(chunks_b.by_ref()) {
        acc0 = vfmaq_f64(acc0, vld1q_f64(ca.as_ptr()), vld1q_f64(cb.as_ptr()));
        acc1 = vfmaq_f64(acc1, vld1q_f64(ca.as_ptr().add(2)), vld1q_f64(cb.as_ptr().add(2)));
        acc2 = vfmaq_f64(acc2, vld1q_f64(ca.as_ptr().add(4)), vld1q_f64(cb.as_ptr().add(4)));
        acc3 = vfmaq_f64(acc3, vld1q_f64(ca.as_ptr().add(6)), vld1q_f64(cb.as_ptr().add(6)));
    }

    acc0 = vaddq_f64(acc0, acc1);
    acc2 = vaddq_f64(acc2, acc3);
    acc0 = vaddq_f64(acc0, acc2);

    let mut sum = vgetq_lane_f64(acc0, 0) + vgetq_lane_f64(acc0, 1);
    for (&ra, &rb) in chunks_a.remainder().iter().zip(chunks_b.remainder()) {
        sum += ra * rb;
    }
    sum
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
/// Maximum of f64 slice using NEON.
///
/// # Safety
/// Caller must ensure the current CPU supports `neon`.
pub unsafe fn max_f64_neon(a: &[f64]) -> f64 {
    if a.is_empty() {
        return f64::NEG_INFINITY;
    }
    let mut vmax0 = vdupq_n_f64(f64::NEG_INFINITY);
    let mut vmax1 = vdupq_n_f64(f64::NEG_INFINITY);
    let mut vmax2 = vdupq_n_f64(f64::NEG_INFINITY);
    let mut vmax3 = vdupq_n_f64(f64::NEG_INFINITY);
    
    let mut chunks = a.chunks_exact(8);
    for chunk in chunks.by_ref() {
        vmax0 = vmaxq_f64(vmax0, vld1q_f64(chunk.as_ptr()));
        vmax1 = vmaxq_f64(vmax1, vld1q_f64(chunk.as_ptr().add(2)));
        vmax2 = vmaxq_f64(vmax2, vld1q_f64(chunk.as_ptr().add(4)));
        vmax3 = vmaxq_f64(vmax3, vld1q_f64(chunk.as_ptr().add(6)));
    }

    vmax0 = vmaxq_f64(vmax0, vmax1);
    vmax2 = vmaxq_f64(vmax2, vmax3);
    vmax0 = vmaxq_f64(vmax0, vmax2);

    let mut m = f64::max(vgetq_lane_f64(vmax0, 0), vgetq_lane_f64(vmax0, 1));
    for &v in chunks.remainder() {
        m = m.max(v);
    }
    m
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
/// Sum of f64 slice using NEON.
///
/// # Safety
/// Caller must ensure the current CPU supports `neon`.
pub unsafe fn sum_f64_neon(a: &[f64]) -> f64 {
    let mut acc0 = vdupq_n_f64(0.0);
    let mut acc1 = vdupq_n_f64(0.0);
    let mut acc2 = vdupq_n_f64(0.0);
    let mut acc3 = vdupq_n_f64(0.0);
    
    let mut chunks = a.chunks_exact(8);
    for chunk in chunks.by_ref() {
        acc0 = vaddq_f64(acc0, vld1q_f64(chunk.as_ptr()));
        acc1 = vaddq_f64(acc1, vld1q_f64(chunk.as_ptr().add(2)));
        acc2 = vaddq_f64(acc2, vld1q_f64(chunk.as_ptr().add(4)));
        acc3 = vaddq_f64(acc3, vld1q_f64(chunk.as_ptr().add(6)));
    }

    acc0 = vaddq_f64(acc0, acc1);
    acc2 = vaddq_f64(acc2, acc3);
    acc0 = vaddq_f64(acc0, acc2);

    let mut sum = vgetq_lane_f64(acc0, 0) + vgetq_lane_f64(acc0, 1);
    for &v in chunks.remainder() {
        sum += v;
    }
    sum
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
/// Scale f64 slice in-place: y[i] *= alpha, using NEON.
///
/// # Safety
/// Caller must ensure the current CPU supports `neon`.
pub unsafe fn scale_f64_neon(alpha: f64, y: &mut [f64]) {
    let valpha = vdupq_n_f64(alpha);
    let mut chunks = y.chunks_exact_mut(8);

    for chunk in chunks.by_ref() {
        vst1q_f64(chunk.as_mut_ptr(), vmulq_f64(vld1q_f64(chunk.as_ptr()), valpha));
        vst1q_f64(chunk.as_mut_ptr().add(2), vmulq_f64(vld1q_f64(chunk.as_ptr().add(2)), valpha));
        vst1q_f64(chunk.as_mut_ptr().add(4), vmulq_f64(vld1q_f64(chunk.as_ptr().add(4)), valpha));
        vst1q_f64(chunk.as_mut_ptr().add(6), vmulq_f64(vld1q_f64(chunk.as_ptr().add(6)), valpha));
    }

    for v in chunks.into_remainder() {
        *v *= alpha;
    }
}

#[cfg(not(target_arch = "aarch64"))]
/// # Safety
/// Fallback for non-AArch64; unsafe for API parity.
pub unsafe fn dot_f64_neon(a: &[f64], b: &[f64]) -> f64 {
    let len = a.len().min(b.len());
    a[..len].iter().zip(&b[..len]).map(|(&x, &y)| x * y).sum()
}

#[cfg(not(target_arch = "aarch64"))]
/// # Safety
/// Fallback for non-AArch64; unsafe for API parity.
pub unsafe fn max_f64_neon(a: &[f64]) -> f64 {
    a.iter().copied().fold(f64::NEG_INFINITY, f64::max)
}

#[cfg(not(target_arch = "aarch64"))]
/// # Safety
/// Fallback for non-AArch64; unsafe for API parity.
pub unsafe fn sum_f64_neon(a: &[f64]) -> f64 {
    a.iter().sum()
}

#[cfg(not(target_arch = "aarch64"))]
/// # Safety
/// Fallback for non-AArch64; unsafe for API parity.
pub unsafe fn scale_f64_neon(alpha: f64, y: &mut [f64]) {
    for v in y.iter_mut() {
        *v *= alpha;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_popcount_empty() {
        assert_eq!(unsafe { popcount_neon(&[]) }, 0);
    }

    #[test]
    fn test_popcount_known_values() {
        // 0xFFFF_FFFF_FFFF_FFFF has 64 set bits
        assert_eq!(unsafe { popcount_neon(&[u64::MAX]) }, 64);
        assert_eq!(unsafe { popcount_neon(&[0]) }, 0);
        assert_eq!(unsafe { popcount_neon(&[1]) }, 1);
        assert_eq!(unsafe { popcount_neon(&[0b1010_1010]) }, 4);
    }

    #[test]
    fn test_popcount_multiple_words() {
        let data = [u64::MAX, u64::MAX, 1];
        assert_eq!(unsafe { popcount_neon(&data) }, 129); // 64+64+1
    }

    #[test]
    fn test_dot_f64_simple() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        let result = unsafe { dot_f64_neon(&a, &b) };
        assert!((result - 32.0).abs() < 1e-10); // 1*4 + 2*5 + 3*6 = 32
    }

    #[test]
    fn test_dot_f64_empty() {
        let result = unsafe { dot_f64_neon(&[], &[]) };
        assert!((result - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_dot_f64_mismatched_length() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [1.0, 1.0];
        let result = unsafe { dot_f64_neon(&a, &b) };
        assert!((result - 3.0).abs() < 1e-10); // 1*1 + 2*1
    }

    #[test]
    fn test_max_f64() {
        let a = [1.0, 5.0, 3.0, 2.0, 4.0];
        assert!((unsafe { max_f64_neon(&a) } - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_max_f64_empty() {
        assert!(unsafe { max_f64_neon(&[]) } == f64::NEG_INFINITY);
    }

    #[test]
    fn test_max_f64_negative() {
        let a = [-5.0, -1.0, -3.0];
        assert!((unsafe { max_f64_neon(&a) } - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_sum_f64() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((unsafe { sum_f64_neon(&a) } - 15.0).abs() < 1e-10);
    }

    #[test]
    fn test_sum_f64_empty() {
        assert!((unsafe { sum_f64_neon(&[]) } - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_scale_f64() {
        let mut y = [1.0, 2.0, 3.0, 4.0, 5.0];
        unsafe { scale_f64_neon(2.0, &mut y) };
        assert!((y[0] - 2.0).abs() < 1e-10);
        assert!((y[4] - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_scale_f64_zero() {
        let mut y = [1.0, 2.0, 3.0];
        unsafe { scale_f64_neon(0.0, &mut y) };
        assert!(y.iter().all(|&v| v == 0.0));
    }
}
