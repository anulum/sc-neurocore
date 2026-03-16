// SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li

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
    let mut acc = vdupq_n_f64(0.0);
    let mut chunks_a = a[..len].chunks_exact(2);
    let mut chunks_b = b[..len].chunks_exact(2);

    for (ca, cb) in chunks_a.by_ref().zip(chunks_b.by_ref()) {
        let va = vld1q_f64(ca.as_ptr());
        let vb = vld1q_f64(cb.as_ptr());
        acc = vfmaq_f64(acc, va, vb);
    }

    let mut sum = vgetq_lane_f64(acc, 0) + vgetq_lane_f64(acc, 1);
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
    let mut vmax = vdupq_n_f64(f64::NEG_INFINITY);
    let mut chunks = a.chunks_exact(2);

    for chunk in chunks.by_ref() {
        let va = vld1q_f64(chunk.as_ptr());
        vmax = vmaxq_f64(vmax, va);
    }

    let mut m = f64::max(vgetq_lane_f64(vmax, 0), vgetq_lane_f64(vmax, 1));
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
    let mut acc = vdupq_n_f64(0.0);
    let mut chunks = a.chunks_exact(2);

    for chunk in chunks.by_ref() {
        let va = vld1q_f64(chunk.as_ptr());
        acc = vaddq_f64(acc, va);
    }

    let mut sum = vgetq_lane_f64(acc, 0) + vgetq_lane_f64(acc, 1);
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
    let mut chunks = y.chunks_exact_mut(2);

    for chunk in chunks.by_ref() {
        let vy = vld1q_f64(chunk.as_ptr());
        let scaled = vmulq_f64(vy, valpha);
        vst1q_f64(chunk.as_mut_ptr(), scaled);
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
